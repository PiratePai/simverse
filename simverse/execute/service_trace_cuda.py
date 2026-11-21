"""Worker-local probes for the pinned vLLM GPUModelRunner."""
from __future__ import annotations

import time
from functools import wraps

from .request_trace import NS_PER_MS
from .service_trace_nvml import _DeviceMemorySampler


class _RankCapture:
    def __init__(self, worker, cuda, speculative, now=time.monotonic_ns, *, nvml=None):
        self.cuda = cuda
        self.rank = worker.rank
        self.device = worker.device
        self.requested_memory_bytes = getattr(worker, "requested_memory", None)
        self.kv_cache_budget_bytes = getattr(worker, "available_kv_cache_memory_bytes", None)
        self.now = now
        self.backend = cuda.get_allocator_backend()
        self.steps = []
        self.pending = []
        self.patches = []
        self.step = None
        self.verification = None
        self.phase_reason = None

        cuda.synchronize(self.device)
        self.start_ns = now()
        if self.backend == "native":
            cuda.reset_peak_memory_stats(self.device)
            self.start_bytes = self._memory_bytes()

        if speculative:
            self._instrument(worker.model_runner)
        uuid = str(getattr(cuda.get_device_properties(self.device), "uuid", "") or "") if nvml else ""
        self.device_memory = _DeviceMemorySampler(nvml, uuid, now)

    def _memory_bytes(self):
        return {name: getattr(self.cuda, f"memory_{name}")(self.device)
                for name in ("allocated", "reserved")}

    def _patch(self, obj, name, wrap):
        original = getattr(obj, name)
        setattr(obj, name, wraps(original)(wrap(original)))
        self.patches.append((obj, name, original))

    def _instrument(self, runner):
        self.runner = runner
        hooks = (
            (runner, "execute_model", self._execute),
            (runner, "sample_tokens", self._sample),
            (getattr(runner, "kv_connector", None), "pre_forward", self._verify_start),
            (getattr(runner, "kv_connector", None), "finish_forward", self._verify_end),
            (getattr(runner, "speculator", None), "propose", self._draft),
            (runner, "postprocess_sampled", self._recover),
        )
        missing = [name for obj, name, _ in hooks if not callable(getattr(obj, name, None))]
        if missing:
            self.phase_reason = "缺少 vLLM 轮内采集挂点：" + ", ".join(missing)
            return
        for obj, name, wrap in hooks:
            self._patch(obj, name, wrap)
        if callable(getattr(runner, "rejection_sampler", None)):
            self._patch(runner, "rejection_sampler", self._rejection_sampling)

    def _execute(self, original):
        def execute(scheduler_output, intermediate_tensors=None, dummy_run=False, *args, **kwargs):
            if dummy_run or not scheduler_output.total_num_scheduled_tokens:
                return original(scheduler_output, intermediate_tensors, dummy_run, *args, **kwargs)
            self.step = step = {
                "step": len(self.steps) + 1,
                "batch_request_ids": sorted(scheduler_output.num_scheduled_tokens),
                "verify_request_ids": sorted(rid for rid, tokens in
                                             scheduler_output.scheduled_spec_decode_tokens.items() if tokens),
                "draft_tokens_to_schedule": 0,
                "phases": {},
            }
            self.steps.append(step)
            output = original(scheduler_output, intermediate_tensors, dummy_run, *args, **kwargs)
            # This is the resolved count after skip_drafts, not the configured K.
            state = self.runner.execute_model_state
            if state is not None:
                step["draft_tokens_to_schedule"] = state.num_spec_tokens_to_schedule
            return output

        return execute

    def _sample(self, original):
        def sample(*args, **kwargs):
            try:
                return original(*args, **kwargs)
            finally:
                self.step = None

        return sample

    def _begin_span(self):
        stream = self.cuda.current_stream(self.device)
        event = self.cuda.Event(enable_timing=True)
        host_ns = self.now()
        event.record(stream)
        return stream, event, host_ns

    def _end_span(self, phase, start):
        stream, event, host_ns = start
        if stream != self.cuda.current_stream(self.device):
            return  # A stream change leaves this phase missing.
        end = self.cuda.Event(enable_timing=True)
        end.record(stream)
        self.pending.append((self.step, phase, stream.cuda_stream, event, end, host_ns, self.now()))

    def _verify_start(self, original):
        def start(*args, **kwargs):
            if self.step and self.step["verify_request_ids"]:
                self.verification = self._begin_span()
            return original(*args, **kwargs)

        return start

    def _verify_end(self, original):
        def end(*args, **kwargs):
            if self.verification is not None:
                self._end_span("verify", self.verification)
                self.verification = None
            return original(*args, **kwargs)

        return end

    def _record(self, phase, original, args, kwargs):
        start = self._begin_span()
        result = original(*args, **kwargs)
        self._end_span(phase, start)
        return result

    def _draft(self, original):
        def propose(*args, **kwargs):
            if self.step and self.step["draft_tokens_to_schedule"]:
                return self._record("draft", original, args, kwargs)
            return original(*args, **kwargs)

        return propose

    def _recover(self, original):
        def recover(*args, **kwargs):
            if self.step and self.step["verify_request_ids"]:
                return self._record("recovery", original, args, kwargs)
            return original(*args, **kwargs)

        return recover

    def _rejection_sampling(self, original):
        def sample(*args, **kwargs):
            if self.step and self.step["verify_request_ids"]:
                return self._record("rejection_sampling", original, args, kwargs)
            return original(*args, **kwargs)

        return sample

    def finish(self):
        try:
            try:
                self.cuda.synchronize(self.device)
                end_ns = self.now()
                memory = None
                if self.backend == "native":
                    end_bytes = self._memory_bytes()
                    memory = {name: {"start_bytes": start, "end_bytes": end_bytes[name],
                                     "peak_bytes": getattr(self.cuda, f"max_memory_{name}")(self.device)}
                              for name, start in self.start_bytes.items()}
            finally:
                device_memory = self.device_memory.finish()
            anchors = {}
            for step, phase, stream, start, end, host_start, host_end in self.pending:
                anchor = anchors.setdefault(stream, start)
                if phase in step["phases"]:
                    raise ValueError(f"rank {self.rank} 重复采集 step {step['step']} 的 {phase} 阶段")
                start_ns = round(anchor.elapsed_time(start) * NS_PER_MS)
                duration_ns = round(start.elapsed_time(end) * NS_PER_MS)
                step["phases"][phase] = {
                    "stream_id": str(stream), "start_ns": start_ns, "end_ns": start_ns + duration_ns,
                    "host_start_ns": host_start, "host_end_ns": host_end,
                }
            return {"rank": self.rank, "device_index": self.device.index, "allocator_backend": self.backend,
                    "requested_memory_bytes": self.requested_memory_bytes,
                    "kv_cache_budget_bytes": self.kv_cache_budget_bytes,
                    "window": {"start_ns": self.start_ns, "end_ns": end_ns}, "memory": memory,
                    "device_memory": device_memory, "steps": self.steps, "phase_reason": self.phase_reason}
        finally:
            for obj, name, original in reversed(self.patches):
                setattr(obj, name, original)


class ServiceTraceWorkerExtension:
    """Loaded by vLLM worker_extension_cls on all local TP ranks."""

    def simverse_begin_trace(self, speculative):
        from torch import cuda
        from vllm.utils.import_utils import import_pynvml

        try:
            nvml = import_pynvml()
        except ImportError:
            nvml = None
        self._simverse_capture = _RankCapture(self, cuda, speculative, nvml=nvml)

    def simverse_end_trace(self):
        try:
            return self._simverse_capture.finish()
        finally:
            del self._simverse_capture
