"""Engine-core probes for pinned vLLM decode rounds."""
from __future__ import annotations

import ast
import inspect
import time
from textwrap import dedent


_ROLLBACK = ast.dump(ast.parse(dedent("""
    if not output_is_stale:
        if request.num_computed_tokens > 0:
            request.num_computed_tokens -= num_rejected
        if request.num_output_placeholders > 0:
            request.num_output_placeholders -= num_rejected
""")).body[0])


def _instrument_rollback(original):
    """Observe the rollback and the round counts before native output truncation."""
    source, lineno = inspect.getsourcelines(original)
    tree = ast.parse(dedent("".join(source)))

    class Instrument(ast.NodeTransformer):
        matches = 0

        def visit_If(self, node):
            if ast.dump(node) != _ROLLBACK:
                return self.generic_visit(node)
            self.matches += 1
            before = ast.parse(dedent("""
                _simverse_start = None
                if (self._simverse_rollback is not None and not output_is_stale
                        and num_rejected > 0
                        and (request.num_computed_tokens > 0 or request.num_output_placeholders > 0)):
                    _simverse_start = self._simverse_rollback.now()
            """)).body
            after = ast.parse(dedent("""
                if _simverse_start is not None:
                    self._simverse_rollback.record(req_id, num_rejected, _simverse_start)
                if self._simverse_rollback is not None:
                    self._simverse_rollback.note_round(
                        req_id, request.num_output_tokens, num_draft_tokens,
                        num_verified, num_accepted, num_sampled,
                    )
            """)).body
            for stmt in before + after:
                ast.increment_lineno(stmt, node.lineno - stmt.lineno)
                ast.copy_location(stmt, node)
            return [*before, node, *after]

    instrument = Instrument()
    instrument.visit(tree)
    if instrument.matches != 1:
        raise ValueError("固定调度器的拒绝回退代码块缺失或不唯一")
    ast.increment_lineno(ast.fix_missing_locations(tree), lineno - 1)
    namespace = {}
    exec(compile(tree, original.__code__.co_filename, "exec"), original.__globals__, namespace)
    return namespace[original.__name__]


class _SchedulerCapture:
    def __init__(self, now=time.monotonic_ns):
        self.now = now
        self.started = now()
        self.steps = 0
        self.rollbacks = []
        self.rounds = []
        self.pending_rounds = {}

    def record(self, request_id, rejected, start):
        end = self.now()
        self.rollbacks.append({"step": self.steps, "request_id": request_id, "rejected_tokens": rejected,
                               "start_ns": start, "end_ns": end})

    def note_round(self, request_id, token_start, proposed, verified, accepted, num_sampled):
        round_row = {
            "step": self.steps, "backend_request_id": request_id, "token_start": token_start,
            "proposed": proposed, "verified": verified, "accepted": accepted,
            "target_offset": accepted if num_sampled == 1 else None, "backend_token_ids": None,
        }
        self.pending_rounds[request_id] = round_row
        self.rounds.append(round_row)

    def finalize_round_outputs(self, outputs):
        if self.pending_rounds:
            for batch in outputs.values():
                for output in batch.outputs:
                    row = self.pending_rounds.get(output.request_id)
                    if row is not None:
                        row["backend_token_ids"] = list(output.new_token_ids)
                        if output.finish_reason is not None:
                            row["backend_finish_reason"] = output.finish_reason.name.lower()
            self.pending_rounds.clear()

    def finish(self, reason):
        return {"window": {"start_ns": self.started, "end_ns": self.now()},
                "step_count": self.steps, "rollback_count": len(self.rollbacks) if reason is None else None,
                "rollbacks": self.rollbacks, "rounds": self.rounds, "reason": reason}


def _scheduler_trace_control(core, start):
    # The pinned engine-core utility queue handles this in the scheduler process.
    scheduler = core.scheduler
    if start:
        scheduler._simverse_rollback = _SchedulerCapture()
        return None
    capture = scheduler._simverse_rollback
    scheduler._simverse_rollback = None
    return capture.finish(scheduler._simverse_rollback_reason)


class TraceScheduler:
    """Construct the native synchronous or asynchronous scheduler with the timing hook."""

    def __new__(cls, *, vllm_config, **kwargs):
        from vllm.v1.core.sched.async_scheduler import AsyncScheduler
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.v1.engine.core import EngineCore

        base = AsyncScheduler if vllm_config.scheduler_config.async_scheduling else Scheduler
        update = base.update_from_output
        reason = None
        try:
            update = _instrument_rollback(update)
        except (OSError, TypeError, ValueError, SyntaxError) as exc:
            reason = f"CPU scheduler 投机轮挂点不可用 ({type(exc).__name__})"

        class InstrumentedScheduler(base):
            _simverse_rollback = None
            _simverse_rollback_reason = reason

            def update_from_output(self, scheduler_output, model_runner_output):
                capture = self._simverse_rollback
                if capture is not None and scheduler_output.total_num_scheduled_tokens > 0:
                    capture.steps += 1
                outputs = update(self, scheduler_output, model_runner_output)
                if capture is not None:
                    capture.finalize_round_outputs(outputs)
                return outputs

        EngineCore.simverse_scheduler_trace = _scheduler_trace_control
        return InstrumentedScheduler(vllm_config=vllm_config, **kwargs)
