"""Sample whole physical-device memory from NVML, independently of PyTorch."""
from __future__ import annotations

import threading
import time

from .request_trace import NS_PER_S

SAMPLE_INTERVAL_NS = 20_000_000


def _physical_handle(nvml, cuda_uuid):
    if cuda_uuid.startswith("MIG-"):
        return nvml.nvmlDeviceGetDeviceHandleFromMigDeviceHandle(nvml.nvmlDeviceGetHandleByUUID(cuda_uuid))
    suffix = cuda_uuid.removeprefix("GPU-")
    try:
        return nvml.nvmlDeviceGetHandleByUUID(f"GPU-{suffix}")
    except nvml.NVMLError as exc:
        if exc.value not in (nvml.NVML_ERROR_NOT_FOUND, nvml.NVML_ERROR_INVALID_ARGUMENT):
            raise
        mig = nvml.nvmlDeviceGetHandleByUUID(f"MIG-{suffix}")
        return nvml.nvmlDeviceGetDeviceHandleFromMigDeviceHandle(mig)


class _DeviceMemorySampler:
    def __init__(self, nvml, cuda_uuid, now=time.monotonic_ns):
        self.nvml = nvml
        self.now = now
        self.device_uuid = None
        self.samples = []
        self.reason = None
        self.initialized = False
        self.thread = None
        self.stop = threading.Event()
        if nvml is None or not cuda_uuid:
            self.reason = "缺少 NVML 或 CUDA 设备 UUID"
            return
        try:
            nvml.nvmlInit()
            self.initialized = True
            self.handle = _physical_handle(nvml, cuda_uuid)
            uuid = nvml.nvmlDeviceGetUUID(self.handle)
            self.device_uuid = uuid.decode("ascii") if isinstance(uuid, bytes) else uuid
        except nvml.NVMLError as exc:
            self.reason = f"NVML 物理设备不可用 ({type(exc).__name__})"
            return
        self._sample()
        self.thread = threading.Thread(target=self._poll, name="simverse-nvml", daemon=True)
        self.thread.start()

    def _sample(self):
        try:
            memory = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
        except self.nvml.NVMLError as exc:
            self.reason = f"NVML 整卡显存采样失败 ({type(exc).__name__})"
            return
        self.samples.append({"time_ns": self.now(), "used_bytes": memory.used, "total_bytes": memory.total})

    def _poll(self):
        while not self.stop.wait(SAMPLE_INTERVAL_NS / NS_PER_S):
            self._sample()

    def finish(self):
        if self.thread is not None:
            self.stop.set()
            self.thread.join()
            self._sample()
        if self.initialized:
            try:
                self.nvml.nvmlShutdown()
            except self.nvml.NVMLError as exc:
                self.reason = self.reason or f"NVML 关闭失败 ({type(exc).__name__})"
        return {"device_uuid": self.device_uuid, "sample_interval_ns": SAMPLE_INTERVAL_NS,
                "samples": self.samples, "reason": self.reason}
