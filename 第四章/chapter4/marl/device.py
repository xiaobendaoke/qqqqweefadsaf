"""第四章 MARL 设备选择工具模块。

该模块统一处理 CUDA/CPU 设备请求、自动回退策略与运行时信息导出，
确保训练、评估与论文实验入口共享一致的设备解析行为。
"""

from __future__ import annotations

from typing import Any

import torch


def normalize_device_request(device: str | None) -> str:
    """将外部设备请求归一化为统一字符串表示。"""
    requested = "auto" if device is None else str(device).strip().lower()
    if not requested:
        return "auto"
    if requested in {"auto", "cpu", "cuda"}:
        return requested
    if requested.startswith("cuda:") and requested.split(":", maxsplit=1)[1].isdigit():
        return requested
    raise ValueError(
        "Unsupported device request. Use one of: auto, cpu, cuda, cuda:0, cuda:1, ..."
    )


def resolve_device(device: str | None) -> str:
    """将请求设备解析为 torch 可直接使用的实际设备。"""
    requested = normalize_device_request(device)
    if requested == "cpu":
        return "cpu"
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device was requested, but torch.cuda.is_available() is False. "
            "Install a CUDA-enabled torch build or rerun with --device cpu."
        )
    device_count = torch.cuda.device_count()
    if requested == "cuda":
        return "cuda:0"
    device_index = int(requested.split(":", maxsplit=1)[1])
    if device_index >= device_count:
        raise RuntimeError(
            f"Requested CUDA device index {device_index}, but only {device_count} CUDA device(s) are visible."
        )
    return requested


def configure_torch_runtime(resolved_device: str) -> None:
    """为 CUDA 运行时开启安全的默认加速设置。"""
    if not resolved_device.startswith("cuda"):
        return
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True


def describe_runtime_device(
    requested_device: str | None,
    *,
    resolved_device: str | None = None,
) -> dict[str, Any]:
    """导出当前 torch 运行时与设备选择信息。"""
    requested = normalize_device_request(requested_device)
    resolved = resolved_device or resolve_device(requested)
    gpu_name: str | None = None
    if resolved.startswith("cuda") and torch.cuda.is_available():
        device_index = int(resolved.split(":", maxsplit=1)[1])
        if device_index < torch.cuda.device_count():
            gpu_name = str(torch.cuda.get_device_name(device_index))
    return {
        "device": resolved,
        "requested_device": requested,
        "resolved_device": resolved,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_version": torch.version.cuda,
        "gpu_name": gpu_name,
    }
