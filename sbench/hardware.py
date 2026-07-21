"""Hardware peak table aligned with MoE-CAP's hardware_utils values."""

from __future__ import annotations

MEM_BW = {
    "NVIDIA-A100-PCIe-80GB": 1935e9,
    "NVIDIA-A100-PCIe-40GB": 1555e9,
    "NVIDIA-A100-SXM4-80GB": 2039e9,
    "NVIDIA-A100-SXM4-40GB": 1555e9,
    "NVIDIA-H100-PCIe-80GB": 2039e9,
    "NVIDIA-RTX-A5000-24GB": 768e9,
    "NVIDIA-RTX-A6000-48GB": 768e9,
    "NVIDIA-H100-HBM3-80GB": 3350e9,
    "NVIDIA-H100-NVL-94GB": 3900e9,
    "NVIDIA-H100-NVL-96GB": 3900e9,
    "NVIDIA-H200-141GB": 4800e9,
    "NVIDIA-H200-140GB": 4800e9,
    "NVIDIA-B200-183GB": 8000e9,
    "NVIDIA-GH200-96GB": 4096e9,
    "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition-96GB": 1597e9,
    "AMD-Instinct-MI355X-288GB": 8000e9,
    "AMD-Instinct-MI300X-192GB": 5300e9,
    "AMD-Instinct-MI325X-256GB": 6000e9,
    "AMD-Instinct-MI250X-128GB": 3276e9,
}

PEAK_FLOPS = {
    "float32": {
        "NVIDIA-A100-PCIe-80GB": 312e12,
        "NVIDIA-A100-PCIe-40GB": 312e12,
        "NVIDIA-A100-SXM4-80GB": 312e12,
        "NVIDIA-A100-SXM4-40GB": 312e12,
        "NVIDIA-H100-PCIe-80GB": 756e12,
        "NVIDIA-RTX-A5000-24GB": 222.2e12,
        "NVIDIA-RTX-A6000-48GB": 309.7e12,
        "NVIDIA-H100-HBM3-80GB": 989e12,
        "NVIDIA-H100-NVL-94GB": 835e12,
        "NVIDIA-H100-NVL-96GB": 835e12,
        "NVIDIA-H200-141GB": 989e12,
        "NVIDIA-H200-140GB": 989e12,
        "NVIDIA-B200-183GB": 80e12,
        "NVIDIA-GH200-96GB": 989e12,
        "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition-96GB": 500e12,
        "AMD-Instinct-MI355X-288GB": 78.6e12,
        "AMD-Instinct-MI300X-192GB": 81.7e12,
        "AMD-Instinct-MI325X-256GB": 81.7e12,
        "AMD-Instinct-MI250X-128GB": 47.9e12,
    },
    "float16": {
        "NVIDIA-A100-PCIe-80GB": 624e12,
        "NVIDIA-A100-PCIe-40GB": 624e12,
        "NVIDIA-A100-SXM4-80GB": 624e12,
        "NVIDIA-A100-SXM4-40GB": 624e12,
        "NVIDIA-H100-PCIe-80GB": 1513e12,
        "NVIDIA-RTX-A5000-24GB": 222.2e12,
        "NVIDIA-RTX-A6000-48GB": 309.7e12,
        "NVIDIA-H100-HBM3-80GB": 1979e12,
        "NVIDIA-H100-NVL-94GB": 1671e12,
        "NVIDIA-H100-NVL-96GB": 1671e12,
        "NVIDIA-H200-141GB": 1979e12,
        "NVIDIA-H200-140GB": 1979e12,
        "NVIDIA-B200-183GB": 2250e12,
        "NVIDIA-GH200-96GB": 1979e12,
        "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition-96GB": 1000e12,
        "AMD-Instinct-MI355X-288GB": 2500e12,
        "AMD-Instinct-MI300X-192GB": 1307e12,
        "AMD-Instinct-MI325X-256GB": 1307e12,
        "AMD-Instinct-MI250X-128GB": 383e12,
    },
    "bfloat16": {
        "NVIDIA-A100-PCIe-80GB": 624e12,
        "NVIDIA-A100-PCIe-40GB": 624e12,
        "NVIDIA-A100-SXM4-80GB": 624e12,
        "NVIDIA-A100-SXM4-40GB": 624e12,
        "NVIDIA-H100-PCIe-80GB": 1513e12,
        "NVIDIA-RTX-A5000-24GB": 222.2e12,
        "NVIDIA-RTX-A6000-48GB": 309.7e12,
        "NVIDIA-H100-HBM3-80GB": 1979e12,
        "NVIDIA-H100-NVL-94GB": 1671e12,
        "NVIDIA-H100-NVL-96GB": 1671e12,
        "NVIDIA-H200-141GB": 1979e12,
        "NVIDIA-H200-140GB": 1979e12,
        "NVIDIA-B200-183GB": 2250e12,
        "NVIDIA-GH200-96GB": 1979e12,
        "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition-96GB": 1000e12,
        "AMD-Instinct-MI355X-288GB": 2500e12,
        "AMD-Instinct-MI300X-192GB": 1307e12,
        "AMD-Instinct-MI325X-256GB": 1307e12,
        "AMD-Instinct-MI250X-128GB": 383e12,
    },
    "int8": {
        "NVIDIA-A100-PCIe-80GB": 1248e12,
        "NVIDIA-A100-PCIe-40GB": 1248e12,
        "NVIDIA-A100-SXM4-80GB": 1248e12,
        "NVIDIA-A100-SXM4-40GB": 1248e12,
        "NVIDIA-H100-PCIe-80GB": 3026e12,
        "NVIDIA-RTX-A5000-24GB": 222.2e12,
        "NVIDIA-RTX-A6000-48GB": 309.7e12,
        "NVIDIA-H100-HBM3-80GB": 3958e12,
        "NVIDIA-H100-NVL-94GB": 3341e12,
        "NVIDIA-H100-NVL-96GB": 3341e12,
        "NVIDIA-H200-141GB": 3958e12,
        "NVIDIA-H200-140GB": 3958e12,
        "NVIDIA-B200-183GB": 4500e12,
        "NVIDIA-GH200-96GB": 3958e12,
        "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition-96GB": 2000e12,
        "AMD-Instinct-MI355X-288GB": 5050e12,
        "AMD-Instinct-MI300X-192GB": 2614e12,
        "AMD-Instinct-MI325X-256GB": 2614e12,
        "AMD-Instinct-MI250X-128GB": 383e12,
    },
    "fp8": {
        "NVIDIA-A100-PCIe-80GB": 1248e12,
        "NVIDIA-A100-PCIe-40GB": 1248e12,
        "NVIDIA-A100-SXM4-80GB": 1248e12,
        "NVIDIA-A100-SXM4-40GB": 1248e12,
        "NVIDIA-H100-PCIe-80GB": 3026e12,
        "NVIDIA-RTX-A5000-24GB": 0,
        "NVIDIA-RTX-A6000-48GB": 0,
        "NVIDIA-H100-HBM3-80GB": 3958e12,
        "NVIDIA-H100-NVL-94GB": 3341e12,
        "NVIDIA-H100-NVL-96GB": 3341e12,
        "NVIDIA-H200-141GB": 3958e12,
        "NVIDIA-H200-140GB": 3958e12,
        "NVIDIA-B200-183GB": 4500e12,
        "NVIDIA-GH200-96GB": 3958e12,
        "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition-96GB": 2000e12,
        "AMD-Instinct-MI355X-288GB": 5050e12,
        "AMD-Instinct-MI300X-192GB": 2614e12,
        "AMD-Instinct-MI325X-256GB": 2614e12,
        "AMD-Instinct-MI250X-128GB": 0,
    },
    "fp4": {
        "NVIDIA-A100-PCIe-80GB": 1248e12,
        "NVIDIA-A100-SXM4-80GB": 1248e12,
        "NVIDIA-A100-SXM4-40GB": 1248e12,
        "NVIDIA-H100-PCIe-80GB": 3026e12,
        "NVIDIA-RTX-A5000-24GB": 0,
        "NVIDIA-RTX-A6000-48GB": 0,
        "NVIDIA-H100-HBM3-80GB": 3958e12,
        "NVIDIA-H100-NVL-94GB": 3341e12,
        "NVIDIA-H100-NVL-96GB": 3341e12,
        "NVIDIA-H200-141GB": 3958e12,
        "NVIDIA-H200-140GB": 3958e12,
        "NVIDIA-B200-183GB": 9000e12,
        "NVIDIA-GH200-96GB": 3958e12,
        "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition-96GB": 4000e12,
        "AMD-Instinct-MI355X-288GB": 10100e12,
        "AMD-Instinct-MI300X-192GB": 0,
        "AMD-Instinct-MI325X-256GB": 0,
        "AMD-Instinct-MI250X-128GB": 0,
    },
    "int4": {
        "NVIDIA-A100-PCIe-80GB": 2496e12,
        "NVIDIA-A100-PCIe-40GB": 2496e12,
        "NVIDIA-A100-SXM4-80GB": 2496e12,
        "NVIDIA-A100-SXM4-40GB": 2496e12,
        "NVIDIA-H100-PCIe-80GB": 3026e12,
        "NVIDIA-RTX-A5000-24GB": 222.2e12,
        "NVIDIA-RTX-A6000-48GB": 309.7e12,
        "NVIDIA-H100-HBM3-80GB": 3958e12,
        "NVIDIA-H100-NVL-94GB": 3341e12,
        "NVIDIA-H100-NVL-96GB": 3341e12,
        "NVIDIA-H200-141GB": 3958e12,
        "NVIDIA-H200-140GB": 3958e12,
        "NVIDIA-B200-183GB": 9000e12,
        "NVIDIA-GH200-96GB": 3958e12,
        "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition-96GB": 4000e12,
        "AMD-Instinct-MI355X-288GB": 10100e12,
        "AMD-Instinct-MI300X-192GB": 0,
        "AMD-Instinct-MI325X-256GB": 0,
        "AMD-Instinct-MI250X-128GB": 383e12,
    },
}

ALIASES = {
    "NVIDIA A100-SXM4-40GB": "NVIDIA-A100-SXM4-40GB",
    "NVIDIA A100-PCIE-40GB": "NVIDIA-A100-PCIe-40GB",
    "NVIDIA-A100-PCIE-40GB": "NVIDIA-A100-PCIe-40GB",
    "NVIDIA A100 PCIe 40GB": "NVIDIA-A100-PCIe-40GB",
    "NVIDIA A100-SXM4-80GB": "NVIDIA-A100-SXM4-80GB",
    "NVIDIA A100-PCIE-80GB": "NVIDIA-A100-PCIe-80GB",
    "NVIDIA-A100-PCIE-80GB": "NVIDIA-A100-PCIe-80GB",
    "NVIDIA H100 NVL": "NVIDIA-H100-NVL-94GB",
    "NVIDIA H100 80GB HBM3": "NVIDIA-H100-HBM3-80GB",
    "NVIDIA-H100-SXM5-80GB": "NVIDIA-H100-HBM3-80GB",
    "NVIDIA H100 PCIe 80GB": "NVIDIA-H100-PCIe-80GB",
    "NVIDIA-H100-PCIE-80GB": "NVIDIA-H100-PCIe-80GB",
    "NVIDIA H200 141GB": "NVIDIA-H200-141GB",
    "NVIDIA H200 140GB": "NVIDIA-H200-140GB",
    "NVIDIA B200 183GB": "NVIDIA-B200-183GB",
    "NVIDIA GH200 96GB": "NVIDIA-GH200-96GB",
}


def canonical_gpu_type(gpu_type: str | None) -> str | None:
    if gpu_type is None:
        return None
    value = str(gpu_type).strip()
    if value in MEM_BW:
        return value
    if value in ALIASES:
        return ALIASES[value]
    hyphenated = value.replace(" ", "-")
    return ALIASES.get(hyphenated, hyphenated if hyphenated in MEM_BW else value)


def precision_key(precision: str) -> str:
    value = str(precision).lower()
    if value in {"bfloat16", "bf16"}:
        return "bfloat16"
    if value in {"float16", "fp16"}:
        return "float16"
    if value in {"float32", "fp32"}:
        return "float32"
    if value in {"int8", "fp8", "fp4", "int4"}:
        return value
    return value


def peak_bandwidth_tb(gpu_type: str | None) -> float:
    gpu = canonical_gpu_type(gpu_type)
    if gpu not in MEM_BW:
        raise ValueError(f"unknown GPU type for S-MBU normalization: {gpu_type!r}; set SBENCH_GPU_TYPE or ANALYZE_GPU_TYPE")
    return MEM_BW[gpu] / 1e12


def peak_flops_tf(gpu_type: str | None, precision: str) -> float:
    gpu = canonical_gpu_type(gpu_type)
    key = precision_key(precision)
    if gpu not in MEM_BW:
        raise ValueError(f"unknown GPU type for S-MFU normalization: {gpu_type!r}; set SBENCH_GPU_TYPE or ANALYZE_GPU_TYPE")
    if key not in PEAK_FLOPS or gpu not in PEAK_FLOPS[key]:
        raise ValueError(f"unknown precision/GPU pair for S-MFU normalization: precision={precision!r}, gpu_type={gpu_type!r}")
    return PEAK_FLOPS[key][gpu] / 1e12
