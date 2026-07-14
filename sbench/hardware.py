"""Small hardware peak table used by analyzer."""

PEAKS = {
    "NVIDIA-H100-NVL-94GB": {"bandwidth": 3900e9, "bf16": 1671e12, "fp16": 1671e12, "fp32": 835e12},
    "NVIDIA-H100-NVL-96GB": {"bandwidth": 3900e9, "bf16": 1671e12, "fp16": 1671e12, "fp32": 835e12},
    "NVIDIA H100 NVL": {"bandwidth": 3900e9, "bf16": 1671e12, "fp16": 1671e12, "fp32": 835e12},
    "NVIDIA-H100-SXM5-80GB": {"bandwidth": 3350e9, "bf16": 1979e12, "fp16": 1979e12, "fp32": 989e12},
    "NVIDIA H100 80GB HBM3": {"bandwidth": 3350e9, "bf16": 1979e12, "fp16": 1979e12, "fp32": 989e12},
    "NVIDIA-A100-SXM4-40GB": {"bandwidth": 1555e9, "bf16": 312e12, "fp16": 312e12, "fp32": 19.5e12},
    "NVIDIA-A100-PCIE-40GB": {"bandwidth": 1555e9, "bf16": 312e12, "fp16": 312e12, "fp32": 19.5e12},
    "NVIDIA A100-SXM4-40GB": {"bandwidth": 1555e9, "bf16": 312e12, "fp16": 312e12, "fp32": 19.5e12},
    "NVIDIA A100-PCIE-40GB": {"bandwidth": 1555e9, "bf16": 312e12, "fp16": 312e12, "fp32": 19.5e12},
    "NVIDIA GeForce RTX 3070 Ti": {"bandwidth": 608.3e9, "bf16": 87.0e12, "fp16": 87.0e12, "fp32": 21.75e12},
}


def precision_key(precision: str) -> str:
    value = precision.lower()
    if value in {"bfloat16", "bf16"}:
        return "bf16"
    if value in {"float16", "fp16"}:
        return "fp16"
    return "fp32"


def peak_bandwidth_tb(gpu_type: str | None) -> float:
    if gpu_type not in PEAKS:
        raise ValueError(f"unknown GPU type for S-MBU normalization: {gpu_type!r}; set SBENCH_GPU_TYPE or ANALYZE_GPU_TYPE")
    return PEAKS[gpu_type]["bandwidth"] / 1e12


def peak_flops_tf(gpu_type: str | None, precision: str) -> float:
    if gpu_type not in PEAKS:
        raise ValueError(f"unknown GPU type for S-MFU normalization: {gpu_type!r}; set SBENCH_GPU_TYPE or ANALYZE_GPU_TYPE")
    table = PEAKS[gpu_type]
    return table[precision_key(precision)] / 1e12
