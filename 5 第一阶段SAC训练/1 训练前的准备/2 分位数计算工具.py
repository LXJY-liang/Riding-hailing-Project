# -*- coding: utf-8 -*-
"""统一101维分位数的压缩、插值采样和经验CDF计算。"""

from __future__ import annotations

import numpy as np


QUANTILE_LEVELS = np.asarray(
    [0.005] + [level / 100 for level in range(1, 100)] + [0.995], dtype=np.float32
)
KEY_LEVELS = np.asarray([0.005, 0.05, 0.10, 0.50, 0.90, 0.95, 0.995], dtype=np.float32)


def enforce_quantiles(values: np.ndarray, lower: float = 0.0) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if not np.isfinite(array).all():
        raise FloatingPointError("分位数曲线包含NaN/inf，禁止静默替换。")
    return np.maximum.accumulate(np.maximum(array, lower), axis=-1)


def compress_quantiles(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    bins = []
    for index in range(20):
        left = index / 20
        right = (index + 1) / 20
        mask = (QUANTILE_LEVELS > left) & (QUANTILE_LEVELS <= right)
        bins.append(values[..., mask].mean(axis=-1))
    key_index = [int(np.argmin(np.abs(QUANTILE_LEVELS - level))) for level in KEY_LEVELS]
    return np.concatenate(
        [np.stack(bins, axis=-1), values[..., key_index]], axis=-1
    ).astype(np.float32)


def sample_quantiles(values: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    uniforms = np.clip(
        np.asarray(uniforms, dtype=np.float32), QUANTILE_LEVELS[0], QUANTILE_LEVELS[-1]
    )
    flat = values.reshape(-1, values.shape[-1])
    u = np.broadcast_to(uniforms, values.shape[:-1]).reshape(-1)
    result = np.empty(len(flat), dtype=np.float32)
    for row in range(len(flat)):
        result[row] = np.interp(u[row], QUANTILE_LEVELS, flat[row])
    return result.reshape(values.shape[:-1])


def sample_quantile_scenarios(values: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    uniforms = np.clip(
        np.asarray(uniforms, dtype=np.float32), QUANTILE_LEVELS[0], QUANTILE_LEVELS[-1]
    )
    if values.ndim != 2 or uniforms.ndim != 2 or len(values) != len(uniforms):
        raise ValueError("分位数场景采样要求values=[N,101]、uniforms=[N,S]。")
    result = np.empty_like(uniforms, dtype=np.float32)
    for row in range(len(values)):
        result[row] = np.interp(uniforms[row], QUANTILE_LEVELS, values[row])
    return result


def mixture_quantiles(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim < 2:
        raise ValueError("混合分布样本至少需要二维。")
    flat = samples.reshape(samples.shape[0], -1)
    return np.quantile(flat, QUANTILE_LEVELS, axis=1).T.astype(np.float32)


def cdf_at(values: np.ndarray, points: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    points = np.asarray(points, dtype=np.float32)
    flat = values.reshape(-1, values.shape[-1])
    x = np.broadcast_to(points, values.shape[:-1]).reshape(-1)
    result = np.empty(len(flat), dtype=np.float32)
    for row in range(len(flat)):
        result[row] = np.interp(x[row], flat[row], QUANTILE_LEVELS, left=0.0, right=1.0)
    return result.reshape(values.shape[:-1])
