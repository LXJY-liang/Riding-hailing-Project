# -*- coding: utf-8 -*-
"""冻结最佳Qwen+LoRA，只导出SAC动态轨迹实际会访问的512维状态。"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer


HERE = Path(__file__).resolve().parent


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cfg = load(HERE.parent / "1 数据准备" / "1 Vit预训练配置.py", "vit_pretrain_config")


class TextDataset(Dataset):
    def __init__(self, text: list[str]):
        self.text = text

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        return self.text[index]


def required_states_for_split(days: list[pd.Timestamp]) -> dict[str, np.ndarray]:
    """返回一个split内各日必需状态；跨日映射到真实次日而不是夹到23:45。"""

    day_keys = {f"{day:%Y-%m-%d}" for day in days}
    masks: dict[str, np.ndarray] = {}
    region_ids = None
    episodes = {}
    for day in days:
        key = f"{day:%Y-%m-%d}"
        path = cfg.SAC_EPISODE_DIR / f"episode_{key}.npz"
        if not path.exists():
            raise FileNotFoundError(f"缺少SAC日级episode，无法确定必需编码：{path}")
        with np.load(path, allow_pickle=False) as loaded:
            current_regions = loaded["region_ids"].astype(str)
            episodes[key] = {name: loaded[name] for name in (
                "group_origin", "group_slot", "od_j", "od_slot", "tau_q"
            )}
        if region_ids is None:
            region_ids = current_regions
        elif not np.array_equal(region_ids, current_regions):
            raise ValueError(f"{key}的区域顺序与同一split其他日期不一致。")
        masks[key] = np.zeros(
            (cfg.TIME_STEPS_PER_DAY, len(current_regions)), dtype=bool
        )

    for key, episode in episodes.items():
        day = pd.Timestamp(key)
        origin = episode["group_origin"].astype(np.int64)
        slot = episode["group_slot"].astype(np.int64)
        masks[key][slot, origin] = True

        waiting_raw = slot + 1
        for offset in np.unique(waiting_raw // cfg.TIME_STEPS_PER_DAY):
            selected = waiting_raw // cfg.TIME_STEPS_PER_DAY == offset
            target_day = f"{day + pd.Timedelta(days=int(offset)):%Y-%m-%d}"
            if target_day in day_keys:
                masks[target_day][
                    waiting_raw[selected] % cfg.TIME_STEPS_PER_DAY, origin[selected]
                ] = True

        destination = episode["od_j"].astype(np.int64)
        departure_slot = episode["od_slot"].astype(np.int64)
        tau = episode["tau_q"].astype(np.float32)
        minimum_step = np.maximum(np.ceil(tau.min(axis=1) / 15.0), 1).astype(np.int16)
        maximum_step = np.maximum(
            np.ceil(tau.max(axis=1) / 15.0), minimum_step
        ).astype(np.int16)
        for step in range(1, int(maximum_step.max(initial=1)) + 1):
            selected = (minimum_step <= step) & (step <= maximum_step)
            if not selected.any():
                continue
            arrival_raw = departure_slot[selected] + step
            selected_destination = destination[selected]
            for offset in np.unique(arrival_raw // cfg.TIME_STEPS_PER_DAY):
                within_offset = arrival_raw // cfg.TIME_STEPS_PER_DAY == offset
                target_day = f"{day + pd.Timedelta(days=int(offset)):%Y-%m-%d}"
                if target_day in day_keys:
                    masks[target_day][
                        arrival_raw[within_offset] % cfg.TIME_STEPS_PER_DAY,
                        selected_destination[within_offset],
                    ] = True
    return masks


def export_day(
    day: pd.Timestamp,
    required_mask: np.ndarray,
    model,
    tokenizer,
    device,
    overwrite: bool = False,
) -> None:
    state_file = cfg.STATE_TEXT_DIR / f"vit_state_{day:%Y-%m-%d}.parquet"
    if not state_file.exists():
        raise FileNotFoundError(f"缺少Vit状态文本：{state_file}")
    frame = pd.read_parquet(state_file).sort_values(["t", "region_index"], kind="mergesort")
    region_ids = np.asarray(
        frame.loc[frame["t"] == frame["t"].min()]
        .sort_values("region_index")["region_id"]
        .astype(str)
        .tolist(),
        dtype=np.str_,
    )
    region_count = len(region_ids)
    expected = cfg.TIME_STEPS_PER_DAY * region_count
    if len(frame) != expected:
        raise ValueError(f"{day:%Y-%m-%d}状态文本应有 {expected} 行，实际 {len(frame)} 行。")

    if required_mask.shape != (cfg.TIME_STEPS_PER_DAY, region_count):
        raise ValueError(f"{day:%Y-%m-%d}必需状态掩码形状错误：{required_mask.shape}")
    output_file = cfg.STATE_FEATURE_DIR / f"vit_features_{day:%Y-%m-%d}.npz"
    # `features`始终保存CPU侧的完整日级NumPy状态矩阵。不要在下方的
    # Qwen推理循环中复用这个变量名，否则会被CUDA Tensor覆盖，导致
    # 增量结果无法写回NumPy数组。
    features = np.zeros(
        (cfg.TIME_STEPS_PER_DAY, region_count, cfg.STATE_FEATURE_DIM), dtype=np.float16
    )
    available = np.zeros((cfg.TIME_STEPS_PER_DAY, region_count), dtype=bool)
    if output_file.exists() and not overwrite:
        with np.load(output_file, allow_pickle=True) as existing:
            if not np.array_equal(existing["region_ids"].astype(str), region_ids.astype(str)):
                raise ValueError(f"{output_file.name}区域顺序不一致，不能增量补全。")
            if existing["features"].shape != features.shape:
                raise ValueError(f"{output_file.name}维度不一致，不能增量补全。")
            features[:] = existing["features"].astype(np.float16)
            available[:] = existing["available"].astype(bool)
    missing_mask = required_mask & ~available
    selected_position = np.flatnonzero(missing_mask.reshape(-1))
    if len(selected_position) == 0:
        print(f"{day:%Y-%m-%d}所需512维状态已全部覆盖，跳过。")
        return
    selected_text = frame.iloc[selected_position]["text"].tolist()

    def collate(texts):
        encoded = tokenizer(
            texts,
            truncation=False,
            padding=True,
            return_tensors="pt",
        )
        if encoded["input_ids"].shape[1] > cfg.MAX_LENGTH:
            raise ValueError("Vit状态Prompt超过模型最大长度，已拒绝截断。")
        return encoded

    loader = DataLoader(
        TextDataset(selected_text),
        batch_size=cfg.EXPORT_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    parts = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"导出 {day:%Y-%m-%d}", dynamic_ncols=True):
            encoded_batch = model.encode(
                batch["input_ids"].to(device, non_blocking=True),
                batch["attention_mask"].to(device, non_blocking=True),
            )
            parts.append(encoded_batch.cpu().to(torch.float16).numpy())
    encoded_missing = np.concatenate(parts, axis=0)
    if encoded_missing.shape != (len(selected_position), cfg.STATE_FEATURE_DIM):
        raise ValueError(
            "新增512维编码数量或维度错误："
            f"预期{(len(selected_position), cfg.STATE_FEATURE_DIM)}，"
            f"实际{encoded_missing.shape}。"
        )
    if not np.isfinite(encoded_missing).all():
        raise FloatingPointError(f"{day:%Y-%m-%d}新增512维编码出现NaN或inf。")
    features.reshape(-1, cfg.STATE_FEATURE_DIM)[selected_position] = encoded_missing
    available |= missing_mask
    np.savez(
        output_file,
        date=np.asarray(str(day.date())),
        region_ids=region_ids,
        features=features,
        available=available,
    )
    print(
        f"{day:%Y-%m-%d}新增编码{len(selected_position):,}个状态；"
        f"当前覆盖{int(available.sum()):,}个状态。"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", choices=tuple(cfg.SPLITS), default=list(cfg.SPLITS))
    parser.add_argument("--overwrite", action="store_true", help="忽略已有编码并完整重算。")
    args = parser.parse_args()
    cfg.ensure_directories()
    model_module = load(HERE / "1 定义Vit分位数模型.py", "vit_quantile_model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else (
        torch.float16 if device.type == "cuda" else torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.TOKENIZER_DIR, local_files_only=True, trust_remote_code=True)
    model = model_module.VitQuantileModel(
        model_dtype=dtype,
        adapter_path=cfg.BEST_LORA_DIR,
        is_trainable=False,
    )
    model.state_projection.load_state_dict(
        torch.load(cfg.BEST_STATE_PROJECTION_FILE, map_location="cpu", weights_only=True)
    )
    model = model.to(device).eval()
    required_by_day: dict[str, np.ndarray] = {}
    for split in args.splits:
        start, end = map(pd.Timestamp, cfg.SPLITS[split])
        split_days = list(pd.date_range(start.normalize(), end.normalize(), freq="D"))
        for day, mask in required_states_for_split(split_days).items():
            if day in required_by_day:
                required_by_day[day] |= mask
            else:
                required_by_day[day] = mask
    for key in sorted(required_by_day):
        export_day(
            pd.Timestamp(key), required_by_day[key], model, tokenizer, device, args.overwrite
        )


if __name__ == "__main__":
    main()
