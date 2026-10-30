# -*- coding: utf-8 -*-
"""监督预训练Qwen3-4B、Vit专用LoRA、状态投影和101维分位数头。"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import torch
from datasets import concatenate_datasets, load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from transformers.trainer_pt_utils import LengthGroupedSampler


HERE = Path(__file__).resolve().parent


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cfg = load(HERE.parent / "1 数据准备" / "1 Vit预训练配置.py", "vit_pretrain_config")
model_module = load(HERE / "1 定义Vit分位数模型.py", "vit_quantile_model")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_and_dtype() -> tuple[torch.device, torch.dtype]:
    if cfg.DEVICE != "auto":
        device = torch.device(cfg.DEVICE)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif device.type == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float32
    return device, dtype


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, samples):
        batch = self.tokenizer.pad(
            [
                {"input_ids": sample["input_ids"], "attention_mask": sample["attention_mask"]}
                for sample in samples
            ],
            padding=True,
            return_tensors="pt",
        )
        batch["log_supply"] = torch.tensor([sample["log_supply"] for sample in samples], dtype=torch.float32)
        batch["demand_summary"] = torch.tensor(
            [sample["demand_summary"] for sample in samples], dtype=torch.float32
        )
        batch["labels"] = torch.tensor([sample["labels"] for sample in samples], dtype=torch.float32)
        return batch


def token_lengths(dataset) -> list[int]:
    """直接从Arrow偏移量读取每条Prompt长度，不展开十亿级Token列表。"""

    lengths: list[int] = []
    for chunk in dataset.data.column("input_ids").chunks:
        lengths.extend(pc.list_value_length(chunk).to_pylist())
    if len(lengths) != len(dataset):
        raise RuntimeError(f"Prompt长度数量错误：{len(lengths):,} != {len(dataset):,}")
    return lengths


def length_grouped_loader(
    dataset,
    lengths: list[int],
    batch_size: int,
    collator,
    device: torch.device,
    seed: int,
) -> DataLoader:
    """随机化巨型桶后按相近Prompt长度组batch，减少动态Padding。"""

    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = LengthGroupedSampler(
        batch_size=batch_size,
        lengths=lengths,
        generator=generator,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.NUM_WORKERS > 0,
    )


def benchmark_batch_sizes(
    model,
    train_data,
    train_lengths: list[int],
    collator,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[int, int]:
    """用同一批真实样本短测batch 2/4，按samples/s选择正式训练配置。"""

    sample_count = min(cfg.BATCH_BENCHMARK_SAMPLES, len(train_data))
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    indices = rng.choice(len(train_data), size=sample_count, replace=False).tolist()
    benchmark_data = train_data.select(indices)
    benchmark_lengths = [train_lengths[index] for index in indices]
    results: list[dict[str, object]] = []

    print(
        f"开始Vit batch短基准：候选={cfg.BATCH_BENCHMARK_CANDIDATES}，"
        f"固定样本={sample_count:,}，长度分桶={cfg.USE_LENGTH_GROUPED_SAMPLER}",
        flush=True,
    )
    for batch_size in cfg.BATCH_BENCHMARK_CANDIDATES:
        if cfg.EFFECTIVE_BATCH_SIZE % batch_size:
            raise ValueError(
                f"有效batch={cfg.EFFECTIVE_BATCH_SIZE}不能被候选batch={batch_size}整除。"
            )
        loader = length_grouped_loader(
            benchmark_data,
            benchmark_lengths,
            batch_size,
            collator,
            device,
            cfg.RANDOM_SEED,
        )
        model.train()
        model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        start_time: float | None = None
        measured_samples = 0
        measured_tokens = 0
        status = "ok"
        error = ""
        try:
            for step, batch in enumerate(loader, start=1):
                if step == cfg.BATCH_BENCHMARK_WARMUP_STEPS + 1:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    start_time = time.perf_counter()
                labels = batch["labels"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                with torch.autocast(
                    device_type="cuda",
                    dtype=amp_dtype,
                    enabled=device.type == "cuda",
                ):
                    prediction = model(
                        batch["input_ids"].to(device, non_blocking=True),
                        attention_mask,
                        batch["log_supply"].to(device, non_blocking=True),
                        batch["demand_summary"].to(device, non_blocking=True),
                    )
                    loss = model_module.quantile_curve_loss(prediction, labels)
                loss.backward()
                model.zero_grad(set_to_none=True)
                if start_time is not None:
                    measured_samples += len(labels)
                    measured_tokens += int(attention_mask.sum().item())
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            status = "oom"
            error = str(exc)
            model.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start_time if start_time is not None else 0.0
        samples_per_second = measured_samples / elapsed if elapsed > 0 else 0.0
        tokens_per_second = measured_tokens / elapsed if elapsed > 0 else 0.0
        peak_gb = (
            torch.cuda.max_memory_allocated(device) / 1024**3
            if device.type == "cuda"
            else 0.0
        )
        result = {
            "batch_size": batch_size,
            "gradient_accumulation_steps": cfg.EFFECTIVE_BATCH_SIZE // batch_size,
            "status": status,
            "measured_samples": measured_samples,
            "elapsed_seconds": elapsed,
            "samples_per_second": samples_per_second,
            "tokens_per_second": tokens_per_second,
            "peak_memory_gb": peak_gb,
            "error": error,
        }
        results.append(result)
        print(
            f"batch={batch_size}：status={status}，samples/s={samples_per_second:.3f}，"
            f"tokens/s={tokens_per_second:.1f}，peak_memory={peak_gb:.2f}GB",
            flush=True,
        )

    valid_results = [result for result in results if result["status"] == "ok"]
    if not valid_results:
        raise RuntimeError("batch 2和batch 4短基准均失败，无法开始正式训练。")
    best = max(valid_results, key=lambda result: float(result["samples_per_second"]))
    selected_batch = int(best["batch_size"])
    accumulation = cfg.EFFECTIVE_BATCH_SIZE // selected_batch
    report = {
        "selection_rule": "同一批真实样本、长度分桶后的最高samples_per_second",
        "selected_batch_size": selected_batch,
        "selected_gradient_accumulation_steps": accumulation,
        "effective_batch_size": cfg.EFFECTIVE_BATCH_SIZE,
        "results": results,
    }
    cfg.BATCH_BENCHMARK_FILE.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"batch短基准选择：batch_size={selected_batch}，"
        f"gradient_accumulation_steps={accumulation}，"
        f"effective_batch={cfg.EFFECTIVE_BATCH_SIZE}",
        flush=True,
    )
    return selected_batch, accumulation


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in tqdm(loader, desc="验证", dynamic_ncols=True):
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            prediction = model(
                batch["input_ids"].to(device, non_blocking=True),
                batch["attention_mask"].to(device, non_blocking=True),
                batch["log_supply"].to(device, non_blocking=True),
                batch["demand_summary"].to(device, non_blocking=True),
            )
            loss = model_module.quantile_curve_loss(prediction, labels)
        total += float(loss.item()) * len(labels)
        count += len(labels)
    return total / max(count, 1)


def load_validation_data(collator, device: torch.device):
    """训练启动前完整加载验证集，并按配置固定抽取同一个验证子集。"""

    if not cfg.VALIDATION_READY_FILE.exists():
        raise FileNotFoundError(
            "验证集尚未完整上传，拒绝启动训练。缺少完成标记："
            f"{cfg.VALIDATION_READY_FILE}"
        )
    ready = json.loads(cfg.VALIDATION_READY_FILE.read_text(encoding="utf-8"))
    if int(ready.get("rows", 0)) != cfg.EXPECTED_VALID_ROWS:
        raise ValueError(f"验证集READY标记行数错误：{ready}")
    parts = []
    for day in cfg.days("valid"):
        path = cfg.TOKENIZED_VALID_PART_DIR / f"vit_valid_{day:%Y-%m-%d}"
        if not path.exists():
            raise FileNotFoundError(f"READY已存在但缺少验证分片：{path}")
        parts.append(load_from_disk(str(path)))
    full_data = concatenate_datasets(parts)
    if len(full_data) != cfg.EXPECTED_VALID_ROWS:
        raise ValueError(
            f"完整验证集应为{cfg.EXPECTED_VALID_ROWS:,}行，实际{len(full_data):,}行。"
        )
    sample_count = len(full_data) // cfg.VALIDATION_SAMPLE_DENOMINATOR
    rng = np.random.default_rng(cfg.VALIDATION_SAMPLE_SEED)
    indices = np.sort(
        rng.choice(len(full_data), size=sample_count, replace=False)
    ).tolist()
    sampled_data = full_data.select(indices)
    loader = DataLoader(
        sampled_data,
        batch_size=cfg.EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.NUM_WORKERS > 0,
    )
    print(
        f"验证集已加载：完整{len(full_data):,}行；固定子集{len(sampled_data):,}行；"
        f"固定随机种子={cfg.VALIDATION_SAMPLE_SEED}，抽样比例="
        f"1/{cfg.VALIDATION_SAMPLE_DENOMINATOR}。",
        flush=True,
    )
    return loader, len(sampled_data)


def save_best(
    model,
    epoch: int,
    global_update: int,
    progress_fraction: float,
    train_loss: float,
    valid_loss: float,
    training_rows: int,
    validation_rows: int,
) -> None:
    if cfg.BEST_LORA_DIR.exists():
        shutil.rmtree(cfg.BEST_LORA_DIR)
    model.backbone.save_pretrained(cfg.BEST_LORA_DIR)
    torch.save(model.state_projection.state_dict(), cfg.BEST_STATE_PROJECTION_FILE)
    torch.save(model.quantile_head.state_dict(), cfg.BEST_QUANTILE_HEAD_FILE)
    with cfg.BEST_MODEL_INFO_FILE.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "selection_rule": (
                    "6月25—27日固定"
                    f"1/{cfg.VALIDATION_SAMPLE_DENOMINATOR}验证子集最低损失"
                ),
                "best_epoch": epoch,
                "best_global_update": global_update,
                "best_training_progress": progress_fraction,
                "train_loss": train_loss,
                "best_valid_loss": valid_loss,
                "training_rows": training_rows,
                "validation_sample_rows": validation_rows,
                "validation_full_rows": cfg.EXPECTED_VALID_ROWS,
                "validation_sample_denominator": cfg.VALIDATION_SAMPLE_DENOMINATOR,
                "validation_sample_seed": cfg.VALIDATION_SAMPLE_SEED,
            },
            stream,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:
    cfg.ensure_directories()
    set_seed(cfg.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    device, model_dtype = device_and_dtype()
    amp_dtype = torch.bfloat16 if model_dtype == torch.bfloat16 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.TOKENIZER_DIR,
        local_files_only=True,
        trust_remote_code=True,
    )
    tokenizer.padding_side = cfg.PADDING_SIDE
    collator = Collator(tokenizer)
    daily_parts = []
    for day in cfg.days("train"):
        path = cfg.TOKENIZED_TRAIN_PART_DIR / f"vit_train_{day:%Y-%m-%d}"
        if not path.exists():
            raise FileNotFoundError(f"缺少全量Tokenize分片：{path}")
        daily_parts.append(load_from_disk(str(path)))
    train_data = concatenate_datasets(daily_parts)
    print(f"全量Vit训练样本：{len(train_data):,}行（6月1—24日，无抽样）")
    train_lengths = token_lengths(train_data)
    print(
        f"Prompt Token长度：min={min(train_lengths):,}，"
        f"mean={np.mean(train_lengths):.2f}，max={max(train_lengths):,}",
        flush=True,
    )
    validation_loader, validation_rows = load_validation_data(collator, device)

    model = model_module.VitQuantileModel(model_dtype=model_dtype).to(device)
    train_batch_size, gradient_accumulation_steps = benchmark_batch_sizes(
        model,
        train_data,
        train_lengths,
        collator,
        device,
        amp_dtype,
    )
    train_loader = length_grouped_loader(
        train_data,
        train_lengths,
        train_batch_size,
        collator,
        device,
        cfg.RANDOM_SEED,
    )
    lora_parameters = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    head_parameters = list(model.state_projection.parameters()) + list(model.quantile_head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_parameters, "lr": cfg.LORA_LEARNING_RATE},
            {"params": head_parameters, "lr": cfg.HEAD_LEARNING_RATE},
        ],
        weight_decay=cfg.WEIGHT_DECAY,
    )
    updates_per_epoch = math.ceil(len(train_loader) / gradient_accumulation_steps)
    total_updates = updates_per_epoch * cfg.NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_updates * cfg.WARMUP_RATIO),
        num_training_steps=total_updates,
    )

    validation_updates = sorted(
        {
            max(1, math.ceil(total_updates * check / cfg.VALIDATION_CHECKS_PER_EPOCH))
            for check in range(1, cfg.VALIDATION_CHECKS_PER_EPOCH + 1)
        }
    )
    print(f"中途验证更新点：{validation_updates}（共{len(validation_updates)}次）")
    best_valid_loss = float("inf")
    global_update = 0
    validation_count = 0
    with cfg.TRAINING_HISTORY_FILE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "event", "epoch", "global_update", "training_progress",
                "train_loss", "valid_loss", "validation_rows", "is_best",
            ],
        )
        writer.writeheader()
        for epoch in range(1, cfg.NUM_EPOCHS + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            total_rows = 0
            progress = tqdm(train_loader, desc=f"训练 epoch {epoch}", dynamic_ncols=True)
            for step, batch in enumerate(progress, start=1):
                labels = batch["labels"].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                    prediction = model(
                        batch["input_ids"].to(device, non_blocking=True),
                        batch["attention_mask"].to(device, non_blocking=True),
                        batch["log_supply"].to(device, non_blocking=True),
                        batch["demand_summary"].to(device, non_blocking=True),
                    )
                    loss = model_module.quantile_curve_loss(prediction, labels)
                    scaled_loss = loss / gradient_accumulation_steps
                scaled_loss.backward()
                if step % gradient_accumulation_steps == 0 or step == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_update += 1
                    if global_update in validation_updates:
                        train_loss_so_far = total_loss / max(total_rows, 1)
                        valid_loss = evaluate(
                            model, validation_loader, device, amp_dtype
                        )
                        validation_count += 1
                        progress_fraction = global_update / total_updates
                        is_best = valid_loss < best_valid_loss
                        if is_best:
                            best_valid_loss = valid_loss
                            save_best(
                                model=model,
                                epoch=epoch,
                                global_update=global_update,
                                progress_fraction=progress_fraction,
                                train_loss=train_loss_so_far,
                                valid_loss=valid_loss,
                                training_rows=len(train_data),
                                validation_rows=validation_rows,
                            )
                            print("已按独立验证损失保存新的最佳Vit模型。")
                        writer.writerow(
                            {
                                "event": "step_validation",
                                "epoch": epoch,
                                "global_update": global_update,
                                "training_progress": progress_fraction,
                                "train_loss": train_loss_so_far,
                                "valid_loss": valid_loss,
                                "validation_rows": validation_rows,
                                "is_best": is_best,
                            }
                        )
                        stream.flush()
                        print(
                            f"验证{validation_count}/{cfg.VALIDATION_CHECKS_PER_EPOCH}："
                            f"progress={progress_fraction:.1%}, valid_loss={valid_loss:.6f}, "
                            f"best={best_valid_loss:.6f}",
                            flush=True,
                        )
                        model.train()
                total_loss += float(loss.item()) * len(labels)
                total_rows += len(labels)
                progress.set_postfix(loss=f"{loss.item():.4f}")

            train_loss = total_loss / max(total_rows, 1)
            print(f"epoch={epoch}, train_loss={train_loss:.6f}, rows={total_rows:,}")
    if validation_count != len(validation_updates):
        raise RuntimeError(
            f"预期验证{len(validation_updates)}次，实际{validation_count}次。"
        )


if __name__ == "__main__":
    main()
