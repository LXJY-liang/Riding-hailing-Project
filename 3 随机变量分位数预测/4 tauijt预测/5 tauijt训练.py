from pathlib import Path
import argparse
import importlib.util
import json
import math
import random
import shutil
import sys

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup


# =========================
# 1. 读取模块
# =========================

def load_module(file_name, module_name):
    file_path = Path(__file__).resolve().parent / file_name
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


config = load_module("1 tauijt配置.py", "tauijt_config")
model_module = load_module("3 tauijt模型.py", "tauijt_model")


# =========================
# 2. 训练辅助函数
# =========================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if config.device != "auto":
        return torch.device(config.device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_model_dtype(device):
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    if device.type == "mps":
        return torch.float16

    return torch.float32


class tauijt_collator:
    """对当前批次进行动态补齐，并整理标签。"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, samples):
        labels = torch.tensor(
            [sample["labels"] for sample in samples],
            dtype=torch.float32,
        )

        model_inputs = [
            {
                "input_ids": sample["input_ids"],
                "attention_mask": sample["attention_mask"],
            }
            for sample in samples
        ]

        batch = self.tokenizer.pad(
            model_inputs,
            padding=True,
            return_tensors="pt",
        )
        batch["labels"] = labels
        return batch


def evaluate(model, data_loader, device, amp_dtype):
    """计算完整验证集损失。"""

    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.inference_mode():
        for batch in tqdm(
            data_loader,
            desc="验证",
            dynamic_ncols=True,
        ):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=device.type == "cuda",
            ):
                predictions = model(input_ids, attention_mask)
                loss, _, _ = model_module.tauijt_quantile_loss(
                    predictions,
                    labels,
                )

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


def save_best_model(model, epoch, global_step, valid_loss):
    """保存验证损失最低的LoRA和分位数头。"""

    if config.lora_adapter_path.exists():
        shutil.rmtree(config.lora_adapter_path)

    model.backbone.save_pretrained(str(config.lora_adapter_path))
    torch.save(
        model.quantile_head.state_dict(),
        config.quantile_head_path,
    )

    with open(
        config.best_model_info_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "best_epoch": epoch,
                "best_global_step": global_step,
                "best_valid_loss": valid_loss,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )


def get_resume_checkpoint():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume_from_checkpoint",
        default=config.resume_from_checkpoint,
        help="可恢复checkpoint目录，例如output/checkpoints/checkpoint-step-10000",
    )
    args = parser.parse_args()

    if args.resume_from_checkpoint:
        return Path(args.resume_from_checkpoint).expanduser().resolve()

    return None


def log_gpu_memory(stage):
    """打印当前CUDA显存，便于训练启动后观察。"""

    if not torch.cuda.is_available():
        print(f"{stage}：CUDA不可用，当前使用非CUDA设备。")
        return

    allocated = torch.cuda.memory_allocated() / 1024 ** 3
    reserved = torch.cuda.memory_reserved() / 1024 ** 3
    max_allocated = torch.cuda.max_memory_allocated() / 1024 ** 3
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    print(
        f"{stage}：allocated={allocated:.2f}GB，"
        f"reserved={reserved:.2f}GB，"
        f"max_allocated={max_allocated:.2f}GB，"
        f"free={free_bytes / 1024 ** 3:.2f}GB，"
        f"total={total_bytes / 1024 ** 3:.2f}GB"
    )


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    grad_scaler,
    epoch,
    micro_step_in_epoch,
    global_step,
    best_valid_loss,
    history,
):
    checkpoint_dir = (
        config.checkpoint_output_dir / f"checkpoint-step-{global_step}"
    )
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)

    lora_dir = checkpoint_dir / "lora_adapter"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(str(lora_dir))
    torch.save(
        model.quantile_head.state_dict(),
        checkpoint_dir / "quantile_head.pt",
    )
    torch.save(
        {
            "epoch": epoch,
            "micro_step_in_epoch": micro_step_in_epoch,
            "global_step": global_step,
            "best_valid_loss": best_valid_loss,
            "history": history,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "grad_scaler": grad_scaler.state_dict(),
            "random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.random.get_rng_state(),
            "cuda_random_state_all": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
        },
        checkpoint_dir / "trainer_state.pt",
    )
    (config.checkpoint_output_dir / "latest_checkpoint.txt").write_text(
        str(checkpoint_dir),
        encoding="utf-8",
    )
    print(f"已保存可恢复checkpoint：{checkpoint_dir}")


def torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_checkpoint_state(
    checkpoint_dir,
    model,
    optimizer,
    scheduler,
    grad_scaler,
    device,
):
    state_path = checkpoint_dir / "trainer_state.pt"
    head_path = checkpoint_dir / "quantile_head.pt"
    if not state_path.exists() or not head_path.exists():
        raise FileNotFoundError(
            "checkpoint缺少trainer_state.pt或quantile_head.pt："
            f"{checkpoint_dir}"
        )

    model.quantile_head.load_state_dict(
        torch_load(head_path, map_location="cpu")
    )
    trainer_state = torch_load(state_path, map_location=device)
    optimizer.load_state_dict(trainer_state["optimizer"])
    scheduler.load_state_dict(trainer_state["scheduler"])
    grad_scaler.load_state_dict(trainer_state["grad_scaler"])
    random.setstate(trainer_state["random_state"])
    np.random.set_state(trainer_state["numpy_random_state"])
    torch.random.set_rng_state(trainer_state["torch_random_state"].cpu())

    cuda_state = trainer_state.get("cuda_random_state_all")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)

    print(f"已从checkpoint恢复：{checkpoint_dir}")
    return trainer_state


def write_validation_history(history):
    config.log_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(config.training_history_path, index=False)


def sample_train_dataset(train_dataset):
    """按固定seed抽取训练子集；不修改磁盘上的tokenized数据。"""

    fraction = float(config.train_sample_fraction)
    if not 0 < fraction <= 1:
        raise ValueError(
            "train_sample_fraction必须在(0, 1]之间，"
            f"当前值为{config.train_sample_fraction}"
        )

    original_count = len(train_dataset)
    sample_count = (
        original_count
        if fraction >= 1.0
        else max(1, int(original_count * fraction))
    )
    if sample_count < original_count:
        train_dataset = (
            train_dataset.shuffle(seed=config.seed).select(range(sample_count))
        )

    print(
        f"训练集抽样：fraction={fraction:.4f}，"
        f"原始样本数={original_count}，实际抽样数量={sample_count}"
    )
    return train_dataset, original_count, sample_count


def print_validation_trend(history):
    if not history:
        print("validation历史为空。")
        return

    print("完整validation历史：")
    for item in history:
        print(
            f"global_step={item['global_step']}，"
            f"epoch={item['epoch']}，"
            f"micro_step={item['micro_step_in_epoch']}，"
            f"valid_loss={item['valid_loss']:.6f}，"
            f"is_best={item['is_best']}"
        )


def validate_required_paths():
    """训练启动前检查服务器上必须存在的输入。"""

    required_paths = [
        config.base_model_path / "config.json",
        config.base_model_path / "model.safetensors.index.json",
        config.tokenizer_path,
        config.label_scaler_path,
        config.train_tokenized_path,
        config.valid_tokenized_path,
    ]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "tauijt训练缺少必要输入，请先上传或生成：\n"
            + "\n".join(missing_paths)
        )


# =========================
# 3. 训练
# =========================

def main():
    set_seed(config.seed)
    validate_required_paths()
    resume_checkpoint = get_resume_checkpoint()

    device = get_device()
    model_dtype = get_model_dtype(device)
    amp_dtype = (
        torch.bfloat16
        if model_dtype == torch.bfloat16
        else torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(config.tokenizer_path),
        local_files_only=True,
        trust_remote_code=True,
    )
    tokenizer.padding_side = config.padding_side

    train_dataset = load_from_disk(str(config.train_tokenized_path))
    valid_dataset = load_from_disk(str(config.valid_tokenized_path))
    train_dataset, train_original_count, train_sample_count = (
        sample_train_dataset(train_dataset)
    )

    collator = tauijt_collator(tokenizer)

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = model_module.tauijt_quantile_model(
        model_dtype=model_dtype,
        adapter_path=(
            resume_checkpoint / "lora_adapter"
            if resume_checkpoint is not None
            else None
        ),
        is_trainable=True,
    ).to(device)
    log_gpu_memory("模型加载后显存")

    lora_parameters = [
        parameter
        for parameter in model.backbone.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        [
            {
                "params": lora_parameters,
                "lr": config.lora_learning_rate,
            },
            {
                "params": model.quantile_head.parameters(),
                "lr": config.head_learning_rate,
            },
        ],
        weight_decay=config.weight_decay,
    )

    micro_batches_per_epoch = math.ceil(
        len(train_dataset) / config.train_batch_size
    )
    updates_per_epoch = math.ceil(
        micro_batches_per_epoch / config.gradient_accumulation_steps
    )
    total_updates = updates_per_epoch * config.num_epochs
    warmup_steps = int(total_updates * config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    use_grad_scaler = (
        device.type == "cuda"
        and model_dtype == torch.float16
    )
    grad_scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_grad_scaler,
    )

    config.model_output_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_output_dir.mkdir(parents=True, exist_ok=True)
    config.log_dir.mkdir(parents=True, exist_ok=True)

    best_valid_loss = float("inf")
    history = []
    global_step = 0
    start_epoch = 1
    resume_micro_step = 0

    if resume_checkpoint is not None:
        trainer_state = load_checkpoint_state(
            resume_checkpoint,
            model,
            optimizer,
            scheduler,
            grad_scaler,
            device,
        )
        start_epoch = int(trainer_state["epoch"])
        resume_micro_step = int(trainer_state["micro_step_in_epoch"])
        global_step = int(trainer_state["global_step"])
        best_valid_loss = float(trainer_state["best_valid_loss"])
        history = list(trainer_state["history"])
        write_validation_history(history)
        log_gpu_memory("checkpoint恢复后显存")

    print(
        "tauijt正式训练配置："
        f"train_batch_size={config.train_batch_size}，"
        f"gradient_accumulation_steps={config.gradient_accumulation_steps}，"
        f"effective_batch="
        f"{config.train_batch_size * config.gradient_accumulation_steps}，"
        f"train_sample_fraction={config.train_sample_fraction}，"
        f"train_sample_count={train_sample_count}/{train_original_count}，"
        f"num_epochs={config.num_epochs}，"
        f"gradient_checkpointing={config.use_gradient_checkpointing}，"
        f"LoRA层={config.lora_layer_start}-{config.lora_layer_end}，"
        f"checkpoint_save_steps={config.checkpoint_save_steps}，"
        f"validation_steps={config.validation_steps}"
    )

    for epoch in range(start_epoch, config.num_epochs + 1):
        train_generator = torch.Generator()
        train_generator.manual_seed(config.seed + epoch)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.train_batch_size,
            shuffle=True,
            generator=train_generator,
            collate_fn=collator,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_train_loss = 0.0
        total_train_samples = 0
        valid_window_loss = 0.0
        valid_window_samples = 0

        progress = tqdm(
            train_loader,
            desc=f"训练 Epoch {epoch}/{config.num_epochs}",
            dynamic_ncols=True,
        )

        for step, batch in enumerate(progress, start=1):
            if epoch == start_epoch and step <= resume_micro_step:
                continue

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=device.type == "cuda",
            ):
                predictions = model(input_ids, attention_mask)
                loss, pinball_loss, crossing_loss = (
                    model_module.tauijt_quantile_loss(
                        predictions,
                        labels,
                    )
                )
                scaled_loss = (
                    loss / config.gradient_accumulation_steps
                )

            grad_scaler.scale(scaled_loss).backward()

            batch_size = labels.size(0)
            total_train_loss += loss.item() * batch_size
            total_train_samples += batch_size
            valid_window_loss += loss.item() * batch_size
            valid_window_samples += batch_size

            should_update = (
                step % config.gradient_accumulation_steps == 0
                or step == len(train_loader)
            )

            if should_update:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.max_grad_norm,
                )
                grad_scaler.step(optimizer)
                grad_scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step == 1:
                    log_gpu_memory("第1个optimizer step后显存")

                if global_step % config.validation_steps == 0:
                    valid_loss = evaluate(
                        model,
                        valid_loader,
                        device,
                        amp_dtype,
                    )
                    train_window_loss = (
                        valid_window_loss / valid_window_samples
                        if valid_window_samples > 0
                        else float("nan")
                    )
                    is_best = valid_loss < best_valid_loss
                    if is_best:
                        best_valid_loss = valid_loss
                        save_best_model(
                            model,
                            epoch,
                            global_step,
                            valid_loss,
                        )

                    history.append(
                        {
                            "event": "step_validation",
                            "epoch": epoch,
                            "micro_step_in_epoch": step,
                            "global_step": global_step,
                            "train_loss_since_last_valid": train_window_loss,
                            "valid_loss": valid_loss,
                            "best_valid_loss": best_valid_loss,
                            "is_best": is_best,
                        }
                    )
                    write_validation_history(history)
                    print_validation_trend(history)
                    model.train()
                    valid_window_loss = 0.0
                    valid_window_samples = 0

                if global_step % config.checkpoint_save_steps == 0:
                    save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        grad_scaler,
                        epoch,
                        step,
                        global_step,
                        best_valid_loss,
                        history,
                    )

            progress.set_postfix(
                loss=f"{loss.item():.5f}",
                pinball=f"{pinball_loss.item():.5f}",
                crossing=f"{crossing_loss.item():.5f}",
            )

        if total_train_samples == 0:
            print(
                "当前checkpoint已经位于本epoch末尾，"
                "没有剩余micro-batch可继续训练。"
            )
            print_validation_trend(history)
            return

        train_loss = total_train_loss / total_train_samples
        valid_loss = evaluate(
            model,
            valid_loader,
            device,
            amp_dtype,
        )

        history.append(
            {
                "event": "epoch_end_validation",
                "epoch": epoch,
                "micro_step_in_epoch": len(train_loader),
                "global_step": global_step,
                "train_loss": train_loss,
                "train_loss_since_last_valid": (
                    valid_window_loss / valid_window_samples
                    if valid_window_samples > 0
                    else float("nan")
                ),
                "valid_loss": valid_loss,
                "best_valid_loss": min(best_valid_loss, valid_loss),
                "is_best": valid_loss < best_valid_loss,
            }
        )

        print(
            f"Epoch {epoch}："
            f"训练损失={train_loss:.6f}，"
            f"验证损失={valid_loss:.6f}"
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_best_model(model, epoch, global_step, valid_loss)

        write_validation_history(history)
        print_validation_trend(history)
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            grad_scaler,
            epoch,
            len(train_loader),
            global_step,
            best_valid_loss,
            history,
        )
        print(
            "当前epoch已结束。请检查best_lora、"
            "best_quantile_head和validation历史。"
        )


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError as error:
        print(
            "CUDA OOM：train_batch_size=8、gradient_accumulation_steps=4、"
            "LoRA最后12层、gradient_checkpointing=False时显存不足。"
            "已退出，不自动改动模型或batch。"
        )
        print(str(error))
        sys.exit(2)
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        print(
            "CUDA OOM：train_batch_size=8、gradient_accumulation_steps=4、"
            "LoRA最后12层、gradient_checkpointing=False时显存不足。"
            "已退出，不自动改动模型或batch。"
        )
        print(str(error))
        sys.exit(2)
