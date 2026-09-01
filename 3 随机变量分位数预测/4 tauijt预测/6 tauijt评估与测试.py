import argparse
from pathlib import Path
import importlib.util
import json

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer


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
# 2. 评估辅助函数
# =========================

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
    """对预测批次进行动态补齐。"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, samples):
        model_inputs = [
            {
                "input_ids": sample["input_ids"],
                "attention_mask": sample["attention_mask"],
            }
            for sample in samples
        ]

        return self.tokenizer.pad(
            model_inputs,
            padding=True,
            return_tensors="pt",
        )


def load_scaler():
    with open(
        config.label_scaler_path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def inverse_transform(values, scaler):
    """只执行标签反变换，不进行单调修正或非负截断。"""

    values = np.asarray(values, dtype=np.float32)

    if scaler["use_standardization"]:
        values = (
            values * scaler["label_std"]
            + scaler["label_mean"]
        )

    if scaler["use_log1p"]:
        values = np.expm1(values)

    return values


def predict_split(model, tokenizer, tokenized_path, device):
    dataset = load_from_disk(str(tokenized_path))
    data_loader = DataLoader(
        dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=tauijt_collator(tokenizer),
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    prediction_batches = []

    with torch.inference_mode():
        for batch in tqdm(
            data_loader,
            desc=f"预测：{tokenized_path.name}",
            dynamic_ncols=True,
        ):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            predictions = model(input_ids, attention_mask)
            prediction_batches.append(
                predictions.float().cpu().numpy()
            )

    return np.concatenate(prediction_batches, axis=0)


def calculate_metrics(labels, predictions):
    """计算中位数误差、校准误差、区间覆盖率和交叉率。"""

    levels = np.asarray(config.quantile_levels, dtype=np.float32)
    median_index = int(np.argmin(np.abs(levels - 0.50)))
    median_prediction = predictions[:, median_index]

    errors = labels[:, None] - predictions
    pinball = np.maximum(
        levels[None, :] * errors,
        (levels[None, :] - 1.0) * errors,
    )

    actual_coverage = np.mean(
        labels[:, None] <= predictions,
        axis=0,
    )

    crossing = predictions[:, :-1] > predictions[:, 1:]

    metrics = {
        "sample_count": int(len(labels)),
        "mean_pinball_loss": float(pinball.mean()),
        "median_mae": float(
            np.mean(np.abs(labels - median_prediction))
        ),
        "median_rmse": float(
            np.sqrt(np.mean((labels - median_prediction) ** 2))
        ),
        "mean_calibration_error": float(
            np.mean(np.abs(actual_coverage - levels))
        ),
        "crossing_pair_rate": float(crossing.mean()),
        "crossing_sample_rate": float(
            crossing.any(axis=1).mean()
        ),
    }

    for name, lower_level, upper_level in [
        ("50", 0.25, 0.75),
        ("80", 0.10, 0.90),
        ("90", 0.05, 0.95),
        ("98", 0.01, 0.99),
    ]:
        lower_index = int(np.argmin(np.abs(levels - lower_level)))
        upper_index = int(np.argmin(np.abs(levels - upper_level)))
        lower = predictions[:, lower_index]
        upper = predictions[:, upper_index]

        metrics[f"interval_{name}_coverage"] = float(
            np.mean((labels >= lower) & (labels <= upper))
        )
        metrics[f"interval_{name}_mean_width"] = float(
            np.mean(upper - lower)
        )

    coverage = pd.DataFrame(
        {
            "quantile_level": levels,
            "actual_coverage": actual_coverage,
            "coverage_error": actual_coverage - levels,
        }
    )

    return metrics, coverage


def evaluate_split(
    split_name,
    raw_data_path,
    tokenized_path,
    model,
    tokenizer,
    scaler,
    device,
):
    """预测一个数据划分并保存完整原始分位数。"""

    raw_data = pd.read_parquet(
        raw_data_path,
        columns=[
            config.i_column,
            config.j_column,
            config.time_column,
            config.od_id_column,
            config.label_column,
        ],
    ).reset_index(drop=True)

    predictions_scaled = predict_split(
        model,
        tokenizer,
        tokenized_path,
        device,
    )
    predictions = inverse_transform(predictions_scaled, scaler)
    labels = raw_data[config.label_column].to_numpy(dtype=np.float32)

    metrics, coverage = calculate_metrics(labels, predictions)

    result = raw_data.copy()
    result = pd.concat(
        [
            result,
            pd.DataFrame(
                predictions,
                columns=config.quantile_names,
            ),
        ],
        axis=1,
    )

    config.prediction_output_dir.mkdir(parents=True, exist_ok=True)
    config.log_dir.mkdir(parents=True, exist_ok=True)

    prediction_path = (
        config.prediction_output_dir
        / f"tauijt_{split_name}_predictions.parquet"
    )
    metrics_path = config.log_dir / f"tauijt_{split_name}_metrics.json"
    coverage_path = config.log_dir / f"tauijt_{split_name}_coverage.csv"

    result.to_parquet(prediction_path, index=False)
    coverage.to_csv(coverage_path, index=False)

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    print(
        f"{split_name}：样本数={metrics['sample_count']}，"
        f"Pinball={metrics['mean_pinball_loss']:.6f}，"
        f"中位数MAE={metrics['median_mae']:.6f}，"
        f"中位数RMSE={metrics['median_rmse']:.6f}，"
        f"样本交叉率={metrics['crossing_sample_rate']:.6f}"
    )
    print(f"预测结果：{prediction_path}")


# =========================
# 3. 执行指定数据划分的评估
# =========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="评估 tauijt 分位数预测模型。",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("valid", "test"),
        default=("valid", "test"),
        help="要评估的数据划分；默认依次评估 valid 和 test。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    model_dtype = get_model_dtype(device)

    print(f"评估设备：{device}；评估划分：{', '.join(args.splits)}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(config.tokenizer_path),
        local_files_only=True,
        trust_remote_code=True,
    )
    tokenizer.padding_side = config.padding_side

    model = model_module.tauijt_quantile_model(
        model_dtype=model_dtype,
        adapter_path=config.lora_adapter_path,
        is_trainable=False,
    )
    model.quantile_head.load_state_dict(
        torch.load(
            config.quantile_head_path,
            map_location="cpu",
        )
    )
    model = model.to(device)
    model.eval()

    scaler = load_scaler()

    split_paths = {
        "valid": (config.valid_data_path, config.valid_tokenized_path),
        "test": (config.test_data_path, config.test_tokenized_path),
    }
    for split_name in args.splits:
        raw_data_path, tokenized_path = split_paths[split_name]
        evaluate_split(
            split_name,
            raw_data_path,
            tokenized_path,
            model,
            tokenizer,
            scaler,
            device,
        )


if __name__ == "__main__":
    main()
