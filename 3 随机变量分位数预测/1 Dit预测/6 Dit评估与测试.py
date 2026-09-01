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


config = load_module("1 Dit配置.py", "dit_config")
model_module = load_module("3 Dit模型.py", "dit_model")


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


class dit_collator:
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


def torch_load(path, map_location):
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=True,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=map_location,
        )


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
        collate_fn=dit_collator(tokenizer),
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


def get_crps_weights(levels):
    """为离散分位点生成CRPS数值积分权重。"""

    levels = np.asarray(levels, dtype=np.float64)
    weights = np.empty_like(levels)
    weights[0] = (levels[0] + levels[1]) / 2.0
    weights[-1] = 1.0 - (levels[-2] + levels[-1]) / 2.0
    weights[1:-1] = (levels[2:] - levels[:-2]) / 2.0
    return weights.astype(np.float32)


def calculate_pinball(labels, predictions, levels):
    errors = labels[:, None] - predictions
    return np.maximum(
        levels[None, :] * errors,
        (levels[None, :] - 1.0) * errors,
    )


def calculate_forecasting_indicators(labels, predictions):
    """计算总体和按真实Dit区间分组的MAE、RMSE、MPL、CRPS。"""

    levels = np.asarray(config.quantile_levels, dtype=np.float32)
    median_index = int(np.argmin(np.abs(levels - 0.50)))
    median_prediction = predictions[:, median_index]
    crps_weights = get_crps_weights(levels)
    pinball = calculate_pinball(labels, predictions, levels)

    groups = [
        ("Overall", np.ones(len(labels), dtype=bool)),
        ("D=0", labels == 0),
        ("1<=D<=2", (labels >= 1) & (labels <= 2)),
        ("3<=D<=5", (labels >= 3) & (labels <= 5)),
        ("6<=D<=10", (labels >= 6) & (labels <= 10)),
        ("11<=D<=35", (labels >= 11) & (labels <= 35)),
        ("D>=36", labels >= 36),
    ]

    rows = []
    for group_name, mask in groups:
        group_labels = labels[mask]
        group_median = median_prediction[mask]
        group_pinball = pinball[mask]

        if len(group_labels) == 0:
            row = {
                "D_it_group": group_name,
                "sample_count": 0,
                "sample_share": 0.0,
                "label_mean": np.nan,
                "q50_mean": np.nan,
                "MAE_q50": np.nan,
                "RMSE_q50": np.nan,
                "Mean_Pinball_Loss": np.nan,
                "CRPS_approx": np.nan,
                "Bias_q50_pred_minus_true": np.nan,
                "Under_prediction_rate_q50": np.nan,
                "Over_prediction_rate_q50": np.nan,
            }
        else:
            row = {
                "D_it_group": group_name,
                "sample_count": int(len(group_labels)),
                "sample_share": float(mask.mean()),
                "label_mean": float(group_labels.mean()),
                "q50_mean": float(group_median.mean()),
                "MAE_q50": float(
                    np.mean(np.abs(group_labels - group_median))
                ),
                "RMSE_q50": float(
                    np.sqrt(np.mean((group_labels - group_median) ** 2))
                ),
                "Mean_Pinball_Loss": float(group_pinball.mean()),
                "CRPS_approx": float(
                    2.0
                    * np.mean(
                        np.sum(
                            group_pinball * crps_weights[None, :],
                            axis=1,
                        )
                    )
                ),
                "Bias_q50_pred_minus_true": float(
                    np.mean(group_median - group_labels)
                ),
                "Under_prediction_rate_q50": float(
                    np.mean(group_median < group_labels)
                ),
                "Over_prediction_rate_q50": float(
                    np.mean(group_median > group_labels)
                ),
            }
        rows.append(row)

    return pd.DataFrame(rows), pinball


def calculate_metrics(labels, predictions):
    """计算预测指标、校准误差、区间覆盖率和输出合法性。"""

    levels = np.asarray(config.quantile_levels, dtype=np.float32)
    forecasting_indicators, pinball = calculate_forecasting_indicators(
        labels,
        predictions,
    )

    actual_coverage = np.mean(
        labels[:, None] <= predictions,
        axis=0,
    )

    crossing = predictions[:, :-1] > predictions[:, 1:]
    negative = predictions < 0
    overall = forecasting_indicators.iloc[0]

    metrics = {
        "sample_count": int(overall["sample_count"]),
        "mean_pinball_loss": float(overall["Mean_Pinball_Loss"]),
        "crps_approx": float(overall["CRPS_approx"]),
        "median_mae": float(overall["MAE_q50"]),
        "median_rmse": float(overall["RMSE_q50"]),
        "mean_calibration_error": float(
            np.mean(np.abs(actual_coverage - levels))
        ),
        "crossing_pair_rate": float(crossing.mean()),
        "crossing_sample_rate": float(
            crossing.any(axis=1).mean()
        ),
        "negative_prediction_rate_all_quantiles": float(
            negative.mean()
        ),
        "negative_sample_rate_any_quantile": float(
            negative.any(axis=1).mean()
        ),
        "forecasting_indicators": forecasting_indicators.to_dict(
            orient="records"
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

    return metrics, coverage, forecasting_indicators


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
            config.region_column,
            config.time_column,
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

    metrics, coverage, forecasting_indicators = calculate_metrics(
        labels,
        predictions,
    )

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
        / f"dit_{split_name}_predictions.parquet"
    )
    metrics_path = config.log_dir / f"dit_{split_name}_metrics.json"
    coverage_path = config.log_dir / f"dit_{split_name}_coverage.csv"
    forecasting_path = (
        config.log_dir
        / f"dit_{split_name}_forecasting_indicators.csv"
    )

    result.to_parquet(prediction_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    forecasting_indicators.to_csv(forecasting_path, index=False)

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    print(
        f"{split_name}：样本数={metrics['sample_count']}，"
        f"MPL={metrics['mean_pinball_loss']:.6f}，"
        f"CRPS={metrics['crps_approx']:.6f}，"
        f"中位数MAE={metrics['median_mae']:.6f}，"
        f"中位数RMSE={metrics['median_rmse']:.6f}，"
        f"样本交叉率={metrics['crossing_sample_rate']:.6f}，"
        f"负预测样本率="
        f"{metrics['negative_sample_rate_any_quantile']:.6f}"
    )
    print("Forecasting indicators：")
    print(forecasting_indicators.to_string(index=False))
    print(f"预测结果：{prediction_path}")
    print(f"分组预测指标：{forecasting_path}")


def validate_required_paths():
    """评估启动前检查验证集评估必须存在的输入。"""

    required_paths = [
        config.base_model_path / "config.json",
        config.base_model_path / "model.safetensors.index.json",
        config.tokenizer_path,
        config.valid_tokenized_path,
        config.valid_data_path,
        config.label_scaler_path,
        config.lora_adapter_path,
        config.quantile_head_path,
    ]
    missing_paths = [
        str(path)
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Dit验证集评估缺少必要输入，请先上传或生成：\n"
            + "\n".join(missing_paths)
        )


# =========================
# 3. 执行验证集评估
# =========================

def main():
    validate_required_paths()

    device = get_device()
    model_dtype = get_model_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(
        str(config.tokenizer_path),
        local_files_only=True,
        trust_remote_code=True,
    )
    tokenizer.padding_side = config.padding_side

    model = model_module.dit_quantile_model(
        model_dtype=model_dtype,
        adapter_path=config.lora_adapter_path,
        is_trainable=False,
    )
    model.quantile_head.load_state_dict(
        torch_load(
            config.quantile_head_path,
            map_location="cpu",
        )
    )
    model = model.to(device)
    model.eval()

    scaler = load_scaler()

    evaluate_split(
        "valid",
        config.valid_data_path,
        config.valid_tokenized_path,
        model,
        tokenizer,
        scaler,
        device,
    )


if __name__ == "__main__":
    main()
