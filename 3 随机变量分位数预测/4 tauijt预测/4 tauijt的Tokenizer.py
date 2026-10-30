from pathlib import Path
from datetime import datetime, timezone
import importlib.util
import json
import shutil

from datasets import load_dataset
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
data_module = load_module("2 tauijt数据处理.py", "tauijt_data")


# =========================
# 2. Tokenizer
# =========================

def tokenize_split(tokenizer, data_path, output_path):
    """读取一个数据划分，构造自然语言并完成批量Tokenizer。"""

    dataset = load_dataset(
        "parquet",
        data_files=str(data_path),
        split="train",
    )

    def tokenize_batch(batch):
        texts = data_module.build_text_batch(batch)
        encoded = tokenizer(
            texts,
            max_length=config.max_length,
            truncation=True,
            padding=False,
        )
        encoded["labels"] = batch["label_scaled"]
        return encoded

    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=config.tokenizer_batch_size,
        num_proc=config.tokenizer_num_proc,
        remove_columns=dataset.column_names,
        desc=f"Tokenizer：{data_path.name}",
    )

    temporary_path = output_path.with_name(f"{output_path.name}.building")
    if temporary_path.exists():
        shutil.rmtree(temporary_path)
    tokenized_dataset.save_to_disk(str(temporary_path))

    # 新 Tokenizer 结果完整写出后才替换旧目录，避免中途失败破坏旧产物。
    if output_path.exists():
        shutil.rmtree(output_path)
    temporary_path.replace(output_path)
    return len(tokenized_dataset)


# =========================
# 3. 执行Tokenizer
# =========================

def main():
    if not config.data_manifest_path.is_file():
        raise FileNotFoundError(
            f"缺少最新数据清单：{config.data_manifest_path}。"
            "请先运行‘2 tauijt数据处理.py’。"
        )

    with open(config.data_manifest_path, "r", encoding="utf-8") as file:
        data_manifest = json.load(file)

    tokenizer = AutoTokenizer.from_pretrained(
        str(config.base_model_path),
        local_files_only=True,
        trust_remote_code=True,
    )

    tokenizer.padding_side = config.padding_side

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    config.tokenized_output_dir.mkdir(parents=True, exist_ok=True)
    config.model_output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(config.tokenizer_path))

    split_specs = {
        "train": (config.train_data_path, config.train_tokenized_path),
        "valid": (config.valid_data_path, config.valid_tokenized_path),
        "test": (config.test_data_path, config.test_tokenized_path),
    }
    split_counts = {}
    for split_name, (data_path, tokenized_path) in split_specs.items():
        split_counts[split_name] = tokenize_split(
            tokenizer,
            data_path,
            tokenized_path,
        )
        expected_count = int(data_manifest["split_counts"][split_name])
        if split_counts[split_name] != expected_count:
            raise RuntimeError(
                f"{split_name} Tokenizer样本数={split_counts[split_name]}，"
                f"但数据清单记录={expected_count}。"
            )

    tokenization_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_manifest_path": str(config.data_manifest_path),
        "data_manifest_mtime_ns": config.data_manifest_path.stat().st_mtime_ns,
        "base_model_path": str(config.base_model_path),
        "max_length": int(config.max_length),
        "split_counts": split_counts,
        "tokenized_paths": {
            name: str(path) for name, (_, path) in split_specs.items()
        },
    }
    with open(config.tokenization_manifest_path, "w", encoding="utf-8") as file:
        json.dump(tokenization_manifest, file, ensure_ascii=False, indent=2)
    print(f"Tokenizer来源与样本数清单：{config.tokenization_manifest_path}")


if __name__ == "__main__":
    main()
