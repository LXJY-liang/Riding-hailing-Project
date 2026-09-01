from pathlib import Path
import importlib.util

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


config = load_module("1 Dit配置.py", "dit_config")
data_module = load_module("2 Dit数据处理.py", "dit_data")


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

    tokenized_dataset.save_to_disk(str(output_path))


# =========================
# 3. 执行Tokenizer
# =========================

def main():
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

    tokenize_split(
        tokenizer,
        config.train_data_path,
        config.train_tokenized_path,
    )
    tokenize_split(
        tokenizer,
        config.valid_data_path,
        config.valid_tokenized_path,
    )
    tokenize_split(
        tokenizer,
        config.test_data_path,
        config.test_tokenized_path,
    )


if __name__ == "__main__":
    main()
