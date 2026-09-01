from pathlib import Path
import importlib.util

import pyarrow.parquet as pq
from transformers import AutoTokenizer


# =========================
# 1. 项目路径
# =========================

project_root = Path(
    "/Users/liangxue/Documents/8、Python文件-网约车数据/"
    "Riding hailing_Project"
)

prediction_root = project_root / "3 随机变量分位数预测"

batch_size = 2000


# =========================
# 2. 工具函数
# =========================

def load_module(file_path, module_name):
    """按文件路径加载Python模块。"""

    spec = importlib.util.spec_from_file_location(
        module_name,
        file_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def update_statistics(
    tokenizer,
    texts,
    statistics,
    token_limit,
):
    """统计真实Token长度，不截断、不padding、不保存Token。"""

    lengths = tokenizer(
        texts,
        truncation=False,
        padding=False,
        return_length=True,
    )["length"]

    statistics["total"] += len(lengths)
    statistics["over_limit"] += sum(
        length > token_limit
        for length in lengths
    )
    statistics["max_length"] = max(
        statistics["max_length"],
        max(lengths),
    )


def print_statistics(
    model_name,
    statistics,
    token_limit,
):
    """输出一个随机变量的Token长度统计。"""

    print(model_name)
    print(f"总样本数：{statistics['total']}")
    print(
        f"超过{token_limit} Token的样本数："
        f"{statistics['over_limit']}"
    )
    print(f"最长Token数：{statistics['max_length']}")
    print()


# =========================
# 3. Dit、Sit、tau_ijt、p^e_ijt、c_ijt
# =========================

def analyze_standard_model(
    model_name,
    folder_name,
    config_name,
    data_name,
):
    """统计标准连续随机变量模型实际进入Tokenizer的数据。"""

    model_dir = prediction_root / folder_name

    config = load_module(
        model_dir / config_name,
        f"{model_name}_config",
    )

    data_module = load_module(
        model_dir / data_name,
        f"{model_name}_data",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(config.base_model_path),
        local_files_only=True,
        trust_remote_code=True,
    )

    token_limit = config.max_length

    statistics = {
        "total": 0,
        "over_limit": 0,
        "max_length": 0,
    }

    for data_path in [
        config.train_data_path,
        config.valid_data_path,
        config.test_data_path,
    ]:
        parquet_file = pq.ParquetFile(data_path)

        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=config.feature_columns,
        ):
            texts = data_module.build_text_batch(
                batch.to_pydict()
            )

            update_statistics(
                tokenizer,
                texts,
                statistics,
                token_limit,
            )

    print_statistics(
        model_name,
        statistics,
        token_limit,
    )


# =========================
# 4. πijt
# =========================

def analyze_pi():
    """统计πijt文本分支实际进入Tokenizer的数据。"""

    model_dir = prediction_root / "2 πijt预测"

    config = load_module(
        model_dir / "1 πijt配置.py",
        "pi_config",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(config.BASE_MODEL_PATH),
        local_files_only=True,
        trust_remote_code=True,
    )

    token_limit = config.MAX_LENGTH

    statistics = {
        "total": 0,
        "over_limit": 0,
        "max_length": 0,
    }

    for data_path in [
        config.TRAIN_META_FILE,
        config.VALID_META_FILE,
        config.TEST_META_FILE,
    ]:
        parquet_file = pq.ParquetFile(data_path)

        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=["text"],
        ):
            texts = batch.column(0).to_pylist()

            update_statistics(
                tokenizer,
                texts,
                statistics,
                token_limit,
            )

    print_statistics(
        "πijt",
        statistics,
        token_limit,
    )


# =========================
# 5. 执行
# =========================

def main():
    analyze_standard_model(
        "Dit",
        "1 Dit预测",
        "1 Dit配置.py",
        "2 Dit数据处理.py",
    )

    analyze_standard_model(
        "Sit",
        "3 Sit预测",
        "1 Sit配置.py",
        "2 Sit数据处理.py",
    )

    analyze_pi()

    analyze_standard_model(
        "tau_ijt",
        "4 tauijt预测",
        "1 tauijt配置.py",
        "2 tauijt数据处理.py",
    )

    analyze_standard_model(
        "p^e_ijt",
        "5 p^eijt预测",
        "1 p^eijt配置.py",
        "2 p^eijt数据处理.py",
    )

    analyze_standard_model(
        "c_ijt",
        "6 c_ijt预测",
        "1 c_ijt配置.py",
        "2 c_ijt数据处理.py",
    )


if __name__ == "__main__":
    main()
