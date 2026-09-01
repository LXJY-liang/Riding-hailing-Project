from pathlib import Path
import importlib.util

import pandas as pd
import pyarrow.dataset as ds
from transformers import AutoTokenizer


# =========================
# 1. 项目路径与检查样本
# =========================

project_root = Path(
    "/Users/liangxue/Documents/8、Python文件-网约车数据/"
    "Riding hailing_Project"
)

target_region = "17031839100"
target_i = "17031839100"
target_j = "17031833000"
target_time = "2024-06-06 18:15:00"


# =========================
# 2. 工具函数
# =========================

def load_module(file_path, module_name):
    spec = importlib.util.spec_from_file_location(
        module_name,
        file_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def print_token_result(
    model_name,
    text,
    tokenizer,
    max_length,
):
    """显示原始文本、Token长度和模型截断后实际看到的文本。"""

    full_ids = tokenizer(
        text,
        truncation=False,
        padding=False,
    )["input_ids"]

    encoded = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding=False,
    )

    decoded = tokenizer.decode(
        encoded["input_ids"],
        skip_special_tokens=True,
    )

    print("\n" + "=" * 80)
    print(model_name)
    print("=" * 80)

    print("\n转换后的文本：")
    print(text)

    print(f"\n原始Token数：{len(full_ids)}")
    print(f"模型最大Token数：{max_length}")
    print(
        f"是否发生截断："
        f"{'是' if len(full_ids) > max_length else '否'}"
    )

    print("\n模型实际看到的文本：")
    print(decoded)


# =========================
# 3. Dit
# =========================

dit_dir = (
    project_root
    / "3 随机变量分位数预测"
    / "1 Dit预测"
)

dit_config = load_module(
    dit_dir / "1 Dit配置.py",
    "dit_config",
)

dit_data = load_module(
    dit_dir / "2 Dit数据处理.py",
    "dit_data",
)

dataset = ds.dataset(
    str(dit_config.feature_data_path),
    format="parquet",
)

row = dataset.to_table(
    columns=dit_config.all_columns,
    filter=(
        (
            ds.field(dit_config.region_column)
            == str(target_region)
        )
        & (
            ds.field(dit_config.time_column)
            == pd.Timestamp(target_time).to_pydatetime()
        )
    ),
).to_pandas().iloc[0]

tokenizer = AutoTokenizer.from_pretrained(
    str(dit_config.base_model_path),
    local_files_only=True,
    trust_remote_code=True,
)

print_token_result(
    "Dit",
    dit_data.build_text(row),
    tokenizer,
    dit_config.max_length,
)


# =========================
# 4. Sit
# =========================

sit_dir = (
    project_root
    / "3 随机变量分位数预测"
    / "3 Sit预测"
)

sit_config = load_module(
    sit_dir / "1 Sit配置.py",
    "sit_config",
)

sit_data = load_module(
    sit_dir / "2 Sit数据处理.py",
    "sit_data",
)

dataset = ds.dataset(
    str(sit_config.feature_data_path),
    format="parquet",
)

row = dataset.to_table(
    columns=sit_config.all_columns,
    filter=(
        (
            ds.field(sit_config.region_column)
            == str(target_region)
        )
        & (
            ds.field(sit_config.time_column)
            == pd.Timestamp(target_time).to_pydatetime()
        )
    ),
).to_pandas().iloc[0]

tokenizer = AutoTokenizer.from_pretrained(
    str(sit_config.base_model_path),
    local_files_only=True,
    trust_remote_code=True,
)

print_token_result(
    "Sit",
    sit_data.build_text(row),
    tokenizer,
    sit_config.max_length,
)


# =========================
# 5. πijt
# =========================

pi_dir = (
    project_root
    / "3 随机变量分位数预测"
    / "2 πijt预测"
)

pi_config = load_module(
    pi_dir / "1 πijt配置.py",
    "pi_config",
)

pi_data = load_module(
    pi_dir / "2 πijt数据处理.py",
    "pi_data",
)

dataset = ds.dataset(
    str(pi_config.SCALAR_FILE),
    format="parquet",
)

row_df = dataset.to_table(
    columns=pi_data.SCALAR_COLUMNS,
    filter=(
        (ds.field("i") == str(target_i))
        & (
            ds.field("t")
            == pd.Timestamp(target_time).to_pydatetime()
        )
    ),
).to_pandas()

row = next(
    row_df.itertuples(index=False)
)

tokenizer = AutoTokenizer.from_pretrained(
    str(pi_config.BASE_MODEL_PATH),
    local_files_only=True,
    trust_remote_code=True,
)

print_token_result(
    "πijt文本分支",
    pi_data.build_text(row),
    tokenizer,
    pi_config.MAX_LENGTH,
)

print(
    "\nπijt说明：9组801维历史目的地比例不转换成文本，"
    "而是作为9个数值Token与文本Token共同进入Qwen。"
)


# =========================
# 6. tau_ijt、p^e_ijt、c_ijt
# =========================

od_models = [
    (
        "tau_ijt",
        "4 tauijt预测",
        "1 tauijt配置.py",
        "2 tauijt数据处理.py",
    ),
    (
        "p^e_ijt",
        "5 p^eijt预测",
        "1 p^eijt配置.py",
        "2 p^eijt数据处理.py",
    ),
    (
        "c_ijt",
        "6 c_ijt预测",
        "1 c_ijt配置.py",
        "2 c_ijt数据处理.py",
    ),
]

for (
    model_name,
    folder_name,
    config_name,
    data_name,
) in od_models:

    model_dir = (
        project_root
        / "3 随机变量分位数预测"
        / folder_name
    )

    config = load_module(
        model_dir / config_name,
        f"{model_name}_config",
    )

    data_module = load_module(
        model_dir / data_name,
        f"{model_name}_data",
    )

    dataset = ds.dataset(
        str(config.feature_data_path),
        format="parquet",
    )

    row = dataset.to_table(
        columns=config.all_columns,
        filter=(
            (
                ds.field(config.i_column)
                == str(target_i)
            )
            & (
                ds.field(config.j_column)
                == str(target_j)
            )
            & (
                ds.field(config.time_column)
                == pd.Timestamp(target_time).to_pydatetime()
            )
        ),
    ).to_pandas().iloc[0]

    tokenizer = AutoTokenizer.from_pretrained(
        str(config.base_model_path),
        local_files_only=True,
        trust_remote_code=True,
    )

    print_token_result(
        model_name,
        data_module.build_text(row),
        tokenizer,
        config.max_length,
    )
