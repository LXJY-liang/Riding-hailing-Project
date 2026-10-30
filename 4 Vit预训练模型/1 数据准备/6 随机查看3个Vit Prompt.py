# -*- coding: utf-8 -*-
"""直接从Vit输入特征中随机构造并检查当前版本的完整中文Prompt。"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from transformers import AutoTokenizer


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("vit_pretrain_config", HERE / "1 Vit预训练配置.py")
cfg = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(cfg)

prompt_spec = importlib.util.spec_from_file_location(
    "vit_prompt_builder", HERE / "3 构建Vit中文Prompt与全量样本.py"
)
prompt_builder = importlib.util.module_from_spec(prompt_spec)
assert prompt_spec.loader is not None
prompt_spec.loader.exec_module(prompt_builder)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="随机查看Vit中文Prompt和对应Oracle标签")
    parser.add_argument("--samples", type=int, default=3, help="抽取样本数，默认3。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认42。")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="可选输出文本路径；默认保存到1 数据准备/output/log。",
    )
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples必须为正整数。")
    return args


def main() -> None:
    args = parse_args()
    cfg.ensure_directories()
    days = cfg.days("train")
    files = [
        cfg.INPUT_FEATURE_DIR / f"vit_input_features_{day:%Y-%m-%d}.parquet"
        for day in days
    ]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "缺少Vit输入特征，请先运行“2 构建Vit输入特征.py”：\n"
            + "\n".join(missing[:5])
        )

    row_counts = np.asarray([pq.ParquetFile(path).metadata.num_rows for path in files], np.int64)
    cumulative = np.cumsum(row_counts)
    total_rows = int(cumulative[-1])
    if args.samples > total_rows:
        raise ValueError(f"要抽取{args.samples}行，但数据只有{total_rows}行。")

    rng = np.random.default_rng(args.seed)
    global_positions = np.sort(rng.choice(total_rows, size=args.samples, replace=False))
    selected: list[tuple[int, int, int]] = []
    for sample_number, global_position in enumerate(global_positions, start=1):
        file_index = int(np.searchsorted(cumulative, global_position, side="right"))
        previous = int(cumulative[file_index - 1]) if file_index else 0
        selected.append((sample_number, file_index, int(global_position - previous)))

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.TOKENIZER_DIR,
        local_files_only=True,
        trust_remote_code=True,
    )
    blocks: list[str] = [
        f"Vit Prompt随机检查：seed={args.seed}，样本数={args.samples}，"
        f"全量候选行数={total_rows:,}",
    ]
    cache: dict[int, pd.DataFrame] = {}
    for sample_number, file_index, local_position in selected:
        if file_index not in cache:
            cache[file_index] = pd.read_parquet(files[file_index])
        row = cache[file_index].iloc[local_position]
        prompt = prompt_builder.build_prompt(row)
        token_count = len(
            tokenizer(
                prompt,
                truncation=False,
                padding=False,
            )["input_ids"]
        )
        blocks.extend(
            [
                "",
                "=" * 100,
                f"样本 {sample_number}/{args.samples}",
                f"来源文件：{files[file_index]}",
                f"决策时点：{pd.Timestamp(row['t'])}",
                f"区域：{row['region_id']}",
                f"Prompt字符数：{len(prompt):,}",
                f"Prompt Token数：{token_count:,}",
                "",
                "完整Prompt：",
                prompt,
            ]
        )

    report = "\n".join(blocks) + "\n"
    output = args.output or (
        cfg.DATA_PREPARATION_LOG_DIR
        / f"vit_prompt_samples_seed{args.seed}_n{args.samples}.txt"
    )
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(report)
    print(f"检查报告已保存：{output}")


if __name__ == "__main__":
    main()
