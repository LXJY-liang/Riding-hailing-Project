# -*- coding: utf-8 -*-
"""兼容入口：Vit的预测与导出已归还到“4 Vit预训练模型”。"""

from __future__ import annotations

import runpy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET = (
    PROJECT_ROOT / "4 Vit预训练模型" / "2 训练与评估"
    / "5 预测并导出离线SAC使用的Vit分布.py"
)

if __name__ == "__main__":
    print(f"Vit导出逻辑已移至：{TARGET}")
    runpy.run_path(str(TARGET), run_name="__main__")
