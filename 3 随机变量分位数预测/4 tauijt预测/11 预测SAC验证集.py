# -*- coding: utf-8 -*-
"""预测SAC验证集：2024-06-25至2024-06-27。"""

import importlib.util
from pathlib import Path


path = Path(__file__).resolve().parent / "9 预测SAC分组公共函数.py"
spec = importlib.util.spec_from_file_location("tauijt_sac_export", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.export_sac_split("valid", "2024-06-25 00:00:00", "2024-06-27 23:45:00")
