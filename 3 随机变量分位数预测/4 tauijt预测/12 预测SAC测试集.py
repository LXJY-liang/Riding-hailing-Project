# -*- coding: utf-8 -*-
"""预测SAC测试集：2024-06-28至2024-06-30。"""

import importlib.util
from pathlib import Path


path = Path(__file__).resolve().parent / "9 预测SAC分组公共函数.py"
spec = importlib.util.spec_from_file_location("tauijt_sac_export", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.export_sac_split("test", "2024-06-28 00:00:00", "2024-06-30 23:45:00")
