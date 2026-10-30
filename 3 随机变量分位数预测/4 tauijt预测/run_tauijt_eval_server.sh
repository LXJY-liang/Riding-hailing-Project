#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${RIDING_HAILING_PROJECT_ROOT:-/root/autodl-tmp/Riding hailing_Project}"
TAUIJT_DIR="$PROJECT_ROOT/3 随机变量分位数预测/4 tauijt预测"
QWEN_DIR="${QWEN3_4B_BASE_PATH:-$PROJECT_ROOT/0 基座模型_Qwen3-4B-Base}"
SESSION_NAME="${TAUIJT_EVAL_TMUX_SESSION:-tauijt_eval_valid}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-1}"
ATTACH_AFTER_START="${ATTACH_AFTER_START:-1}"
EVAL_TAG="${TAUIJT_EVAL_TAG:-}"
LOG_DIR="$TAUIJT_DIR/output/log"
LOG_FILE="$LOG_DIR/tauijt_eval_valid_$(date +%Y%m%d_%H%M%S).log"

if [ -d "$PROJECT_ROOT/.venv" ]; then
  source "$PROJECT_ROOT/.venv/bin/activate"
elif [ -d "$PROJECT_ROOT/venv" ]; then
  source "$PROJECT_ROOT/venv/bin/activate"
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif [ -x "/root/miniconda3/bin/python" ]; then
    PYTHON_BIN="/root/miniconda3/bin/python"
  elif [ -x "/root/anaconda3/bin/python" ]; then
    PYTHON_BIN="/root/anaconda3/bin/python"
  elif [ -x "/opt/conda/bin/python" ]; then
    PYTHON_BIN="/opt/conda/bin/python"
  else
    echo "找不到python解释器。请先激活虚拟环境，或设置PYTHON_BIN=/path/to/python。" >&2
    exit 1
  fi
fi

mkdir -p "$LOG_DIR"
command -v tmux >/dev/null 2>&1 || {
  echo "找不到tmux，请先安装tmux。" >&2
  exit 1
}

required_paths=(
  "$TAUIJT_DIR/1 tauijt配置.py"
  "$TAUIJT_DIR/3 tauijt模型.py"
  "$TAUIJT_DIR/6 tauijt评估与测试.py"
  "$TAUIJT_DIR/output/data/tauijt_valid.parquet"
  "$TAUIJT_DIR/output/tokenized/tauijt_valid_tokenized"
  "$TAUIJT_DIR/output/model/tokenizer"
  "$TAUIJT_DIR/output/model/label_scaler.json"
  "$TAUIJT_DIR/output/model/best_lora"
  "$TAUIJT_DIR/output/model/best_quantile_head.pt"
  "$QWEN_DIR/config.json"
  "$QWEN_DIR/model.safetensors.index.json"
  "$QWEN_DIR/model-00001-of-00003.safetensors"
  "$QWEN_DIR/model-00002-of-00003.safetensors"
  "$QWEN_DIR/model-00003-of-00003.safetensors"
)
for path in "${required_paths[@]}"; do
  [ -e "$path" ] || { echo "缺少必要文件或目录：$path" >&2; exit 1; }
done

"$PYTHON_BIN" - <<'PY'
import importlib
import sys
from packaging.version import Version

for package in ("torch", "transformers", "datasets", "peft", "pandas", "numpy", "tqdm", "packaging"):
    module = importlib.import_module(package)
    print(f"{package}=={getattr(module, '__version__', 'unknown')}")

import torch
import transformers
if Version(transformers.__version__) < Version("4.51.0"):
    print("transformers版本过低；Qwen3需要transformers>=4.51.0。", file=sys.stderr)
    sys.exit(2)
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("未检测到CUDA；为避免CPU长时间评估，已停止。", file=sys.stderr)
    sys.exit(2)
print(f"gpu={torch.cuda.get_device_name(0)}")
PY

export RIDING_HAILING_PROJECT_ROOT="$PROJECT_ROOT"
export QWEN3_4B_BASE_PATH="$QWEN_DIR"
export PYTHONUNBUFFERED=1

EVAL_SCRIPT="$TAUIJT_DIR/6 tauijt评估与测试.py"
VENV_DIR="$PROJECT_ROOT/.venv"
VENV_ACTIVATE="$VENV_DIR/bin/activate"
LEGACY_VENV_DIR="$PROJECT_ROOT/venv"
LEGACY_VENV_ACTIVATE="$LEGACY_VENV_DIR/bin/activate"
for name in PROJECT_ROOT QWEN_DIR PYTHON_BIN EVAL_SCRIPT LOG_FILE VENV_DIR VENV_ACTIVATE LEGACY_VENV_DIR LEGACY_VENV_ACTIVATE; do
  printf -v "${name}_Q" '%q' "${!name}"
done

EVAL_TAG_ARGUMENT=""
if [ -n "$EVAL_TAG" ]; then
  if [[ ! "$EVAL_TAG" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "TAUIJT_EVAL_TAG只能包含字母、数字、点、下划线和连字符。" >&2
    exit 1
  fi
  printf -v EVAL_TAG_Q '%q' "$EVAL_TAG"
  EVAL_TAG_ARGUMENT="--output-tag $EVAL_TAG_Q"
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session 已存在：$SESSION_NAME" >&2
  echo "进入会话：tmux attach -t '$SESSION_NAME'" >&2
  exit 1
fi

AFTER_EVAL="echo \"评估结束，服务器保持开机。\" | tee -a $LOG_FILE_Q"
if [ "$AUTO_SHUTDOWN" = "1" ]; then
  AFTER_EVAL="echo \"正在执行 /usr/bin/shutdown -h now 关闭实例。\" | tee -a $LOG_FILE_Q; sync; /usr/bin/shutdown -h now"
fi

TMUX_COMMAND="export RIDING_HAILING_PROJECT_ROOT=$PROJECT_ROOT_Q; export QWEN3_4B_BASE_PATH=$QWEN_DIR_Q; cd $PROJECT_ROOT_Q; if [ -d $VENV_DIR_Q ]; then source $VENV_ACTIVATE_Q; elif [ -d $LEGACY_VENV_DIR_Q ]; then source $LEGACY_VENV_ACTIVATE_Q; fi; $PYTHON_BIN_Q $EVAL_SCRIPT_Q --splits valid $EVAL_TAG_ARGUMENT 2>&1 | tee -a $LOG_FILE_Q; eval_status=\${PIPESTATUS[0]}; echo \"Python评估进程已退出，退出码：\$eval_status\" | tee -a $LOG_FILE_Q; $AFTER_EVAL; exit \$eval_status"

tmux new-session -d -s "$SESSION_NAME" "bash -lc '$TMUX_COMMAND'"

echo "tauijt 验证集评估已启动。"
echo "tmux 会话：$SESSION_NAME"
echo "评估日志：$LOG_FILE"
echo "仅评估 valid，不会误跑 test。"
if [ -n "$EVAL_TAG" ]; then
  echo "稳定性实验输出标签：$EVAL_TAG"
fi
if [ "$AUTO_SHUTDOWN" = "1" ]; then
  echo "Python评估进程退出后将自动关机。"
fi
if [ "$ATTACH_AFTER_START" = "1" ]; then
  echo "正在进入tmux实时查看评估进度。"
  echo "如需断开查看但保持评估：按 Ctrl+B，再按 D。"
  exec tmux attach -t "$SESSION_NAME"
fi
