#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${RIDING_HAILING_PROJECT_ROOT:-/root/autodl-tmp/Riding hailing_Project}"
TAUIJT_DIR="$PROJECT_ROOT/3 随机变量分位数预测/4 tauijt预测"
QWEN_DIR="${QWEN3_4B_BASE_PATH:-$PROJECT_ROOT/0 基座模型_Qwen3-4B-Base}"
SESSION_NAME="${TAUIJT_TMUX_SESSION:-tauijt_train}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-1}"
ATTACH_AFTER_START="${ATTACH_AFTER_START:-1}"
LOG_DIR="$TAUIJT_DIR/output/log"
LOG_FILE="$LOG_DIR/tauijt_train_$(date +%Y%m%d_%H%M%S).log"

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
    echo "找不到python解释器。请先激活虚拟环境，或设置PYTHON_BIN=/path/to/python。"
    exit 1
  fi
fi

mkdir -p "$LOG_DIR"

if ! command -v tmux >/dev/null 2>&1; then
  echo "找不到tmux，请先安装tmux。"
  exit 1
fi

required_paths=(
  "$TAUIJT_DIR/1 tauijt配置.py"
  "$TAUIJT_DIR/3 tauijt模型.py"
  "$TAUIJT_DIR/5 tauijt训练.py"
  "$TAUIJT_DIR/output/tokenized/tauijt_train_tokenized"
  "$TAUIJT_DIR/output/tokenized/tauijt_valid_tokenized"
  "$TAUIJT_DIR/output/model/tokenizer"
  "$TAUIJT_DIR/output/model/label_scaler.json"
  "$QWEN_DIR/config.json"
  "$QWEN_DIR/model.safetensors.index.json"
  "$QWEN_DIR/model-00001-of-00003.safetensors"
  "$QWEN_DIR/model-00002-of-00003.safetensors"
  "$QWEN_DIR/model-00003-of-00003.safetensors"
)

for path in "${required_paths[@]}"; do
  if [ ! -e "$path" ]; then
    echo "缺少必要文件或目录：$path"
    exit 1
  fi
done

"$PYTHON_BIN" - <<'PY'
import importlib
import sys
from packaging.version import Version

packages = [
    "torch", "transformers", "datasets", "peft", "pandas", "numpy",
    "tqdm", "packaging",
]
for package in packages:
    module = importlib.import_module(package)
    version = getattr(module, "__version__", "unknown")
    print(f"{package}=={version}")

import transformers
if Version(transformers.__version__) < Version("4.51.0"):
    print(
        "transformers版本过低；Qwen3需要transformers>=4.51.0。",
        file=sys.stderr,
    )
    sys.exit(2)

import torch
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
elif not bool(int(__import__("os").environ.get("ALLOW_NON_CUDA", "0"))):
    print("未检测到CUDA；为避免误用CPU长时训练，已停止。", file=sys.stderr)
    sys.exit(2)
PY

export RIDING_HAILING_PROJECT_ROOT="$PROJECT_ROOT"
export QWEN3_4B_BASE_PATH="$QWEN_DIR"
export PYTHONUNBUFFERED=1

TRAIN_SCRIPT="$TAUIJT_DIR/5 tauijt训练.py"
VENV_DIR="$PROJECT_ROOT/.venv"
VENV_ACTIVATE="$VENV_DIR/bin/activate"
LEGACY_VENV_DIR="$PROJECT_ROOT/venv"
LEGACY_VENV_ACTIVATE="$LEGACY_VENV_DIR/bin/activate"

printf -v PROJECT_ROOT_Q "%q" "$PROJECT_ROOT"
printf -v QWEN_DIR_Q "%q" "$QWEN_DIR"
printf -v PYTHON_BIN_Q "%q" "$PYTHON_BIN"
printf -v TRAIN_SCRIPT_Q "%q" "$TRAIN_SCRIPT"
printf -v LOG_FILE_Q "%q" "$LOG_FILE"
printf -v VENV_DIR_Q "%q" "$VENV_DIR"
printf -v VENV_ACTIVATE_Q "%q" "$VENV_ACTIVATE"
printf -v LEGACY_VENV_DIR_Q "%q" "$LEGACY_VENV_DIR"
printf -v LEGACY_VENV_ACTIVATE_Q "%q" "$LEGACY_VENV_ACTIVATE"

RESUME_EXPORT=""
if [ -n "${TAUIJT_RESUME_FROM_CHECKPOINT:-}" ]; then
  printf -v RESUME_FROM_CHECKPOINT_Q "%q" "$TAUIJT_RESUME_FROM_CHECKPOINT"
  RESUME_EXPORT="export TAUIJT_RESUME_FROM_CHECKPOINT=$RESUME_FROM_CHECKPOINT_Q;"
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session 已存在：$SESSION_NAME"
  echo "进入会话：tmux attach -t '$SESSION_NAME'"
  exit 1
fi

AFTER_TRAIN="echo \"已保留服务器开机；如需关机请手动操作。\" | tee -a $LOG_FILE_Q"
if [ "$AUTO_SHUTDOWN" = "1" ]; then
  AFTER_TRAIN="echo \"正在执行 /usr/bin/shutdown -h now 关闭实例。\" | tee -a $LOG_FILE_Q; sync; /usr/bin/shutdown -h now"
fi

TMUX_COMMAND="export RIDING_HAILING_PROJECT_ROOT=$PROJECT_ROOT_Q; export QWEN3_4B_BASE_PATH=$QWEN_DIR_Q; $RESUME_EXPORT cd $PROJECT_ROOT_Q; if [ -d $VENV_DIR_Q ]; then source $VENV_ACTIVATE_Q; elif [ -d $LEGACY_VENV_DIR_Q ]; then source $LEGACY_VENV_ACTIVATE_Q; fi; $PYTHON_BIN_Q $TRAIN_SCRIPT_Q 2>&1 | tee -a $LOG_FILE_Q; train_status=\${PIPESTATUS[0]}; echo \"Python训练进程已退出，退出码：\$train_status\" | tee -a $LOG_FILE_Q; $AFTER_TRAIN; exit \$train_status"

tmux new-session -d -s "$SESSION_NAME" "bash -lc '$TMUX_COMMAND'"

echo "tauijt 正式训练已启动。"
echo "tmux 会话：$SESSION_NAME"
echo "训练日志：$LOG_FILE"
echo "进入会话：tmux attach -t '$SESSION_NAME'"
echo "后台看日志：tail -f '$LOG_FILE'"
if [ "$AUTO_SHUTDOWN" = "1" ]; then
  echo "Python训练进程退出后将自动关机。"
else
  echo "训练结束后不会自动关机。"
fi

if [ "$ATTACH_AFTER_START" = "1" ]; then
  echo "正在进入tmux实时查看训练进度。"
  echo "如需断开查看但保持训练：按 Ctrl+B，再按 D。"
  exec tmux attach -t "$SESSION_NAME"
fi
