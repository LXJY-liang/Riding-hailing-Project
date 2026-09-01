#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${RIDING_HAILING_PROJECT_ROOT:-/root/autodl-tmp/Riding hailing_Project}"
DIT_DIR="$PROJECT_ROOT/3 随机变量分位数预测/1 Dit预测"
QWEN_DIR="${QWEN3_4B_BASE_PATH:-$PROJECT_ROOT/0 基座模型_Qwen3-4B-Base}"
ARTIFACT_VERSION="${DIT_ARTIFACT_VERSION:-current}"
if [ "$ARTIFACT_VERSION" = "current" ]; then
  ARTIFACT_DIR="$DIT_DIR/output"
else
  ARTIFACT_DIR="$DIT_DIR/output/$ARTIFACT_VERSION"
fi
SESSION_NAME="${DIT_TMUX_SESSION:-dit_train}"
LOG_DIR="$ARTIFACT_DIR/log"
LOG_FILE="$LOG_DIR/dit_train_$(date +%Y%m%d_%H%M%S).log"

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

required_paths=(
  "$DIT_DIR/5 Dit训练.py"
  "$ARTIFACT_DIR/tokenized/dit_train_tokenized"
  "$ARTIFACT_DIR/tokenized/dit_valid_tokenized"
  "$ARTIFACT_DIR/model/tokenizer"
  "$ARTIFACT_DIR/model/label_scaler.json"
  "$QWEN_DIR/config.json"
  "$QWEN_DIR/model.safetensors.index.json"
)

for path in "${required_paths[@]}"; do
  if [ ! -e "$path" ]; then
    echo "缺少必要文件或目录：$path"
    exit 1
  fi
done

"$PYTHON_BIN" - <<'PY'
import importlib

packages = ["torch", "transformers", "datasets", "peft", "pandas", "numpy", "tqdm"]
for package in packages:
    module = importlib.import_module(package)
    version = getattr(module, "__version__", "unknown")
    print(f"{package}=={version}")
PY

export RIDING_HAILING_PROJECT_ROOT="$PROJECT_ROOT"
export QWEN3_4B_BASE_PATH="$QWEN_DIR"
export DIT_ARTIFACT_VERSION="$ARTIFACT_VERSION"
export DIT_DIR
export LOG_FILE
export PYTHON_BIN
export PYTHONUNBUFFERED=1

TRAIN_SCRIPT="$DIT_DIR/5 Dit训练.py"
VENV_DIR="$PROJECT_ROOT/.venv"
VENV_ACTIVATE="$VENV_DIR/bin/activate"
LEGACY_VENV_DIR="$PROJECT_ROOT/venv"
LEGACY_VENV_ACTIVATE="$LEGACY_VENV_DIR/bin/activate"

printf -v PROJECT_ROOT_Q "%q" "$PROJECT_ROOT"
printf -v QWEN_DIR_Q "%q" "$QWEN_DIR"
printf -v ARTIFACT_VERSION_Q "%q" "$ARTIFACT_VERSION"
printf -v PYTHON_BIN_Q "%q" "$PYTHON_BIN"
printf -v TRAIN_SCRIPT_Q "%q" "$TRAIN_SCRIPT"
printf -v LOG_FILE_Q "%q" "$LOG_FILE"
printf -v VENV_DIR_Q "%q" "$VENV_DIR"
printf -v VENV_ACTIVATE_Q "%q" "$VENV_ACTIVATE"
printf -v LEGACY_VENV_DIR_Q "%q" "$LEGACY_VENV_DIR"
printf -v LEGACY_VENV_ACTIVATE_Q "%q" "$LEGACY_VENV_ACTIVATE"

RESUME_EXPORT=""
if [ -n "${DIT_RESUME_FROM_CHECKPOINT:-}" ]; then
  printf -v RESUME_FROM_CHECKPOINT_Q "%q" "$DIT_RESUME_FROM_CHECKPOINT"
  RESUME_EXPORT="export DIT_RESUME_FROM_CHECKPOINT=$RESUME_FROM_CHECKPOINT_Q;"
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session 已存在：$SESSION_NAME"
  echo "查看日志：tail -f '$LOG_FILE'"
  echo "进入会话：tmux attach -t '$SESSION_NAME'"
  exit 1
fi

TMUX_COMMAND="export RIDING_HAILING_PROJECT_ROOT=$PROJECT_ROOT_Q; export QWEN3_4B_BASE_PATH=$QWEN_DIR_Q; export DIT_ARTIFACT_VERSION=$ARTIFACT_VERSION_Q; $RESUME_EXPORT cd $PROJECT_ROOT_Q; if [ -d $VENV_DIR_Q ]; then source $VENV_ACTIVATE_Q; elif [ -d $LEGACY_VENV_DIR_Q ]; then source $LEGACY_VENV_ACTIVATE_Q; fi; $PYTHON_BIN_Q $TRAIN_SCRIPT_Q 2>&1 | tee -a $LOG_FILE_Q; train_status=\${PIPESTATUS[0]}; echo \"Python训练进程已退出，退出码：\$train_status\" | tee -a $LOG_FILE_Q; echo \"正在执行 /usr/bin/shutdown -h now 关闭AutoDL实例。\" | tee -a $LOG_FILE_Q; sync; /usr/bin/shutdown -h now; exit \$train_status"

tmux new-session -d -s "$SESSION_NAME" "bash -lc '$TMUX_COMMAND'"

echo "Dit 正式训练已启动。"
echo "数据/模型版本：$ARTIFACT_VERSION"
echo "tmux 会话：$SESSION_NAME"
echo "训练日志：$LOG_FILE"
echo "进入会话：tmux attach -t '$SESSION_NAME'"
echo "后台看日志：tail -f '$LOG_FILE'"
echo "Python训练进程退出后将自动执行：/usr/bin/shutdown -h now"
