#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
QWEN_DIR="${QWEN3_4B_BASE_PATH:-$PROJECT_ROOT/0 基座模型_Qwen3-4B-Base}"
SESSION_NAME="${TAUIJT_SAC_PRED_SESSION:-tauijt_sac_prediction}"
SHUTDOWN_SESSION="${SESSION_NAME}_auto_shutdown"
BATCH_SIZE="${TAUIJT_SAC_PRED_BATCH_SIZE:-16}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-1}"
LOG_DIR="$SCRIPT_DIR/output/log"
LOG_FILE="$LOG_DIR/tauijt_sac_prediction_$(date +%Y%m%d_%H%M%S).log"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
TRAIN_SCRIPT="$SCRIPT_DIR/10 预测SAC训练集.py"

for path in \
  "$SCRIPT_DIR/10 预测SAC训练集.py" \
  "$SCRIPT_DIR/output/data/tauijt_train.parquet" \
  "$SCRIPT_DIR/output/tokenized/tauijt_train_tokenized" \
  "$SCRIPT_DIR/output/model/best_lora" \
  "$SCRIPT_DIR/output/model/best_quantile_head.pt" \
  "$SCRIPT_DIR/output/model/tokenizer" \
  "$SCRIPT_DIR/output/model/label_scaler.json" \
  "$QWEN_DIR/model.safetensors.index.json" \
  "$PYTHON_BIN"; do
  if [ ! -e "$path" ]; then
    echo "缺少预测所需文件：$path" >&2
    exit 1
  fi
done

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux会话已存在：$SESSION_NAME" >&2
  exit 1
fi

mkdir -p "$LOG_DIR" "$SCRIPT_DIR/output/prediction"

printf -v PROJECT_ROOT_Q '%q' "$PROJECT_ROOT"
printf -v QWEN_DIR_Q '%q' "$QWEN_DIR"
printf -v PYTHON_BIN_Q '%q' "$PYTHON_BIN"
printf -v TRAIN_SCRIPT_Q '%q' "$TRAIN_SCRIPT"
printf -v LOG_FILE_Q '%q' "$LOG_FILE"
printf -v BATCH_SIZE_Q '%q' "$BATCH_SIZE"

COMMAND="cd $PROJECT_ROOT_Q; export RIDING_HAILING_PROJECT_ROOT=$PROJECT_ROOT_Q; export QWEN3_4B_BASE_PATH=$QWEN_DIR_Q; export PYTHONUNBUFFERED=1; export TAUIJT_SAC_PRED_BATCH_SIZE=$BATCH_SIZE_Q; $PYTHON_BIN_Q $TRAIN_SCRIPT_Q 2>&1 | tee -a $LOG_FILE_Q; train_status=\${PIPESTATUS[0]}; echo \"tauijt训练期预测结束，退出码：\$train_status\" | tee -a $LOG_FILE_Q; exit \$train_status"

tmux new-session -d -s "$SESSION_NAME" "bash -lc '$COMMAND'"

if [ "$AUTO_SHUTDOWN" = "1" ]; then
  printf -v SESSION_NAME_Q '%q' "$SESSION_NAME"
  SHUTDOWN_COMMAND="while tmux has-session -t $SESSION_NAME_Q 2>/dev/null; do sleep 60; done; sleep 30; sync; /usr/bin/shutdown -h now"
  tmux new-session -d -s "$SHUTDOWN_SESSION" "bash -lc '$SHUTDOWN_COMMAND'"
fi

echo "tauijt训练期预测已启动。"
echo "tmux会话：$SESSION_NAME"
echo "batch_size：$BATCH_SIZE"
echo "日志：$LOG_FILE"
if [ "$AUTO_SHUTDOWN" = "1" ]; then
  echo "预测进程结束后将自动关机。"
fi
