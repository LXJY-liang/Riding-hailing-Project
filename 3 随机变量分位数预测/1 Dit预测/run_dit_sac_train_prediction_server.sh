#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${RIDING_HAILING_PROJECT_ROOT:-/root/autodl-tmp/Riding hailing_Project}"
DIT_DIR="$PROJECT_ROOT/3 随机变量分位数预测/1 Dit预测"
QWEN_DIR="${QWEN3_4B_BASE_PATH:-$PROJECT_ROOT/0 基座模型_Qwen3-4B-Base}"
ARTIFACT_VERSION="${DIT_ARTIFACT_VERSION:-final_full}"
ARTIFACT_DIR="$DIT_DIR/output/$ARTIFACT_VERSION"
SESSION_NAME="${DIT_SAC_TRAIN_TMUX_SESSION:-dit_sac_train_prediction}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
LOG_DIR="$ARTIFACT_DIR/log"
LOG_FILE="$LOG_DIR/dit_sac_train_prediction_$(date +%Y%m%d_%H%M%S).log"
PREDICTION_FILE="$ARTIFACT_DIR/prediction/dit_train_predictions.parquet"

shutdown_instance() {
  echo "正在执行 /usr/bin/shutdown -h now 关闭AutoDL实例。"
  sync
  /usr/bin/shutdown -h now
}

mkdir -p "$LOG_DIR" "$ARTIFACT_DIR/prediction"

required_paths=(
  "$DIT_DIR/9 预测SAC分组公共函数.py"
  "$DIT_DIR/10 预测SAC训练集.py"
  "$ARTIFACT_DIR/data/dit_train.parquet"
  "$ARTIFACT_DIR/tokenized/dit_train_tokenized"
  "$ARTIFACT_DIR/model/best_lora"
  "$ARTIFACT_DIR/model/best_quantile_head.pt"
  "$ARTIFACT_DIR/model/label_scaler.json"
  "$ARTIFACT_DIR/model/tokenizer"
  "$QWEN_DIR/config.json"
  "$QWEN_DIR/model.safetensors.index.json"
  "$PYTHON_BIN"
)

for path in "${required_paths[@]}"; do
  if [ ! -e "$path" ]; then
    echo "缺少必要文件或目录：$path" | tee -a "$LOG_FILE"
    shutdown_instance
    exit 1
  fi
done

if [ -e "$PREDICTION_FILE" ]; then
  echo "目标文件已经存在，为避免覆盖而停止：$PREDICTION_FILE" | tee -a "$LOG_FILE"
  shutdown_instance
  exit 1
fi

if tmux has-session -t "=$SESSION_NAME" 2>/dev/null; then
  echo "tmux session 已存在：$SESSION_NAME"
  exit 1
fi

printf -v PROJECT_ROOT_Q "%q" "$PROJECT_ROOT"
printf -v QWEN_DIR_Q "%q" "$QWEN_DIR"
printf -v ARTIFACT_VERSION_Q "%q" "$ARTIFACT_VERSION"
printf -v PYTHON_BIN_Q "%q" "$PYTHON_BIN"
printf -v SCRIPT_Q "%q" "$DIT_DIR/10 预测SAC训练集.py"
printf -v LOG_FILE_Q "%q" "$LOG_FILE"

TMUX_COMMAND="set +e; cd $PROJECT_ROOT_Q; export RIDING_HAILING_PROJECT_ROOT=$PROJECT_ROOT_Q; export QWEN3_4B_BASE_PATH=$QWEN_DIR_Q; export DIT_ARTIFACT_VERSION=$ARTIFACT_VERSION_Q; export DIT_SAC_PRED_BATCH_SIZE=16; export PYTHONUNBUFFERED=1; $PYTHON_BIN_Q $SCRIPT_Q 2>&1 | tee -a $LOG_FILE_Q; prediction_status=\${PIPESTATUS[0]}; echo \"Dit SAC训练集预测进程已退出，退出码：\$prediction_status\" | tee -a $LOG_FILE_Q; echo \"正在执行 /usr/bin/shutdown -h now 关闭AutoDL实例。\" | tee -a $LOG_FILE_Q; sync; /usr/bin/shutdown -h now; exit \$prediction_status"

tmux new-session -d -s "$SESSION_NAME" "bash -lc '$TMUX_COMMAND'"

echo "Dit SAC训练集预测已启动。"
echo "模型版本：$ARTIFACT_VERSION"
echo "预测范围：2024-06-11 00:00:00 至 2024-06-24 23:45:00"
echo "tmux 会话：$SESSION_NAME"
echo "日志：$LOG_FILE"
echo "结果：$PREDICTION_FILE"
echo "查看进度：tail -f '$LOG_FILE'"
echo "Python进程无论成功或失败，退出后都会执行：/usr/bin/shutdown -h now"
