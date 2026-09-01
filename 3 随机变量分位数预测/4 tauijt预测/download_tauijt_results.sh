#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH_HOST="${SSH_HOST:-root@connect.nmb1.seetacloud.com}"
SSH_PORT="${SSH_PORT:-36453}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/root/autodl-tmp/Riding hailing_Project}"
REMOTE_TAUIJT="$REMOTE_PROJECT/3 随机变量分位数预测/4 tauijt预测"
SHUTDOWN_AFTER_DOWNLOAD="${SHUTDOWN_AFTER_DOWNLOAD:-0}"
SSH_CONTROL_PATH="/tmp/tauijt-download-%C"
SSH_OPTIONS=(
  -p "$SSH_PORT"
  -o ControlMaster=auto
  -o ControlPersist=600
  -o "ControlPath=$SSH_CONTROL_PATH"
)

echo "正在核验服务器训练结果……"
printf -v REMOTE_TAUIJT_Q "%q" "$REMOTE_TAUIJT"
ssh "${SSH_OPTIONS[@]}" "$SSH_HOST" \
  "bash -s -- $REMOTE_TAUIJT_Q" <<'REMOTE'
set -euo pipefail
tau_dir="$1"
required=(
  "$tau_dir/output/model/best_lora"
  "$tau_dir/output/model/best_quantile_head.pt"
  "$tau_dir/output/log/best_model_info.json"
  "$tau_dir/output/log/training_history.csv"
)
for path in "${required[@]}"; do
  if [ ! -e "$path" ]; then
    echo "缺少最终结果：$path" >&2
    exit 2
  fi
done
if ! grep -q "epoch_end_validation" "$tau_dir/output/log/training_history.csv"; then
  echo "training_history.csv中没有epoch_end_validation，暂不能确认完整训练结束。" >&2
  exit 3
fi
echo "已确认：训练和epoch末完整验证均已结束。"
echo "--- best_model_info.json ---"
cat "$tau_dir/output/log/best_model_info.json"
echo
echo "--- training_history.csv ---"
cat "$tau_dir/output/log/training_history.csv"
if [ -f "$tau_dir/output/log/tauijt_valid_metrics.json" ]; then
  echo
  echo "已检测到验证集独立评估结果。"
  cat "$tau_dir/output/log/tauijt_valid_metrics.json"
fi
REMOTE

mkdir -p "$SCRIPT_DIR/output/model" "$SCRIPT_DIR/output/log" "$SCRIPT_DIR/output/prediction"
escaped_remote_tauijt="${REMOTE_TAUIJT// /\\ }"
ssh_command="ssh -p $SSH_PORT -o ControlMaster=auto -o ControlPersist=600 -o ControlPath=$SSH_CONTROL_PATH"

echo "正在下载模型、Tokenizer、Scaler和训练日志……"
rsync -avh --partial --progress \
  -e "$ssh_command" \
  "$SSH_HOST:$escaped_remote_tauijt/output/model/" \
  "$SCRIPT_DIR/output/model/"
rsync -avh --partial --progress \
  -e "$ssh_command" \
  "$SSH_HOST:$escaped_remote_tauijt/output/log/" \
  "$SCRIPT_DIR/output/log/"

if ssh "${SSH_OPTIONS[@]}" "$SSH_HOST" \
  "test -d '$REMOTE_TAUIJT/output/prediction'"; then
  echo "正在下载已有的验证/测试预测结果……"
  rsync -avh --partial --progress \
    -e "$ssh_command" \
    "$SSH_HOST:$escaped_remote_tauijt/output/prediction/" \
    "$SCRIPT_DIR/output/prediction/"
fi

echo "下载完成：$SCRIPT_DIR/output"

if [ "$SHUTDOWN_AFTER_DOWNLOAD" = "1" ]; then
  echo "下载已成功，正在关闭服务器以停止计费……"
  ssh "${SSH_OPTIONS[@]}" "$SSH_HOST" "/usr/bin/shutdown -h now" || true
fi
