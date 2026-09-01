#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SSH_HOST="${SSH_HOST:-root@connect.nmb1.seetacloud.com}"
SSH_PORT="${SSH_PORT:-36453}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/root/autodl-tmp/Riding hailing_Project}"
SSH_CONTROL_PATH="/tmp/tauijt-valid-eval-upload-%C"
SSH_OPTIONS=(
  -p "$SSH_PORT"
  -o ControlMaster=auto
  -o ControlPersist=600
  -o "ControlPath=$SSH_CONTROL_PATH"
)

cd "$PROJECT_ROOT"

for path in \
  "./3 随机变量分位数预测/4 tauijt预测/1 tauijt配置.py" \
  "./3 随机变量分位数预测/4 tauijt预测/3 tauijt模型.py" \
  "./3 随机变量分位数预测/4 tauijt预测/6 tauijt评估与测试.py" \
  "./3 随机变量分位数预测/4 tauijt预测/run_tauijt_eval_server.sh" \
  "./3 随机变量分位数预测/4 tauijt预测/output/data/tauijt_valid.parquet"; do
  if [ ! -e "$path" ]; then
    echo "本地缺少评估所需文件：$path" >&2
    exit 1
  fi
done

echo "正在确认服务器保留了训练模型和验证集Tokenized数据……"
ssh "${SSH_OPTIONS[@]}" "$SSH_HOST" \
  "test -d '$REMOTE_PROJECT/3 随机变量分位数预测/4 tauijt预测/output/model/best_lora' && test -f '$REMOTE_PROJECT/3 随机变量分位数预测/4 tauijt预测/output/model/best_quantile_head.pt' && test -d '$REMOTE_PROJECT/3 随机变量分位数预测/4 tauijt预测/output/tokenized/tauijt_valid_tokenized'" || {
    echo "服务器上缺少训练产物或验证集Tokenized数据；请勿继续，先检查远程目录。" >&2
    exit 2
  }

ssh "${SSH_OPTIONS[@]}" "$SSH_HOST" "mkdir -p '$REMOTE_PROJECT'"

sources=(
  "./3 随机变量分位数预测/4 tauijt预测/1 tauijt配置.py"
  "./3 随机变量分位数预测/4 tauijt预测/3 tauijt模型.py"
  "./3 随机变量分位数预测/4 tauijt预测/6 tauijt评估与测试.py"
  "./3 随机变量分位数预测/4 tauijt预测/run_tauijt_eval_server.sh"
  "./3 随机变量分位数预测/4 tauijt预测/output/data/tauijt_valid.parquet"
)

escaped_remote_project="${REMOTE_PROJECT// /\\ }"
rsync -avhR --partial --progress \
  -e "ssh -p $SSH_PORT -o ControlMaster=auto -o ControlPersist=600 -o ControlPath=$SSH_CONTROL_PATH" \
  "${sources[@]}" \
  "$SSH_HOST:$escaped_remote_project/"

echo "验证评估文件上传完成。仅新增/更新了脚本和原始验证集，不上传测试集。"
