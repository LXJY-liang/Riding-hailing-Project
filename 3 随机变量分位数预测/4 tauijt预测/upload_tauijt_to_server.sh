#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SSH_HOST="${SSH_HOST:-root@connect.nmb1.seetacloud.com}"
SSH_PORT="${SSH_PORT:-33991}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/root/autodl-tmp/Riding hailing_Project}"
UPLOAD_BASE_MODEL="${UPLOAD_BASE_MODEL:-auto}"
SSH_CONTROL_PATH="/tmp/tauijt-upload-%C"
SSH_OPTIONS=(
  -p "$SSH_PORT"
  -o ControlMaster=auto
  -o ControlPersist=600
  -o "ControlPath=$SSH_CONTROL_PATH"
)

cd "$PROJECT_ROOT"

if [ "$UPLOAD_BASE_MODEL" = "auto" ]; then
  if ssh "${SSH_OPTIONS[@]}" "$SSH_HOST" \
    "test -f '$REMOTE_PROJECT/0 基座模型_Qwen3-4B-Base/model.safetensors.index.json' && test -f '$REMOTE_PROJECT/0 基座模型_Qwen3-4B-Base/model-00001-of-00003.safetensors' && test -f '$REMOTE_PROJECT/0 基座模型_Qwen3-4B-Base/model-00002-of-00003.safetensors' && test -f '$REMOTE_PROJECT/0 基座模型_Qwen3-4B-Base/model-00003-of-00003.safetensors'"; then
    UPLOAD_BASE_MODEL=0
    echo "服务器已有基座模型，本次跳过。"
  else
    UPLOAD_BASE_MODEL=1
    echo "服务器未找到基座模型，本次将上传。"
  fi
fi

ssh "${SSH_OPTIONS[@]}" "$SSH_HOST" \
  "mkdir -p '$REMOTE_PROJECT'"

sources=(
  "./3 随机变量分位数预测/4 tauijt预测/1 tauijt配置.py"
  "./3 随机变量分位数预测/4 tauijt预测/3 tauijt模型.py"
  "./3 随机变量分位数预测/4 tauijt预测/5 tauijt训练.py"
  "./3 随机变量分位数预测/4 tauijt预测/run_tauijt_train_server.sh"
  "./3 随机变量分位数预测/4 tauijt预测/output/tokenized/tauijt_train_tokenized/"
  "./3 随机变量分位数预测/4 tauijt预测/output/tokenized/tauijt_valid_tokenized/"
  "./3 随机变量分位数预测/4 tauijt预测/output/model/tokenizer/"
  "./3 随机变量分位数预测/4 tauijt预测/output/model/label_scaler.json"
  "./3 随机变量分位数预测/4 tauijt预测/output/data/tauijt_data_manifest.json"
  "./3 随机变量分位数预测/4 tauijt预测/output/tokenized/tauijt_tokenization_manifest.json"
)

if [ "$UPLOAD_BASE_MODEL" = "1" ]; then
  sources+=("./0 基座模型_Qwen3-4B-Base/")
fi

escaped_remote_project="${REMOTE_PROJECT// /\\ }"
rsync -avhzR --partial --progress \
  -e "ssh -p $SSH_PORT -o ControlMaster=auto -o ControlPersist=600 -o ControlPath=$SSH_CONTROL_PATH" \
  "${sources[@]}" \
  "$SSH_HOST:$escaped_remote_project/"

echo "上传完成。远程项目目录：$REMOTE_PROJECT"
