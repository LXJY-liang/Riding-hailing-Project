#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${RIDING_HAILING_PROJECT_ROOT:-/root/autodl-tmp/Riding hailing_Project}"
VIT_ROOT="$PROJECT_ROOT/4 Vit预训练模型"
LOG_DIR="$VIT_ROOT/output/log"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-0}"
RUN_MODE="${VIT_RUN_MODE:-all}"
EXPORT_SPLITS="${VIT_EXPORT_SPLITS:-train valid}"
SESSION="${VIT_TMUX_SESSION:-vit_pretrain_${RUN_MODE}}"
SCRIPT_PATH="$VIT_ROOT/2 训练与评估/5 在服务器预训练并导出.sh"
LOG_FILE="${VIT_LOG_FILE:-$LOG_DIR/server_pretrain_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "$LOG_DIR"

if [[ "$RUN_MODE" != "all" && "$RUN_MODE" != "train" && "$RUN_MODE" != "export" ]]; then
  echo "VIT_RUN_MODE只能是all、train或export。"
  exit 1
fi

# 父进程只负责创建tmux；SSH断开后子进程仍继续。
if [[ "${VIT_TMUX_CHILD:-0}" != "1" ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "服务器未安装tmux。"
    exit 1
  fi
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux会话已存在：$SESSION"
    exit 1
  fi
  COMMAND="env RIDING_HAILING_PROJECT_ROOT='$PROJECT_ROOT' AUTO_SHUTDOWN='$AUTO_SHUTDOWN' VIT_RUN_MODE='$RUN_MODE' VIT_EXPORT_SPLITS='$EXPORT_SPLITS' VIT_TMUX_CHILD=1 VIT_TMUX_SESSION='$SESSION' VIT_LOG_FILE='$LOG_FILE' bash '$SCRIPT_PATH'"
  tmux new-session -d -s "$SESSION" "$COMMAND"
  echo "Vit预训练会话：$SESSION"
  echo "执行模式：$RUN_MODE"
  echo "训练日志：$LOG_FILE"
  echo "实时进度：tail -f '$LOG_FILE'"
  exit 0
fi

if [[ "$AUTO_SHUTDOWN" == "1" ]]; then
  trap 'sync; shutdown -h now' EXIT
fi

cd "$PROJECT_ROOT"
if [[ "$RUN_MODE" == "all" || "$RUN_MODE" == "train" ]]; then
  python "4 Vit预训练模型/2 训练与评估/2 预训练Vit分位数模型.py" 2>&1 | tee "$LOG_FILE"
  python "4 Vit预训练模型/2 训练与评估/3 评估Vit分位数模型.py" --split valid 2>&1 | tee -a "$LOG_FILE"
fi
if [[ "$RUN_MODE" == "all" || "$RUN_MODE" == "export" ]]; then
  python "4 Vit预训练模型/2 训练与评估/4 导出Vit训练所需状态编码.py" --splits $EXPORT_SPLITS 2>&1 | tee -a "$LOG_FILE"
fi
