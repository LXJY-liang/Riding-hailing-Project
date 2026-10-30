#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${RIDING_HAILING_PROJECT_ROOT:-/root/autodl-tmp/Riding hailing_Project}"
VIT_ROOT="$PROJECT_ROOT/4 Vit预训练模型"
LOG_DIR="$VIT_ROOT/2 训练与评估/output/log"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-0}"
RUN_MODE="${VIT_RUN_MODE:-all}"
SESSION="${VIT_TMUX_SESSION:-vit_pretrain_${RUN_MODE}}"
SCRIPT_PATH="$VIT_ROOT/2 训练与评估/6 在服务器预训练并导出.sh"
LOG_FILE="${VIT_LOG_FILE:-$LOG_DIR/server_pretrain_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "$LOG_DIR"

# AutoDL的非登录Shell中可能没有裸命令python。允许显式覆盖，
# 并优先使用AutoDL默认Miniconda解释器。
if [[ -n "${VIT_PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$VIT_PYTHON_BIN"
elif [[ -x /root/miniconda3/bin/python ]]; then
  PYTHON_BIN=/root/miniconda3/bin/python
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "未找到可用的Python解释器。"
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python解释器不可执行：$PYTHON_BIN"
  exit 1
fi

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
  COMMAND="env RIDING_HAILING_PROJECT_ROOT='$PROJECT_ROOT' AUTO_SHUTDOWN='$AUTO_SHUTDOWN' VIT_RUN_MODE='$RUN_MODE' VIT_TMUX_CHILD=1 VIT_TMUX_SESSION='$SESSION' VIT_LOG_FILE='$LOG_FILE' bash '$SCRIPT_PATH'"
  tmux new-session -d -s "$SESSION" "$COMMAND"
  echo "Vit预训练会话：$SESSION"
  echo "执行模式：$RUN_MODE"
  echo "训练日志：$LOG_FILE"
  echo "实时进度：tail -f '$LOG_FILE'"
  exit 0
fi

# tmux子进程退出时统一关机。开启AUTO_SHUTDOWN后，无论完整成功还是任一步骤
# 失败，都会关机，避免无人值守期间服务器持续计费。父进程不安装此trap，
# 因此创建完tmux会话后不会误触发关机。
shutdown_on_exit() {
  exit_status=$?
  trap - EXIT
  if [[ "$AUTO_SHUTDOWN" == "1" ]]; then
    if [[ "$exit_status" -eq 0 ]]; then
      message="Vit任务已全部成功完成，服务器即将自动关机。"
    else
      message="Vit任务执行失败（退出状态：$exit_status），服务器即将自动关机以停止计费。"
    fi
    echo "$message" | tee -a "$LOG_FILE"
    sync
    shutdown -h now || poweroff
  fi
  exit "$exit_status"
}
trap shutdown_on_exit EXIT

export OMP_NUM_THREADS="${VIT_OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${VIT_MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${VIT_OPENBLAS_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_ROOT"
echo "Python解释器：$PYTHON_BIN" | tee -a "$LOG_FILE"
if [[ "$RUN_MODE" == "all" || "$RUN_MODE" == "train" ]]; then
  "$PYTHON_BIN" "4 Vit预训练模型/2 训练与评估/2 预训练Vit分位数模型.py" 2>&1 | tee -a "$LOG_FILE"
  "$PYTHON_BIN" "4 Vit预训练模型/2 训练与评估/3 评估Vit分位数模型.py" 2>&1 | tee -a "$LOG_FILE"
fi
if [[ "$RUN_MODE" == "all" || "$RUN_MODE" == "export" ]]; then
  "$PYTHON_BIN" "4 Vit预训练模型/2 训练与评估/4 导出Vit预测期状态编码.py" --overwrite 2>&1 | tee -a "$LOG_FILE"
  "$PYTHON_BIN" "4 Vit预训练模型/2 训练与评估/5 预测并导出离线SAC使用的Vit分布.py" --overwrite 2>&1 | tee -a "$LOG_FILE"
fi
