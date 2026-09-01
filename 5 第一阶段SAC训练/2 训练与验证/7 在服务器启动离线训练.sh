#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${SAC_OMP_NUM_THREADS:-1}"
PROJECT_ROOT="${RIDING_HAILING_PROJECT_ROOT:-/root/autodl-tmp/Riding hailing_Project}"
RUN_NAME="${SAC_RUN_NAME:-offline_sac_main}"
EPOCHS="${SAC_EPOCHS:-20}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-0}"
TRAIN_DIR="$PROJECT_ROOT/5 第一阶段SAC训练"
PREP_DIR="$TRAIN_DIR/1 训练前的准备"
RUN_DIR="$TRAIN_DIR/2 训练与验证"
LOG_DIR="$TRAIN_DIR/output/log"
SESSION="stage1_${RUN_NAME}"
LOG_FILE="$LOG_DIR/server_${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"
python "$PREP_DIR/7 检查第一阶段离线输入.py" --splits train

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux会话已存在：$SESSION"
    exit 1
fi

COMMAND="set -o pipefail; export OMP_NUM_THREADS='$OMP_NUM_THREADS'; python '$RUN_DIR/4 训练第一阶段离线SAC.py' --run-name '$RUN_NAME' --epochs '$EPOCHS' 2>&1 | tee '$LOG_FILE'; code=\${PIPESTATUS[0]}"
if [[ "$AUTO_SHUTDOWN" == "1" ]]; then
    # 无论训练成功或报错均关机，避免无人值守时继续计费；退出码保留在关机前的日志中。
    COMMAND="$COMMAND; echo TRAIN_EXIT_CODE=\$code | tee -a '$LOG_FILE'; sync; shutdown -h now; exit \$code"
else
    COMMAND="$COMMAND; exit \$code"
fi

tmux new-session -d -s "$SESSION" env OMP_NUM_THREADS="$OMP_NUM_THREADS" bash -c "$COMMAND"
echo "训练会话：$SESSION"
echo "训练epoch：$EPOCHS"
echo "训练日志：$LOG_FILE"
echo "查看日志：tail -n 30 -F '$LOG_FILE'"
