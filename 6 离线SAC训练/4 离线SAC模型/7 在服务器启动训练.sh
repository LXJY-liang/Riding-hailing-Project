#!/usr/bin/env bash
set -euo pipefail

# 任务 7：检查服务器 Python/CUDA 和必需数据，并在 tmux 后台启动正式离线 SAC 训练。

# AutoDL 等 Linux 服务器上的默认项目位置。若位置不同，在命令前设置：
# RIDING_HAILING_PROJECT_ROOT="/实际路径/Riding hailing_Project"
PROJECT_ROOT="${RIDING_HAILING_PROJECT_ROOT:-/root/autodl-tmp/Riding hailing_Project}"
SAC_DIR="$PROJECT_ROOT/6 离线SAC训练/4 离线SAC模型"
TRAIN_SCRIPT="$SAC_DIR/5 训练离线SAC模型.py"

# 主实验默认 gamma_SAC=0.99；消融实验可在启动命令前设置 SAC_GAMMA=0。
SAC_GAMMA="${SAC_GAMMA:-0.99}"
SAC_RUN_NAME="${SAC_RUN_NAME:-gamma099}"
SESSION_NAME="${SAC_TMUX_SESSION:-offline_sac_${SAC_RUN_NAME}}"
# 默认在训练进程结束后关闭 AutoDL 实例。无论训练成功还是异常退出都执行关机，避免
# SSH 已断开时实例继续空转计费。若只调试且不希望关机，可设置 SAC_AUTO_SHUTDOWN=0。
SAC_AUTO_SHUTDOWN="${SAC_AUTO_SHUTDOWN:-1}"
LOG_DIR="$SAC_DIR/output/log"
LOG_FILE="$LOG_DIR/server_${SAC_RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

if [ "$SAC_AUTO_SHUTDOWN" != "0" ] && [ "$SAC_AUTO_SHUTDOWN" != "1" ]; then
  echo "SAC_AUTO_SHUTDOWN 只能设置为 0 或 1。"
  exit 1
fi

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
  else
    echo "找不到 Python。请激活虚拟环境或设置 PYTHON_BIN=/path/to/python。"
    exit 1
  fi
fi

required_paths=(
  "$TRAIN_SCRIPT"
  "$SAC_DIR/../1 数据准备/1 离线SAC统一配置.py"
  "$SAC_DIR/../3 Vit准备/output/V_it_distribution.parquet"
  "$SAC_DIR/output/offline_sac_state.parquet"
)
for path in "${required_paths[@]}"; do
  if [ ! -e "$path" ]; then
    echo "缺少正式训练所需文件：$path"
    echo "请先按 README 的顺序执行数据、S、V 和最终状态准备。"
    exit 1
  fi
done

if ! command -v tmux >/dev/null 2>&1; then
  echo "服务器未安装 tmux。请先安装 tmux，或直接运行 5 训练离线SAC模型.py。"
  exit 1
fi

mkdir -p "$LOG_DIR"

# 在真正启动前检查关键包和 CUDA。这里要求有 CUDA；否则长时训练会退回 CPU。
"$PYTHON_BIN" - <<'PY'
import importlib
import torch

for package in ["numpy", "pandas", "pyarrow", "torch"]:
    module = importlib.import_module(package)
    print(f"{package}=={getattr(module, '__version__', 'unknown')}")

print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("未检测到 CUDA GPU，已停止正式训练。")
print(f"gpu={torch.cuda.get_device_name(0)}")
PY

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux 会话已存在：$SESSION_NAME"
  echo "进入会话：tmux attach -t '$SESSION_NAME'"
  exit 1
fi

printf -v PROJECT_ROOT_Q "%q" "$PROJECT_ROOT"
printf -v PYTHON_BIN_Q "%q" "$PYTHON_BIN"
printf -v TRAIN_SCRIPT_Q "%q" "$TRAIN_SCRIPT"
printf -v LOG_FILE_Q "%q" "$LOG_FILE"
printf -v SAC_GAMMA_Q "%q" "$SAC_GAMMA"
printf -v SAC_RUN_NAME_Q "%q" "$SAC_RUN_NAME"
printf -v SAC_AUTO_SHUTDOWN_Q "%q" "$SAC_AUTO_SHUTDOWN"

TRAIN_COMMAND="cd $PROJECT_ROOT_Q; export PYTHONUNBUFFERED=1; $PYTHON_BIN_Q $TRAIN_SCRIPT_Q --sac-gamma $SAC_GAMMA_Q --run-name $SAC_RUN_NAME_Q 2>&1 | tee -a $LOG_FILE_Q; train_status=\${PIPESTATUS[0]}; echo \"SAC 训练退出码：\$train_status\" | tee -a $LOG_FILE_Q; if [ $SAC_AUTO_SHUTDOWN_Q = 1 ]; then echo \"训练进程已结束，60 秒后自动关机。\" | tee -a $LOG_FILE_Q; sleep 60; /usr/bin/shutdown; fi; exit \$train_status"

tmux new-session -d -s "$SESSION_NAME" "bash -lc '$TRAIN_COMMAND'"

echo "离线 SAC 正式训练已在后台启动。"
echo "实验：$SAC_RUN_NAME，gamma_SAC=$SAC_GAMMA"
echo "tmux 会话：$SESSION_NAME"
echo "训练日志：$LOG_FILE"
if [ "$SAC_AUTO_SHUTDOWN" = "1" ]; then
  echo "自动关机：已启用（训练成功或异常退出后均会关机）"
else
  echo "自动关机：未启用"
fi
echo "进入会话：tmux attach -t '$SESSION_NAME'"
echo "只看日志：tail -f '$LOG_FILE'"
