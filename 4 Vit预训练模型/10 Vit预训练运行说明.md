# Vit 101维分位数预训练与编码导出

Vit预训练使用历史可见信息，不依赖Dit和tauijt预测。所有结构化特征先组成文本Prompt，再Tokenize并输入Qwen3-4B+Vit专用LoRA+101维分位数头。

默认首次实验抽样25,000个训练状态、5,000个验证状态，训练1轮；只在Qwen最后4层的q/v投影上训练LoRA。这用于给SAC内部的Vit TD更新提供合理初值。后续正式扩大样本时，可通过`VIT_TRAIN_MAX_SAMPLES`和`VIT_VALID_MAX_SAMPLES`覆盖默认值。

## 1. 本地CPU数据准备

在项目根目录依次运行：

```bash
python "5 第一阶段SAC训练/1 训练前的准备/2 构建历史基础供给代理.py"
python "4 Vit预训练模型/1 数据准备/2 构建101维Vit轨迹分布标签.py" --splits train valid
python "4 Vit预训练模型/1 数据准备/3 构建历史需求供给与邻近区域特征.py" --splits train valid
python "4 Vit预训练模型/1 数据准备/4 构建历史OD目的地与关联区域特征.py" --splits train valid
python "4 Vit预训练模型/1 数据准备/5 构建Vit文本Prompt与预训练样本.py" --splits train valid
python "4 Vit预训练模型/1 数据准备/6 TokenizeVit预训练样本.py" --splits train valid
```

这些步骤不训练4B模型，可以在Mac本地运行。通常约需30–90分钟，主要时间用于全量Prompt文本化和Tokenize。

## 2. 服务器GPU预训练

将完整的`4 Vit预训练模型`、Qwen基座模型与相关历史数据上传到项目对应路径。服务器需要`torch`、`transformers`、`peft`、`datasets`、`pandas`和`pyarrow`。

如果SAC日级episode尚未生成，先只训练和验证Vit：

```bash
RIDING_HAILING_PROJECT_ROOT="/root/autodl-tmp/Riding hailing_Project" \
VIT_RUN_MODE=train \
AUTO_SHUTDOWN=1 \
bash "4 Vit预训练模型/2 训练与评估/5 在服务器预训练并导出.sh"
```

启动脚本会自动创建tmux后台会话并输出日志路径。实时查看：

```bash
tail -f '启动时显示的完整日志路径'
```

按`Ctrl+C`只会退出日志查看，不会中止训练。Mac断网、VPN断开、SSH退出或Mac关机都不会影响tmux中的任务。`AUTO_SHUTDOWN=1`会在流程结束或异常退出后关闭服务器。

RTX 4090上，当前25,000样本、1轮LoRA预训练预计约8–24小时。实际时间受Prompt动态padding、PyTorch版本和显存速度影响，以开始后前200个batch的平均速度为准。

## 3. 导出SAC所需状态编码

编码导出需要SAC日级episode，因为代码只编码SAC动态轨迹可能访问的出发区域、等待状态和目的地状态，避免对801个区域×96个时段全量运行Qwen。

当Dit、tauijt预测完成并已运行“5 整理日级训练输入.py”后，执行：

```bash
RIDING_HAILING_PROJECT_ROOT="/root/autodl-tmp/Riding hailing_Project" \
VIT_RUN_MODE=export \
VIT_EXPORT_SPLITS=train \
AUTO_SHUTDOWN=1 \
bash "4 Vit预训练模型/2 训练与评估/5 在服务器预训练并导出.sh"
```

主实验暂时只有训练集episode时，使用`VIT_EXPORT_SPLITS=train`；验证集episode准备好后，再单独使用`VIT_EXPORT_SPLITS=valid`导出。如果episode在预训练前已经全部准备好，可以将`VIT_RUN_MODE`设为`all`，一次连续完成预训练、验证和导出。稀疏导出预计约4–16小时，最终以episode实际涉及的区域—时段数量和日志速度为准。
