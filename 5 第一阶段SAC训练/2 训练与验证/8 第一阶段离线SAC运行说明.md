# 第一阶段外生历史状态离线SAC

## 模型口径

- 每天一个episode，每15分钟一个时间步。
- Actor动作只影响当前时段价格、补贴、接受率、成交量与收益。
- 下一决策状态直接读取历史下一状态，动作不改变需求、供给或Vit。
- 只连接同一区域严格下一15分钟的历史OD组；若下一时段无该区域OD决策，延续价值为0。
- 供给是目标日前历史成交出发量的滚动高分位容量代理。
- Dit成交量分布按历史综合完成率恢复为潜在需求分布。
- Qwen、LoRA、512维投影和101维Vit分位数头全部冻结。
- 训练开始前由历史episode确定固定的区域—时段转移数据集。
- 数据集没有容量和FIFO，不保存Actor生成的动作、奖励或新轨迹。
- 固定数据集包含全部历史转移；每次梯度更新只从中随机抽取32个组。
- Actor动作及即时奖励只在更新时计算，绝不写回固定数据集。

## 数据准备顺序

```bash
python "5 第一阶段SAC训练/1 训练前的准备/3 构造滚动固定供给容量.py"

python "5 第一阶段SAC训练/1 训练前的准备/4 将Dit转换为潜在需求.py" \
  --splits train valid

python "5 第一阶段SAC训练/1 训练前的准备/5 整理离线日级Episode.py" \
  --splits train valid

# 只编码新episode必需且尚未编码的状态；已有结果会被保留。
python "4 Vit预训练模型/2 训练与评估/4 导出Vit训练所需状态编码.py" \
  --splits train valid

python "5 第一阶段SAC训练/1 训练前的准备/6 导出冻结Vit价值分布.py" \
  --splits train valid --overwrite

python "5 第一阶段SAC训练/1 训练前的准备/7 检查第一阶段离线输入.py" \
  --splits train valid
```

测试集Dit、tau和512维编码齐备后，把`test`加入相同命令。

其中滚动供给和Dit转换可在本地运行。Qwen的512维编码补全应在GPU服务器运行；
最终101维Vit缓存只运行冻结的小型分位数头，本地或服务器均可快速完成。

截至当前本地数据：训练集512维编码只差60个新episode必需状态；验证集尚无512维编码；
测试集尚无Dit与tauijt预测。检查脚本会对这些缺失直接报错，禁止用零向量代替。

## 服务器训练

```bash
RIDING_HAILING_PROJECT_ROOT="/root/autodl-tmp/Riding hailing_Project" \
SAC_RUN_NAME="offline_sac_main_v1" \
SAC_EPOCHS=20 \
AUTO_SHUTDOWN=1 \
SAC_OMP_NUM_THREADS=1 \
bash "5 第一阶段SAC训练/2 训练与验证/7 在服务器启动离线训练.sh"
```

## 验证集选择与测试

```bash
python "5 第一阶段SAC训练/2 训练与验证/6 在验证集选择最佳模型.py" \
  --run-name offline_sac_main_v1

python "5 第一阶段SAC训练/2 训练与验证/5 评估第一阶段离线SAC.py" \
  --run-name offline_sac_main_v1 --split test
```
