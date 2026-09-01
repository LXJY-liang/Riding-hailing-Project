# 第一版离线 SAC 训练

本目录实现“历史状态 + 反事实动作 + 冻结历史下一状态”的第一版离线 SAC。

## 目录与运行顺序

1. `1 数据准备/1 离线SAC统一配置.py`
   - 集中设置上游数据路径、训练/验证/测试时间、两个折扣因子和训练超参数。
2. `1 数据准备/2 构建有效OD基础数据.py`
   - 读取已有 OD-15min 成交订单聚合表。
   - 只保留 `D_ijt_observed > 0` 的 OD；无订单 OD 不输出价格或补贴。
   - 构造历史价格、乘客最高支付意愿代理、行程时间和成本分位数。
3. `1 数据准备/3 检查有效OD基础数据.py`
   - 检查键唯一性、正成交量、正参考价格和价格代理分位数单调性。
4. `2 Sit重构/1 重构历史供给Sit.py`
   - 根据历史出发量和按实际结束时段统计的到达量，重构冻结的历史供给代理。
   - 强制 `S_it_hist >= D_origin_it`，避免“已有成交但供给为 0”。
5. `2 Sit重构/2 检查历史供给Sit.py`
   - 检查供给覆盖率及 `S_it_hist >= D_origin_it` 是否始终成立。
6. `3 Vit准备/1 构建司机未来价值Vit分布.py`
   - 在历史环境中模拟代表性司机未来 4 小时的随机接单、等待和跨区移动轨迹。
   - 先得到使用未来历史事实的 `V_oracle`，再只用过去日期的同区域、同时段结果构造
     Actor 可见的 `V_proxy`，避免直接把未来事实放进状态。
7. `3 Vit准备/2 检查司机未来价值Vit分布.py`
   - 检查 Vit 分位数单调性、覆盖率和过去信息代理是否可用。
8. `4 离线SAC模型/1 构建离线SAC训练状态.py`
   - 拼接 OD、供给和司机价值代理，生成训练、验证、测试状态。
9. `4 离线SAC模型/2 计算反事实收益.py`
   - 根据候选价格和补贴重算接受率、成交量、约束及平台收益。
10. `4 离线SAC模型/3 定义SAC网络.py`
    - 定义 Actor、双 Critic、目标网络更新和回放池。
11. `4 离线SAC模型/4 测试核心公式.py`
    - 用合成样本检查核心公式，不读取全量训练数据。
12. `4 离线SAC模型/5 训练离线SAC模型.py`
   - Actor 在历史状态上提出新价格和补贴；反事实环境重算接受率、成交量和平台收益。
   - 下一状态始终是历史下一时段；当前动作不更新下一时段的 `S`、`V` 或其他状态。
13. `4 离线SAC模型/6 评估离线SAC策略.py`
   - 默认只在验证集评估 SAC，输出逐 OD 新价格、历史价格、价格差、涨跌幅和补贴。
   - 可显式选择简单基线；二维网格直接优化只在 `--mode all` 时运行。
14. `4 离线SAC模型/7 在服务器启动训练.sh`
    - 检查 CUDA 与必需文件，并使用 tmux 在服务器后台启动正式训练。

## 推荐命令

在项目根目录使用项目虚拟环境运行：

```bash
.venv/bin/python "6 离线SAC训练/1 数据准备/2 构建有效OD基础数据.py"
.venv/bin/python "6 离线SAC训练/1 数据准备/3 检查有效OD基础数据.py"
.venv/bin/python "6 离线SAC训练/2 Sit重构/1 重构历史供给Sit.py"
.venv/bin/python "6 离线SAC训练/2 Sit重构/2 检查历史供给Sit.py"
.venv/bin/python "6 离线SAC训练/3 Vit准备/1 构建司机未来价值Vit分布.py"
.venv/bin/python "6 离线SAC训练/3 Vit准备/2 检查司机未来价值Vit分布.py"
.venv/bin/python "6 离线SAC训练/4 离线SAC模型/1 构建离线SAC训练状态.py"
.venv/bin/python "6 离线SAC训练/4 离线SAC模型/4 测试核心公式.py"
.venv/bin/python "6 离线SAC训练/4 离线SAC模型/5 训练离线SAC模型.py"
.venv/bin/python "6 离线SAC训练/4 离线SAC模型/6 评估离线SAC策略.py" \
  --split valid --mode sac_only \
  --checkpoint "6 离线SAC训练/4 离线SAC模型/output/model/gamma099/best_offline_sac.pt" \
  --output-tag gamma099
```

平台折扣因子消融需要独立训练，不能只在评估时改一个数字：

```bash
.venv/bin/python "6 离线SAC训练/4 离线SAC模型/5 训练离线SAC模型.py" --sac-gamma 0 --run-name gamma000
.venv/bin/python "6 离线SAC训练/4 离线SAC模型/6 评估离线SAC策略.py" \
  --split valid --mode sac_only \
  --checkpoint "6 离线SAC训练/4 离线SAC模型/output/model/gamma000/best_offline_sac.pt" \
  --output-tag gamma000
```

正式运行前先检查 `1 数据准备/1 离线SAC统一配置.py`。其中平台抽成率、补贴上限比例、
价格范围、两个折扣因子和训练步数都集中配置；代码中的默认值只是当前第一版实验设定。

## 上游“1 数据清洗”文件夹何时使用

`1 数据清洗` 中的 `2_2 OD-15min聚合_属于芝加哥.parquet` 和
`3_3 区域-15min聚合（补全空值）结果.parquet` 已经分别用于生成基础 OD 和 Sit；Vit
及最终训练状态又由这些派生结果生成。当前流程没有在正式训练中读取订单级原始表。
只要直接上传并使用本目录 `output` 下已经生成的派生文件，服务器就不必上传
`1 数据清洗`。只有需要从头重建任一派生数据时，才需要上传相应的上游数据文件。
