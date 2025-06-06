# 1. 数据集简介

UCI Adult Income 数据集（又称 Census Income 数据集）是机器学习领域中广泛使用的基准数据集，常用于分类任务的模型训练与评估。该数据集基于 1994 年美国人口普查数据，目标是根据个体的人口统计特征预测其年收入是否超过 50,000 美元。数据集共包含约 48,842 条记录，每条记录对应一个成年人，包含 14 个特征变量，如年龄、性别、教育水平、婚姻状态、职业、工作时长、国籍等，涵盖数值型与分类型变量。目标变量为二分类标签 `income`，取值为 `<=50K` 或 `>50K`。该数据集在特征工程、类别编码、样本不平衡处理等方面具有较高的应用价值，是模型评估与比较的重要实验基础。

# 2. 基本思路

本文旨在比较传统统计方法、机器学习方法与 PPI 方法在伯努利参数 \( p \) 的区间估计任务中的推断性能差异。我们假设数据集中 `income` 字段为一组独立同分布的伯努利样本 \( \text{Bernoulli}(p) \)，目标是构建置信水平为 90% 的参数区间估计，并以置信区间的长度作为性能评价指标，以反映不同方法在推断精度上的差异。

# 3. 数据处理

为了构造半监督学习场景，并对比不同方法对未标记数据的利用效果，我们将原始数据集随机划分为两个子集：

- **D1（有标签子集）**：随机抽取 10,000 条记录并保留其真实标签；
- **D2（无标签子集）**：将剩余的约 38,000 条记录的标签移除，仅保留特征，用于模型推理。

# 4. 实现流程

1. **传统统计方法（置信区间记为 \( C_1 \)）**  
   在 D1 上直接统计 `income` 的样本比例，基于传统统计学方法构建伯努利参数 \( p \) 水平 90% 置信区间。

2. **机器学习增强方法（置信区间记为 \( C_2 \)）**  
   - 使用 D1 训练 XGBoost 二分类模型；
   - 利用模型对 D2 进行推理，生成其 `income` 作为伪标签（pseudo labels）；
   - 将 D2 的伪标签与 D1 的真实标签合并，构成扩展标签集；
   - 在合并数据上重新估计 \( p \)，构造新的 90% 置信区间 \( C_2 \)。

3. **PPI 方法（记为 \( 置信区间间_3 \)）**  
   利用训练模型对 D2 推理所得为标签，结合 D1 的金标数据，采用 PPI（Posterior Predictive Inference）方法构建置信区间 \( C_3 \)，更充分地融合模型预测的不确定性信息。
