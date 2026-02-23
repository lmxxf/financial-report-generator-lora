# 金融研报指标提取 LoRA

从金融研报段落中提取被深度分析的核心指标，输出结构化 JSON。

## 任务

输入一段研报段落，输出：
```json
{
  "analysis": "段落分析思路...",
  "metrics": [
    {"metric_name": "毛利率", "metric_type": "financial", "score": 0.95, "reason": "被深度分析"}
  ]
}
```

指标数量不固定，根据段落实际内容决定（1-4个）。

## 文件结构

```
├── train_lora.py              # QLoRA 训练 + 推理脚本
├── batch_infer.py             # 批量推理 + 简易评测
├── eval_script.py             # 三层评估脚本（语义匹配）
├── training_data_strategy.md  # 训练数据策略文档
├── sample_data.jsonl          # 50 条 demo 数据
├── test_cases.jsonl           # 20 条测试用例（6 个场景）
├── data/                      # 完整训练集（460 条）
│   ├── type_a.jsonl           # 标准正样本（270条）
│   ├── type_b.jsonl           # 边界负样本（72条）—— 教"什么不该提"
│   ├── type_c.jsonl           # 数量变化样本（72条）—— 打破"永远提2个"
│   └── type_d.jsonl           # 混合类型专项（46条）—— 练 financial/business 分类
└── output/                    # 训练输出（.gitignore）
```

## 环境

DGX Spark (128GB) + docker 容器（nvcr.io/nvidia/pytorch:25.11-py3）。

```bash
# 创建容器（映射项目和模型目录）
docker run -it --gpus all --name lora-train \
  -v /home/lmxxf/work/financial-report-generator-lora:/workspace/lora \
  -v /home/lmxxf/work/models:/workspace/models \
  nvcr.io/nvidia/pytorch:25.11-py3 bash

# 容器内安装依赖
pip install peft bitsandbytes trl
```

| 库 | 作用 |
|---|---|
| peft | LoRA 实现，把全量微调变成只训几百万参数 |
| bitsandbytes | 8bit 量化加载，14B 模型从 28GB 压到 ~15GB |
| trl | HuggingFace 训练器，SFTTrainer + SFTConfig 处理 chat 格式对齐 |

## 为什么需要重新训练

原方案（260 条数据微调 Qwen3-14B）完美匹配仅 29%，详细诊断见 [training_data_strategy.md](training_data_strategy.md)，核心问题：

1. **260 条数据全部恰好 2 个指标** —— prompt 写死"数量严格控制：2个"，模型学会了凑数而不是判断
2. **158 条"不一致"中 45 条是字符串匹配误杀** —— "投资活动现金流" vs "投资活动净现金流"（相似度 0.93）被判错
3. **DeepSeek 的标注本身有问题** —— 它也在硬凑第二个指标

## 训练

基座模型：Qwen3-14B，8bit 量化加载（QLoRA）。

```bash
# 下载模型（中国镜像，在宿主机执行）
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen3-14B --local-dir /home/lmxxf/work/models/Qwen3-14B

# 容器内训练（用 50 条 demo 快速验证流程）
cd /workspace/lora
python train_lora.py train --model /workspace/models/Qwen3-14B --data sample_data.jsonl --epochs 3

# 用完整 460 条正式训练
python train_lora.py train --model /workspace/models/Qwen3-14B --data data/ --epochs 5
```

训练参数（默认值）：

| 参数 | 值 | 说明 |
|---|---|---|
| LoRA r | 16 | 低秩矩阵维度 |
| LoRA alpha | 32 | 缩放系数（alpha/r = 2，标准配比） |
| LoRA dropout | 0.05 | 防过拟合 |
| LoRA target | q/k/v/o/gate/up/down_proj | 全注意力层 + MLP，共 7 个模块 |
| trainable params | 64M / 14.8B (0.43%) | 只训练 LoRA 参数 |
| batch size | 2 × 8 (grad accum) = 16 | 有效批大小 |
| learning rate | 2e-4 | 余弦衰减（cosine） |
| warmup | 5% of total steps | 预热步数 |
| max grad norm | 0.3 | 梯度裁剪 |
| optimizer | paged_adamw_8bit | 8bit 优化器，省显存 |
| 混合精度 | bf16 | Blackwell 原生支持，比 fp16 数值更稳定 |
| gradient checkpointing | 开 | 用计算换显存 |
| max seq len | 2048 | 最大序列长度 |
| 量化 | 8bit (bitsandbytes) | QLoRA，int8_threshold=6.0 |
| 显存估算 | ~20GB | 14B 8bit + LoRA + 梯度 |
| save strategy | 每个 epoch | 保留最近 3 个 checkpoint |

训练好的 LoRA 权重：[lmxxf/financial-report-lora-qwen3-14b](https://huggingface.co/lmxxf/financial-report-lora-qwen3-14b)

## 推理

```bash
python train_lora.py infer \
  --lora output/final \
  --title "盈利能力分析" \
  --text "公司毛利率同比提升2.3个百分点至35.8%，受益于产品结构优化和原材料成本下降。"
```

## 评估

```bash
# 模糊匹配（零依赖）
python eval_script.py --pred predictions.jsonl --gold gold.jsonl

# 语义匹配（需要 sentence-transformers）
python eval_script.py --pred predictions.jsonl --gold gold.jsonl --method embedding

# 逐条查看
python eval_script.py --pred predictions.jsonl --gold gold.jsonl --detail
```

三层评估：
1. **核心命中率**（Recall）—— 标注里的核心指标，模型提到了几个
2. **精确率/召回率/F1** —— 多提了什么、漏了什么
3. **类型准确率** —— financial/business 分对了吗

批量评测（20 条测试用例，覆盖 6 个场景）：

```bash
python batch_infer.py --model /workspace/models/Qwen3-14B --lora output/final --test test_cases.jsonl
```

## 当前效果（v1：460 条 / 5 epochs）

20 条测试用例，完美匹配 7/20（35%）：

| 场景 | 通过 | 说明 |
|---|---|---|
| 单指标（不凑数） | 4/5 | 1 条多提了市占率 |
| 多指标（2-3 个） | 0/5 | 指标名模糊匹配其实大部分对了，但数量偏差 |
| 边界判断（多数字少核心） | 0/3 | 全部多提——该提 1 个提了 2-3 个 |
| financial vs business 分类 | 0/3 | 类型分对了，但数量不匹配 |
| 空输出（背景段落） | 2/2 | 完美 |
| 混合类型 | 1/2 | 1 条多提了净利率 |

**已验证的能力：**
- JSON 格式合法率 20/20（100%）
- financial/business 分类基本准确
- 空输出（纯背景段落）判断完美
- analysis 思维链有逻辑

**主要问题：倾向于多提指标。** 边界判断场景期望 1 个核心指标但模型提了 2-3 个，说明 460 条数据（其中边界负样本仅 72 条）不足以让模型学会"克制"。

**下一步改进方向：**
- 增加 Type B（边界负样本）数据量，从 72 条扩充到 150+ 条
- 或增加训练轮次（epochs 5 → 8-10）
