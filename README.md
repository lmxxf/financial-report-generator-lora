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
├── eval_script.py             # 三层评估脚本（语义匹配）
├── training_data_strategy.md  # 训练数据策略文档
├── sample_data.jsonl          # 50 条 demo 数据
└── data/                      # 完整训练集（~500条）
    ├── type_a.jsonl           # 标准正样本（270条）
    ├── type_b.jsonl           # 边界负样本（72条）—— 教"什么不该提"
    ├── type_c.jsonl           # 数量变化样本（72条）—— 打破"永远提2个"
    └── type_d.jsonl           # 混合类型专项（46条）—— 练 financial/business 分类
```

## 环境

DGX Spark (128GB) + magical_bhabha docker 容器。

```bash
docker exec -it magical_bhabha bash
pip install peft bitsandbytes trl  # 其它依赖容器里已有
```

| 库 | 作用 |
|---|---|
| peft | LoRA 实现，把全量微调变成只训几百万参数 |
| bitsandbytes | 8bit 量化加载，14B 模型从 28GB 压到 ~15GB |
| trl | HuggingFace 训练器，SFTTrainer 处理 chat 格式对齐 |

## 训练

基座模型：Qwen3-14B，8bit 量化加载（QLoRA）。

```bash
# 下载模型（中国镜像）
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen3-14B --local-dir /home/lmxxf/work/models/Qwen3-14B

# 用 50 条 demo 快速验证流程
python train_lora.py train --data sample_data.jsonl --epochs 3

# 用完整 500 条正式训练
python train_lora.py train --data data/ --epochs 5
```

训练参数（默认值）：

| 参数 | 值 | 说明 |
|---|---|---|
| LoRA r | 16 | 低秩矩阵维度 |
| LoRA alpha | 32 | 缩放系数 |
| batch size | 2 × 8 (grad accum) = 16 | 有效批大小 |
| learning rate | 2e-4 | 余弦衰减 |
| max seq len | 2048 | 最大序列长度 |
| 量化 | 8bit (bitsandbytes) | QLoRA |
| 显存估算 | ~20GB | 14B 8bit + LoRA |

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
