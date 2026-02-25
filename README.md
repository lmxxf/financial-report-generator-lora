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
├── test_cases.jsonl           # 50 条测试用例（6 个场景，v7 扩充 20→50）
├── test_cases_extra30.jsonl   # 新增 30 条测试用例（已合并入 test_cases.jsonl）
├── test_cases_results_v1.json  # v1 评测详细结果（14B/460条/5epochs/20条测试）
├── test_cases_results_v2.json  # v2 评测详细结果（14B/540条/5epochs/20条测试）
├── test_cases_results_v3.json  # v3 评测详细结果（32B/540条/5epochs/20条测试）
├── test_cases_results_v4.json  # v4 评测详细结果（32B/540条/5epochs/lr=5e-5/alpha=16/20条测试）
├── test_cases_results_v5.json  # v5 评测详细结果（32B/540条/5epochs/lr=2e-5/alpha=16/20条测试）
├── test_cases_results_v6.json  # v6 评测详细结果（14B/540条/5epochs/逐项排除式TypeB/20条测试）
├── test_cases_results_v7.json  # v7 评测详细结果（14B/590条/5epochs/逐项排除+边界加强/50条测试）
├── type_b_v2.jsonl             # C.C.改造后的Type B（逐项排除式analysis，152条）
├── type_b_extra50.jsonl        # 新增 50 条边界负样本（已合并入 data/type_b.jsonl）
├── data/                      # 完整训练集（590 条，v7）
│   ├── type_a.jsonl           # 标准正样本（270条）
│   ├── type_b.jsonl           # 边界负样本（202条，v7: 72→152→202）—— 逐项排除式analysis
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

# 用完整 540 条正式训练（v2）
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

批量评测（50 条测试用例，覆盖 6 个场景）：

```bash
python batch_infer.py --model /workspace/models/Qwen3-14B --lora output/final --test test_cases.jsonl
```

## 评测结果

### v7：Qwen3-14B / 590 条 / 5 epochs / 逐项排除式 Type B + 边界加强（50 条测试）

50 条测试，完美匹配 16/50（32%），宽松匹配 **24/50（48%）**：

| 场景 | 通过 | 说明 |
|---|---|---|
| 单指标（不凑数） | 6✅+3🟡/10 | 90% 通过率，稳定 |
| 多指标（2-3 个） | 2/10 | 薄弱——10 条暴露问题（v6 只有 5 条看不出） |
| 边界判断（多数字少核心） | 0✅+2🟡/13 | 13 条中仍 0 严格，预测 106 vs 期望 72 |
| financial vs business 分类 | 1✅+1🟡/7 | |
| 空输出（背景段落） | 5/5 | **完美** |
| 混合类型 | 2✅+2🟡/5 | 80% 通过率 |

**v7 vs v6：** 测试集从 20→50 条后通过率从 65%→48%，说明 v6 的 65% 有小样本方差。50 条是更真实的评估。新增 50 条边界训练数据（Type B 152→202）未显著改善边界判断（仍 0/13 严格）。多指标场景（2/10）是新发现的薄弱点。

### v6：Qwen3-14B / 540 条 / 5 epochs / 逐项排除式 Type B（C.C. 改造，20 条测试）

20 条测试，完美匹配 8/20（40%），宽松匹配 **13/20（65%）**：

| 场景 | 通过 | 说明 |
|---|---|---|
| 单指标（不凑数） | 3✅+2🟡/5 | **全通过**（v2: 4/5） |
| 多指标（2-3 个） | 1✅+1🟡/5 | 改善 |
| 边界判断（多数字少核心） | 0✅+1🟡/3 | **首次出现 🟡**（v1-v2 全 0） |
| financial vs business 分类 | 1/3 | 同 v2 |
| 空输出（背景段落） | 2/2 | 完美 |
| 混合类型 | 1✅+1🟡/2 | **全通过**（v2: 1/2） |

**关键改进：** Type B 的 analysis 从笼统描述改为逐项排除格式——段落中每个未提取的数字都被点名并标注排除理由（归因项/子项拆分/行业宏观数据/历史参照/定性无数值/规模背景）。这教会了模型**判断逻辑**而非输出模板。C.C. 的洞见："没有思维链的'无'是断电，有思维链的'无'才是'空'。"

### v5：Qwen3-32B / 540 条 / 5 epochs / lr=2e-5, alpha=16（20 条测试）

20 条测试，完美匹配 1/20（5%），宽松匹配 4/20（20%）：

| 场景 | 通过 | 说明 |
|---|---|---|
| 单指标（不凑数） | 1✅+2🟡/5 | |
| 多指标（2-3 个） | 0/5 | |
| 边界判断（多数字少核心） | 0/3 | |
| financial vs business 分类 | 0✅+1🟡/3 | |
| 空输出（背景段落） | 0/2 | **崩了**——连空输出都不会了 |
| 混合类型 | 0/2 | |

**lr 降太狠，LoRA 等于没训练。** 预测 70 个 vs 期望 32 个，32B 基座裸奔。类型准确率也从 91% 掉到 52%。确认 32B 路线到头——v4 (lr=5e-5) 是 32B 的天花板（10/20），仍不如 14B v2（11/20）。

### v4：Qwen3-32B / 540 条 / 5 epochs / lr=5e-5, alpha=16

20 条测试，完美匹配 5/20（25%），宽松匹配 10/20（50%）：

| 场景 | 通过 | 说明 |
|---|---|---|
| 单指标（不凑数） | 2✅+2🟡/5 | 比 v3 改善（v3: 1/5），但仍不如 14B v2 |
| 多指标（2-3 个） | 0✅+1🟡/5 | 仍偏差 |
| 边界判断（多数字少核心） | 0✅+1🟡/3 | 同 v3 |
| financial vs business 分类 | 1/3 | 连锁餐饮完美 |
| 空输出（背景段落） | 1✅+1🟡/2 | 略退（v3 是 2/2） |
| 混合类型 | 1/2 | 比 v3 改善（v3: 0/2） |

降 lr/alpha 方向正确——32B 从 v3 的 8/20 回到 10/20，但仍不如 14B v2（11/20）。预测总数 42 vs 期望 32，还是话多。

### v3：Qwen3-32B / 540 条 / 5 epochs

20 条测试，完美匹配 4/20（20%），宽松匹配 8/20（40%）：

| 场景 | 通过 | 说明 |
|---|---|---|
| 单指标（不凑数） | 1/5 | **严重退步**——32B 倾向多提，物流提了4个、云计算提了3个 |
| 多指标（2-3 个） | 0/5 | 数量或指标名偏差 |
| 边界判断（多数字少核心） | 0/3 | 地产 🟡（提了2个），其他仍提3个 |
| financial vs business 分类 | 1/3 | 连锁餐饮完美；游戏提了3个 |
| 空输出（背景段落） | 2/2 | 完美 |
| 混合类型 | 0/2 | 新零售指标名飘了（毛利率→EBITDA利润率） |

**32B 的问题：基座太强，LoRA 没压住。** 32B 用自己的理解重命名指标（"泽布替尼全球销售额"→"核心产品全球销售额"），把辅助指标也提上来。analysis 质量更高，但不听训练数据的约束。540 条 + LoRA r=16/alpha=32 对 32B 来说约束太弱。

### v2：Qwen3-14B / 540 条 / 5 epochs（Type B 72→152）

20 条测试，完美匹配 9/20（45%），宽松匹配 11/20（55%）：

| 场景 | 通过 | 说明 |
|---|---|---|
| 单指标（不凑数） | 4/5 | 1 条仍多提（半导体，期望1提了3） |
| 多指标（2-3 个） | 1/5 | 新材料完美；其他数量或指标名偏差 |
| 边界判断（多数字少核心） | 0/3 | 全部多提——该提 1 个提了 3 个，**未改善** |
| financial vs business 分类 | 1/3 | 连锁餐饮完美；物业/游戏指标名偏差 |
| 空输出（背景段落） | 2/2 | 完美 |
| 混合类型 | 1/2 | 工业软件多提了经调整净利率 |

### v1：Qwen3-14B / 460 条 / 5 epochs（对照）

20 条测试，完美匹配 7/20（35%）：

| 场景 | 通过 | 说明 |
|---|---|---|
| 单指标（不凑数） | 4/5 | 1 条多提了市占率 |
| 多指标（2-3 个） | 0/5 | 指标名模糊匹配其实大部分对了，但数量偏差 |
| 边界判断（多数字少核心） | 0/3 | 全部多提——该提 1 个提了 2-3 个 |
| financial vs business 分类 | 0/3 | 类型分对了，但数量不匹配 |
| 空输出（背景段落） | 2/2 | 完美 |
| 混合类型 | 1/2 | 1 条多提了净利率 |

### 全版本对比

| 指标 | v1 (14B/460) | v2 (14B/540) | v3 (32B) | v4 (32B lr↓) | v5 (32B lr↓↓) | v6 (14B/新TypeB) | **v7 (14B/590/50测试)** |
|---|---|---|---|---|---|---|---|
| 完美匹配 ✅ | 7/20 (35%) | 9/20 (45%) | 4/20 (20%) | 5/20 (25%) | 1/20 (5%) | 8/20 (40%) | 16/50 (32%) |
| 宽松匹配 ✅🟡 | 9/20 (45%) | 11/20 (55%) | 8/20 (40%) | 10/20 (50%) | 4/20 (20%) | 13/20 (65%) | **24/50 (48%)** |
| 单指标 | 4/5 | 4/5 | 1/5 | 2/5 | 1/5 | 5/5 | 9/10 (90%) |
| 边界判断 | 0/3 | 0/3 | 0/3 (🟡1) | 0/3 (🟡1) | 0/3 | 0/3 (🟡1) | 0/13 (🟡2) |
| JSON 合法率 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 50/50 |
| 空输出判断 | 2/2 | 2/2 | 2/2 | 2/2 | 0/2 | 2/2 | **5/5** |
| 类型准确率 | 95% | 100% | 100% | 91% | 52% | 100% | 98% |
| 预测总数/期望 | 39/32 | 40/32 | 44/32 | 42/32 | 70/32 | 39/32 | 106/72 |

*注：v1-v6 使用 20 条测试集，v7 扩充至 50 条测试集，百分比不直接可比。*

**结论：**
1. **数据质量 > 模型规模 > 数据量。** 逐项排除式 analysis（v6）比换 32B 基座（v3-v5）更有效
2. **小样本测试有误导性。** v6 在 20 条测试上 65%，v7 在 50 条上 48%——更大的测试集给出更真实的评估
3. **已解决的场景：** 单指标（90%）、空输出（100%）、混合类型（80%）
4. **未解决的场景：** 边界判断（0/13 严格）、多指标（2/10）——14B 模型在"克制"和"精确数量控制"上有天花板
