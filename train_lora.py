#!/usr/bin/env python3
"""
金融研报指标提取 —— QLoRA 微调脚本

模型：Qwen3-14B（8bit 量化加载）
方法：QLoRA（LoRA rank=16, alpha=32）
数据：sample_data.jsonl 或 data/ 目录下的完整训练集

用法：
    # 用 50 条 demo 数据快速验证
    python train_lora.py --data sample_data.jsonl --epochs 3

    # 用完整 500 条数据正式训练
    python train_lora.py --data data/ --epochs 5

    # 指定模型路径
    python train_lora.py --model /home/lmxxf/work/models/Qwen3-14B --data data/ --epochs 5

依赖安装：
    pip install torch transformers peft bitsandbytes datasets accelerate trl
"""

import os
import json
import argparse
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


# ============================================================
# 数据处理
# ============================================================

def load_training_data(data_path: str) -> list:
    """加载训练数据，支持单个 jsonl 文件或目录"""
    records = []
    path = Path(data_path)

    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(path.glob("*.jsonl"))
    else:
        raise FileNotFoundError(f"找不到数据路径: {data_path}")

    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[WARN] {f}:{line_no} JSON 解析失败: {e}")

    print(f"加载 {len(records)} 条训练数据，来自 {len(files)} 个文件")
    return records


def format_prompt(record: dict) -> str:
    """将一条训练数据格式化为模型的输入输出对"""
    inp = record["input"]
    out = record["output"]

    # 构造 system prompt
    system = (
        "你是一位金融研报指标提取专家。根据给定的章节标题和段落内容，"
        "提取被深度分析的核心指标，输出 JSON。"
        "先在 analysis 中分析段落结构，再给出 metrics 列表。"
        "指标数量不固定，根据段落实际内容决定。"
    )

    # 构造 user prompt
    user = f"【章节标题】{inp['chapter_title']}\n【段落内容】{inp['paragraph']}"

    # 构造 assistant response
    assistant = json.dumps(out, ensure_ascii=False, indent=2)

    # Qwen3 chat format
    text = (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant}<|im_end|>"
    )
    return text


def prepare_dataset(records: list) -> Dataset:
    """将训练数据转为 HuggingFace Dataset"""
    texts = [format_prompt(r) for r in records]
    return Dataset.from_dict({"text": texts})


# ============================================================
# 模型加载
# ============================================================

def load_model_and_tokenizer(model_path: str):
    """加载 8bit 量化模型和 tokenizer"""
    print(f"加载模型: {model_path}")

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    model = prepare_model_for_kbit_training(model)

    return model, tokenizer


def apply_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05):
    """应用 LoRA 适配器"""
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


# ============================================================
# 训练
# ============================================================

def train(
    model_path: str,
    data_path: str,
    output_dir: str,
    epochs: int = 5,
    batch_size: int = 2,
    grad_accum: int = 8,
    lr: float = 2e-4,
    max_seq_len: int = 2048,
    lora_r: int = 16,
    lora_alpha: int = 32,
):
    """QLoRA 训练主流程"""

    # 1. 加载数据
    records = load_training_data(data_path)
    dataset = prepare_dataset(records)

    # 2. 加载模型
    model, tokenizer = load_model_and_tokenizer(model_path)
    model = apply_lora(model, r=lora_r, alpha=lora_alpha)

    # 3. 训练参数
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        warmup_steps=int(0.05 * epochs * len(dataset) / (batch_size * grad_accum)),
        lr_scheduler_type="cosine",
        report_to="none",
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        max_length=max_seq_len,
    )

    # 4. 训练器
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    # 5. 开跑
    print(f"\n{'='*60}")
    print(f"开始训练")
    print(f"  数据: {len(records)} 条")
    print(f"  模型: {model_path}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch: {batch_size} × {grad_accum} = {batch_size * grad_accum}")
    print(f"  LoRA: r={lora_r}, alpha={lora_alpha}")
    print(f"  输出: {output_dir}")
    print(f"{'='*60}\n")

    trainer.train()

    # 6. 保存 LoRA 权重
    final_dir = os.path.join(output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nLoRA 权重已保存到: {final_dir}")


# ============================================================
# 推理（训练完后测试用）
# ============================================================

def inference(model_path: str, lora_path: str, chapter_title: str, paragraph: str):
    """加载 LoRA 权重做推理测试"""
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()

    system = (
        "你是一位金融研报指标提取专家。根据给定的章节标题和段落内容，"
        "提取被深度分析的核心指标，输出 JSON。"
        "先在 analysis 中分析段落结构，再给出 metrics 列表。"
        "指标数量不固定，根据段落实际内容决定。"
    )
    user = f"【章节标题】{chapter_title}\n【段落内容】{paragraph}"

    prompt = (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(response)
    return response


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="金融研报指标提取 QLoRA 训练")
    sub = parser.add_subparsers(dest="command", help="子命令")

    # train 子命令
    train_parser = sub.add_parser("train", help="训练 LoRA")
    train_parser.add_argument("--model", default="/home/lmxxf/work/models/Qwen3-14B",
                              help="基座模型路径")
    train_parser.add_argument("--data", default="data/", help="训练数据路径（jsonl 文件或目录）")
    train_parser.add_argument("--output", default="output/", help="输出目录")
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--batch-size", type=int, default=2)
    train_parser.add_argument("--grad-accum", type=int, default=8)
    train_parser.add_argument("--lr", type=float, default=2e-4)
    train_parser.add_argument("--max-seq-len", type=int, default=2048)
    train_parser.add_argument("--lora-r", type=int, default=16)
    train_parser.add_argument("--lora-alpha", type=int, default=32)

    # infer 子命令
    infer_parser = sub.add_parser("infer", help="推理测试")
    infer_parser.add_argument("--model", default="/home/lmxxf/work/models/Qwen3-14B",
                              help="基座模型路径")
    infer_parser.add_argument("--lora", required=True, help="LoRA 权重路径")
    infer_parser.add_argument("--title", required=True, help="章节标题")
    infer_parser.add_argument("--text", required=True, help="段落内容")

    args = parser.parse_args()

    if args.command == "train":
        train(
            model_path=args.model,
            data_path=args.data,
            output_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr,
            max_seq_len=args.max_seq_len,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
        )
    elif args.command == "infer":
        inference(
            model_path=args.model,
            lora_path=args.lora,
            chapter_title=args.title,
            paragraph=args.text,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
