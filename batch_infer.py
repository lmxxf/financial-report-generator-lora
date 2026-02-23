#!/usr/bin/env python3
"""
批量推理 + 简易评测

用法（docker 容器内）：
    python batch_infer.py --model /workspace/models/Qwen3-14B --lora output/final --test test_cases.jsonl
"""

import json
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def load_model(model_path, lora_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    return model, tokenizer


def run_inference(model, tokenizer, chapter_title, paragraph):
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
            **inputs, max_new_tokens=512, temperature=0.1,
            do_sample=True, pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def char_similarity(a: str, b: str) -> float:
    """字符级 Jaccard 相似度（bigram）"""
    if not a or not b:
        return 0.0
    a_bigrams = {a[i:i+2] for i in range(len(a) - 1)} if len(a) > 1 else {a}
    b_bigrams = {b[i:i+2] for i in range(len(b) - 1)} if len(b) > 1 else {b}
    intersection = a_bigrams & b_bigrams
    union = a_bigrams | b_bigrams
    return len(intersection) / len(union) if union else 0.0


def match_metrics(gold_name: str, pred_name: str, threshold: float = 0.5) -> bool:
    """判断两个指标名是否匹配：精确、子串包含、或字符相似度"""
    if gold_name == pred_name:
        return True
    if gold_name in pred_name or pred_name in gold_name:
        return True
    if char_similarity(gold_name, pred_name) >= threshold:
        return True
    return False


def evaluate(gold_metrics, pred_metrics):
    """评测：比较指标名称和类型（精确 + 子串 + 相似度）"""
    gold_names = [m["metric_name"] for m in gold_metrics]
    pred_names = [m["metric_name"] for m in pred_metrics]

    # 精确匹配
    exact_hits = set(g for g in gold_names if g in pred_names)

    # 宽松匹配（子串 + 相似度）—— 贪心匹配避免重复
    matched_gold = set()
    matched_pred = set()
    match_pairs = []  # (gold_idx, pred_idx)
    for gi, g in enumerate(gold_names):
        best_score = 0
        best_pi = -1
        for pi, p in enumerate(pred_names):
            if pi in matched_pred:
                continue
            if match_metrics(g, p):
                score = char_similarity(g, p)
                if g == p:
                    score = 1.0
                if score > best_score:
                    best_score = score
                    best_pi = pi
        if best_pi >= 0:
            matched_gold.add(gi)
            matched_pred.add(best_pi)
            match_pairs.append((gi, best_pi))

    # 类型准确率
    gold_types = {i: m["metric_type"] for i, m in enumerate(gold_metrics)}
    pred_types = {i: m["metric_type"] for i, m in enumerate(pred_metrics)}
    type_correct = sum(1 for gi, pi in match_pairs if gold_types[gi] == pred_types[pi])

    return {
        "gold_count": len(gold_names),
        "pred_count": len(pred_names),
        "exact_hits": len(exact_hits),
        "fuzzy_hits": len(matched_gold),
        "type_correct": type_correct,
        "type_total": len(match_pairs),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--lora", required=True)
    parser.add_argument("--test", required=True)
    args = parser.parse_args()

    # 加载测试数据
    cases = []
    with open(args.test, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    print(f"加载 {len(cases)} 条测试用例")
    print(f"加载模型...")
    model, tokenizer = load_model(args.model, args.lora)

    # 逐条推理
    results = []
    total_exact = total_fuzzy = total_gold = total_pred = total_type_ok = total_type_n = 0
    count_ok = count_loose = count_wrong = 0

    for i, case in enumerate(cases):
        inp = case["input"]
        gold = case["output"]
        scenario = case.get("scenario", "unknown")

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(cases)}] {scenario} | {inp['chapter_title']}")

        raw = run_inference(model, tokenizer, inp["chapter_title"], inp["paragraph"])

        # 尝试解析 JSON
        try:
            pred = json.loads(raw)
            pred_metrics = pred.get("metrics", [])
            valid_json = True
        except json.JSONDecodeError:
            pred_metrics = []
            valid_json = False
            print(f"  ⚠ JSON 解析失败")

        gold_metrics = gold.get("metrics", [])
        ev = evaluate(gold_metrics, pred_metrics)

        total_gold += ev["gold_count"]
        total_pred += ev["pred_count"]
        total_exact += ev["exact_hits"]
        total_fuzzy += ev["fuzzy_hits"]
        total_type_ok += ev["type_correct"]
        total_type_n += ev["type_total"]

        # 严格匹配：数量完全一致 + 模糊全覆盖
        count_match = ev["gold_count"] == ev["pred_count"]
        strict_ok = count_match and ev["fuzzy_hits"] == ev["gold_count"]
        # 宽松匹配：数量偏差 ≤1 + 模糊全覆盖 gold
        loose_ok = abs(ev["gold_count"] - ev["pred_count"]) <= 1 and ev["fuzzy_hits"] == ev["gold_count"]

        if strict_ok:
            count_ok += 1
            status = "✅"
        elif loose_ok:
            count_loose += 1
            status = "🟡"
        else:
            count_wrong += 1
            status = "❌"

        gold_names = [m["metric_name"] for m in gold_metrics]
        pred_names = [m["metric_name"] for m in pred_metrics]
        print(f"  {status} 期望 {ev['gold_count']} 个: {gold_names}")
        print(f"     预测 {ev['pred_count']} 个: {pred_names}")
        print(f"     精确命中: {ev['exact_hits']} | 模糊命中: {ev['fuzzy_hits']}")

        results.append({
            "index": i + 1,
            "scenario": scenario,
            "title": inp["chapter_title"],
            "valid_json": valid_json,
            "gold_count": ev["gold_count"],
            "pred_count": ev["pred_count"],
            "exact_hits": ev["exact_hits"],
            "fuzzy_hits": ev["fuzzy_hits"],
            "status": status,
            "pred_raw": raw,
        })

    # 汇总
    print(f"\n{'='*60}")
    print(f"评测结果汇总")
    print(f"{'='*60}")
    print(f"  总用例: {len(cases)}")
    print(f"  严格匹配 ✅: {count_ok}/{len(cases)} ({100*count_ok/len(cases):.0f}%)")
    print(f"  宽松匹配 🟡: {count_loose}/{len(cases)} ({100*count_loose/len(cases):.0f}%) (数量±1, 指标全覆盖)")
    print(f"  合计通过: {count_ok+count_loose}/{len(cases)} ({100*(count_ok+count_loose)/len(cases):.0f}%)")
    print(f"  模糊召回: {total_fuzzy}/{total_gold} ({100*total_fuzzy/total_gold:.0f}%)" if total_gold else "")
    print(f"  精确召回: {total_exact}/{total_gold} ({100*total_exact/total_gold:.0f}%)" if total_gold else "")
    print(f"  类型准确: {total_type_ok}/{total_type_n} ({100*total_type_ok/total_type_n:.0f}%)" if total_type_n else "")
    print(f"  数量偏差: 期望总计 {total_gold} 个，预测总计 {total_pred} 个")

    # 按场景分组统计
    from collections import defaultdict
    by_scenario = defaultdict(lambda: {"total": 0, "strict": 0, "loose": 0, "fail": 0})
    for r in results:
        s = r["scenario"]
        by_scenario[s]["total"] += 1
        if r["status"] == "✅":
            by_scenario[s]["strict"] += 1
        elif r["status"] == "🟡":
            by_scenario[s]["loose"] += 1
        else:
            by_scenario[s]["fail"] += 1

    print(f"\n按场景分组 (✅严格 / 🟡宽松 / ❌失败):")
    for s, v in sorted(by_scenario.items()):
        print(f"  {s}: ✅{v['strict']} 🟡{v['loose']} ❌{v['fail']} / {v['total']}")

    # 保存详细结果
    out_path = Path(args.test).with_suffix(".results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存到: {out_path}")


if __name__ == "__main__":
    main()
