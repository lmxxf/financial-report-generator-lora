#!/usr/bin/env python3
"""
金融研报指标提取 —— 三层评估脚本

评估维度：
1. 核心命中率（语义匹配）：预测指标是否覆盖了标注中的核心指标
2. 精确率/召回率：多提了什么、漏了什么
3. 类型准确率：financial/business 分类是否正确

用法：
    python eval_script.py --pred predictions.jsonl --gold gold.jsonl
    python eval_script.py --pred predictions.jsonl --gold gold.jsonl --method embedding --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

输入格式（每行一个JSON）：
    {"input": {...}, "output": {"metrics": [{"metric_name": "...", "metric_type": "financial|business", "score": 0.9}]}}

    pred 和 gold 的行数必须一致，按行对齐。
"""

import json
import argparse
import sys
from collections import defaultdict
from difflib import SequenceMatcher


# ============================================================
# 匹配策略
# ============================================================

def fuzzy_match(name_a: str, name_b: str, threshold: float = 0.7) -> bool:
    """基于编辑距离的模糊匹配（默认方案，不需要额外依赖）"""
    # 预处理：去空格、统一小写
    a = name_a.strip().lower()
    b = name_b.strip().lower()
    if a == b:
        return True
    # 包含关系
    if a in b or b in a:
        return True
    # SequenceMatcher
    ratio = SequenceMatcher(None, a, b).ratio()
    return ratio >= threshold


def embedding_match(name_a: str, name_b: str, model, threshold: float = 0.85) -> bool:
    """基于 embedding 余弦相似度的语义匹配（需要 sentence-transformers）"""
    from numpy import dot
    from numpy.linalg import norm
    emb_a = model.encode(name_a)
    emb_b = model.encode(name_b)
    cos_sim = dot(emb_a, emb_b) / (norm(emb_a) * norm(emb_b))
    return cos_sim >= threshold


# ============================================================
# 核心评估逻辑
# ============================================================

def evaluate_single(pred_metrics: list, gold_metrics: list, match_fn) -> dict:
    """评估单条样本"""
    result = {
        "gold_count": len(gold_metrics),
        "pred_count": len(pred_metrics),
        "hits": 0,           # 命中数（gold中被匹配到的）
        "type_correct": 0,   # 类型判断正确数（在命中的前提下）
        "false_positives": 0,  # 多提的
        "false_negatives": 0,  # 漏提的
    }

    gold_matched = [False] * len(gold_metrics)
    pred_matched = [False] * len(pred_metrics)

    # 贪心匹配：对每个 gold 指标，找最佳匹配的 pred 指标
    for gi, g in enumerate(gold_metrics):
        for pi, p in enumerate(pred_metrics):
            if pred_matched[pi]:
                continue
            if match_fn(g.get("metric_name", ""), p.get("metric_name", "")):
                gold_matched[gi] = True
                pred_matched[pi] = True
                result["hits"] += 1
                if g.get("metric_type", "").lower() == p.get("metric_type", "").lower():
                    result["type_correct"] += 1
                break

    result["false_positives"] = sum(1 for m in pred_matched if not m)
    result["false_negatives"] = sum(1 for m in gold_matched if not m)

    return result


def compute_metrics(results: list) -> dict:
    """汇总所有样本的评估结果"""
    total = len(results)
    if total == 0:
        return {}

    # 完美匹配（所有gold都命中且没有多余的pred）
    perfect = sum(1 for r in results
                  if r["hits"] == r["gold_count"] and r["false_positives"] == 0)

    total_hits = sum(r["hits"] for r in results)
    total_gold = sum(r["gold_count"] for r in results)
    total_pred = sum(r["pred_count"] for r in results)
    total_type_correct = sum(r["type_correct"] for r in results)
    total_fp = sum(r["false_positives"] for r in results)
    total_fn = sum(r["false_negatives"] for r in results)

    precision = total_hits / total_pred if total_pred > 0 else 0
    recall = total_hits / total_gold if total_gold > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    type_acc = total_type_correct / total_hits if total_hits > 0 else 0

    return {
        "total_samples": total,
        "perfect_match": perfect,
        "perfect_match_rate": round(perfect / total * 100, 2),
        "core_hit_rate": round(recall * 100, 2),  # 核心命中率 = 召回率
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2),
        "type_accuracy": round(type_acc * 100, 2),
        "avg_gold_per_sample": round(total_gold / total, 2),
        "avg_pred_per_sample": round(total_pred / total, 2),
        "total_false_positives": total_fp,
        "total_false_negatives": total_fn,
    }


def compute_metrics_by_type(results_with_type: list) -> dict:
    """按 metric_type 分组统计"""
    # 根据 gold 中的 metric_type 分组
    groups = defaultdict(list)
    for r in results_with_type:
        # 判断样本的主要类型
        gold_types = set(m.get("metric_type", "").lower() for m in r["gold_metrics"])
        if len(gold_types) > 1:
            group = "mixed"
        elif gold_types:
            group = gold_types.pop()
        else:
            group = "unknown"
        groups[group].append(r["eval_result"])

    return {k: compute_metrics(v) for k, v in sorted(groups.items())}


# ============================================================
# 主流程
# ============================================================

def load_jsonl(path: str) -> list:
    """加载 jsonl 文件"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] {path} 第{line_no}行 JSON 解析失败: {e}", file=sys.stderr)
    return data


def extract_metrics(item: dict) -> list:
    """从一条数据中提取 metrics 列表"""
    output = item.get("output", item)
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except json.JSONDecodeError:
            return []
    metrics = output.get("metrics", output.get("candidates", []))
    return metrics if isinstance(metrics, list) else []


def main():
    parser = argparse.ArgumentParser(description="金融研报指标提取 - 三层评估")
    parser.add_argument("--pred", required=True, help="模型预测结果 (jsonl)")
    parser.add_argument("--gold", required=True, help="标注数据 (jsonl)")
    parser.add_argument("--method", choices=["fuzzy", "embedding"], default="fuzzy",
                        help="匹配方法: fuzzy(默认,无依赖) 或 embedding(需要sentence-transformers)")
    parser.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                        help="embedding 模型名 (仅 --method embedding 时生效)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="匹配阈值 (fuzzy默认0.7, embedding默认0.85)")
    parser.add_argument("--detail", action="store_true", help="输出每条样本的详细评估")
    args = parser.parse_args()

    # 加载数据
    pred_data = load_jsonl(args.pred)
    gold_data = load_jsonl(args.gold)

    if len(pred_data) != len(gold_data):
        print(f"[ERROR] pred({len(pred_data)}条) 和 gold({len(gold_data)}条) 行数不一致!", file=sys.stderr)
        sys.exit(1)

    # 配置匹配函数
    if args.method == "embedding":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("[ERROR] 需要安装 sentence-transformers: pip install sentence-transformers", file=sys.stderr)
            sys.exit(1)
        print(f"加载 embedding 模型: {args.model} ...", file=sys.stderr)
        emb_model = SentenceTransformer(args.model)
        threshold = args.threshold or 0.85
        match_fn = lambda a, b: embedding_match(a, b, emb_model, threshold)
    else:
        threshold = args.threshold or 0.7
        match_fn = lambda a, b: fuzzy_match(a, b, threshold)

    # 逐条评估
    all_results = []
    results_with_type = []

    for i, (pred, gold) in enumerate(zip(pred_data, gold_data)):
        pred_metrics = extract_metrics(pred)
        gold_metrics = extract_metrics(gold)
        r = evaluate_single(pred_metrics, gold_metrics, match_fn)
        all_results.append(r)
        results_with_type.append({
            "eval_result": r,
            "gold_metrics": gold_metrics,
        })

        if args.detail:
            gold_names = [m.get("metric_name", "?") for m in gold_metrics]
            pred_names = [m.get("metric_name", "?") for m in pred_metrics]
            status = "PERFECT" if r["hits"] == r["gold_count"] and r["false_positives"] == 0 else "MISS"
            print(f"[{i+1:3d}] {status}  gold={gold_names}  pred={pred_names}  "
                  f"hits={r['hits']} fp={r['false_positives']} fn={r['false_negatives']}")

    # 汇总
    overall = compute_metrics(all_results)
    by_type = compute_metrics_by_type(results_with_type)

    print("\n" + "=" * 60)
    print("整体评估结果")
    print("=" * 60)
    for k, v in overall.items():
        print(f"  {k:25s}: {v}")

    print("\n" + "=" * 60)
    print("按类型分组评估")
    print("=" * 60)
    for group, metrics in by_type.items():
        print(f"\n  [{group}]")
        for k, v in metrics.items():
            print(f"    {k:25s}: {v}")

    print()


if __name__ == "__main__":
    main()
