#!/usr/bin/env python3
"""
Paired comparison of two evaluation runs (e.g. naive vs reranked).

Both runs score the same questions, so each metric is compared question by
question, with a 95% bootstrap confidence interval on the mean difference.

    python scripts/compare_runs.py evals/results/<naive>.json evals/results/<reranked>.json
"""
import json
import random
import statistics as st
import sys

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def main(a_path: str, b_path: str, n_boot: int = 5000, seed: int = 0) -> None:
    a, b = json.load(open(a_path)), json.load(open(b_path))
    A = {r["id"]: r for r in a["rows"]}
    B = {r["id"]: r for r in b["rows"]}
    ids = sorted(set(A) & set(B))
    print(f"{a['pipeline_variant']} vs {b['pipeline_variant']}: {len(ids)} paired questions, "
          f"judge {a['params']['judge_model']}, generator {a['generator_model']}")
    print(f"{'metric':18s} {a['pipeline_variant']:>9s} {b['pipeline_variant']:>9s}   diff   95% CI            better/worse/same")
    rng = random.Random(seed)
    for m in METRICS:
        pairs = [(A[i]["scores"].get(m), B[i]["scores"].get(m)) for i in ids]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        d = [y - x for x, y in pairs]
        boots = sorted(st.mean(rng.choice(d) for _ in d) for _ in range(n_boot))
        lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]
        better = sum(x > 0.05 for x in d)
        worse = sum(x < -0.05 for x in d)
        flag = "  *" if lo > 0 or hi < 0 else ""
        print(f"{m:18s} {st.mean(x for x, _ in pairs):9.3f} {st.mean(y for _, y in pairs):9.3f}  {st.mean(d):+.3f}  "
              f"[{lo:+.3f}, {hi:+.3f}]  {better}/{worse}/{len(d) - better - worse}{flag}")
    print("* interval excludes zero")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
