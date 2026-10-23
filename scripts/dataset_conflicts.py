#!/usr/bin/env python3
"""从 gold 子集提取重复/矛盾文本分组明细。

每行一个唯一文本，记录全部判定历史（verdict/quality/kind/path/time/id）、
按 quality 优先级+最新时间裁决出的 chosen_verdict，以及是否矛盾（同文本既有 ad 又有 clean）。

用法:
  python scripts/dataset_conflicts.py                  # 输出 data/raw/dataset/converted/dataset_conflicts.jsonl
"""
import gzip
import json
from collections import Counter
from pathlib import Path

QUALITY_RANK = {"llm_reviewed": 5, "manual": 4, "rule": 3, "hash": 2, "so_initial": 1, "other": 0}
SRC = "data/raw/dataset/ft/antiad_ft_gold.jsonl.gz"
OUT = "data/raw/dataset/converted/dataset_conflicts.jsonl"


def main():
    groups = {}
    with gzip.open(SRC, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            groups.setdefault(r["text"], []).append(r)

    dups = {t: rows for t, rows in groups.items() if len(rows) > 1}
    out_rows = []
    for text, rows in dups.items():
        rows_sorted = sorted(rows, key=lambda r: (-QUALITY_RANK.get(r["quality"], 0), r["time"]))
        chosen = rows_sorted[0]
        verdicts = {r["verdict"] for r in rows}
        out_rows.append({
            "text": text,
            "chosen_verdict": chosen["verdict"],
            "chosen_quality": chosen["quality"],
            "chosen_kind": chosen.get("ad_kind"),
            "conflict": len(verdicts) > 1,
            "sightings": len(rows),
            "rows": [{"verdict": r["verdict"], "quality": r["quality"],
                      "kind": r.get("ad_kind"), "path": r["path"],
                      "time": r["time"], "id": r["id"]} for r in rows_sorted],
        })

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total_rows = sum(r["sightings"] for r in out_rows)
    print(f"[OK] 重复文本组={len(out_rows)}  涉及行数={total_rows}  -> {OUT}")
    print("矛盾组(同文本 ad+clean 并存):", sum(1 for r in out_rows if r["conflict"]))
    print("矛盾组裁决结果:", dict(Counter(r["chosen_verdict"] for r in out_rows if r["conflict"])))
    print("非矛盾重复组裁决:", dict(Counter(r["chosen_verdict"] for r in out_rows if not r["conflict"])))
    print("sightings 分布:", dict(Counter(r["sightings"] for r in out_rows).most_common(8)))


if __name__ == "__main__":
    main()
