#!/usr/bin/env python3
"""
make_splits.py

Create deterministic train/valid/test splits (8/1/1) over the HYBRID dataset (mm + text),
then write split artifacts into subfolders:

  <data_dir>/train/
  <data_dir>/valid/
  <data_dir>/test/

Each split folder contains the same set of files:
  mm_corpus.jsonl   mm_qrels.tsv   mm_queries.jsonl
  text_corpus.jsonl text_qrels.tsv text_queries.jsonl

Key points:
- Splitting is deterministic via SHA1(seed::qid) over the MERGED (offset) qid space,
  so mm and text are split consistently without qid collisions.
- Each split folder gets:
  - queries/qrels: subsetted to that split's qids
  - corpus: copied in full (same corpus pool for all splits), with image_path reformatted

Image path rewrite:
If a corpus entry has an absolute image_path like:
  /.../images_subset_kg/Q34266/05474001.jpg
it will be rewritten to:
  Q34266/05474001.jpg
(implemented as "last two path components" when possible).

Usage:
  python3 make_splits.py --data_dir /home/xmw5190/KG-MMRAG/MKG_Analogy/out/group_retrieval_no_unused/ \
    --split_seed markg_v1

By default it looks for:
  mm_queries.jsonl, mm_corpus.jsonl, mm_qrels.tsv
  text_queries.jsonl, text_corpus.jsonl, text_qrels.tsv
under --data_dir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Dict, List, Tuple, Iterable


# ----------------------------
# I/O: load jsonl / tsv qrels
# ----------------------------
def load_jsonl(path: Path) -> List[dict]:
    items: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error in {path} at line {line_no}: {e}") from e
    return items


def load_queries(path: Path) -> Dict[int, dict]:
    data = load_jsonl(path)
    out: Dict[int, dict] = {}
    for obj in data:
        if "qid" not in obj:
            raise KeyError(f"Missing 'qid' in queries file: {path}")
        qid = int(obj["qid"])
        out[qid] = obj
    return out


def load_corpus(path: Path) -> Dict[int, dict]:
    data = load_jsonl(path)
    out: Dict[int, dict] = {}
    for obj in data:
        if "doc_id" not in obj:
            raise KeyError(f"Missing 'doc_id' in corpus file: {path}")
        doc_id = int(obj["doc_id"])
        out[doc_id] = obj
    return out


def load_qrels_tsv(path: Path) -> Dict[int, Dict[int, int]]:
    """
    Returns:
      qrels[qid][doc_id] = relevance (int)
    """
    qrels: Dict[int, Dict[int, int]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Bad qrels line in {path} at line {line_no}: '{line}'")
            qid = int(parts[0])
            doc_id = int(parts[1])
            rel = int(parts[2])
            qrels.setdefault(qid, {})[doc_id] = rel
    return qrels


# ----------------------------
# Deterministic split (8/1/1)
# ----------------------------
def split_qids_deterministic(
    qids: List[int],
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: str = "markg_v1",
) -> Tuple[List[int], List[int], List[int]]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-9:
        raise ValueError("train/val/test ratios must sum to 1.0")

    def key(qid: int) -> str:
        s = f"{seed}::{qid}".encode("utf-8")
        return hashlib.sha1(s).hexdigest()

    qids_sorted = sorted(qids, key=key)
    n = len(qids_sorted)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = qids_sorted[:n_train]
    val = qids_sorted[n_train : n_train + n_val]
    test = qids_sorted[n_train + n_val :]
    return train, val, test


# ----------------------------
# Hybrid merge helpers (avoid qid/doc_id collisions)
# ----------------------------
def offset_queries(queries: Dict[int, dict], qid_offset: int) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    for qid, q in queries.items():
        q2 = dict(q)
        q2["qid"] = int(qid) + qid_offset
        out[int(qid) + qid_offset] = q2
    return out


def offset_qrels(
    qrels: Dict[int, Dict[int, int]],
    qid_offset: int,
) -> Dict[int, Dict[int, int]]:
    out: Dict[int, Dict[int, int]] = {}
    for qid, rels in qrels.items():
        new_qid = int(qid) + qid_offset
        out[new_qid] = {int(doc_id): int(rel) for doc_id, rel in rels.items()}
    return out


def merge_hybrid_for_splitting(
    *,
    mm_queries: Dict[int, dict],
    mm_qrels: Dict[int, Dict[int, int]],
    text_queries: Dict[int, dict],
    text_qrels: Dict[int, Dict[int, int]],
) -> Tuple[Dict[int, dict], Dict[int, Dict[int, int]], int]:
    """
    Merge ONLY queries + qrels into a shared qid space by offsetting text qids.

    Returns:
      merged_queries, merged_qrels, qid_offset_used_for_text
    """
    mm_max_qid = max(mm_queries.keys()) if mm_queries else -1
    qid_offset = mm_max_qid + 1

    text_queries_off = offset_queries(text_queries, qid_offset)
    text_qrels_off = offset_qrels(text_qrels, qid_offset)

    merged_queries = dict(mm_queries)
    merged_queries.update(text_queries_off)

    merged_qrels = dict(mm_qrels)
    merged_qrels.update(text_qrels_off)

    return merged_queries, merged_qrels, qid_offset


# ----------------------------
# Split eligibility
# ----------------------------
def overlapping_qids(queries: Dict[int, dict], qrels: Dict[int, Dict[int, int]]) -> List[int]:
    return [qid for qid in sorted(queries.keys()) if qid in qrels]


# ----------------------------
# Image path rewrite
# ----------------------------
def reform_image_path(p: object) -> object:
    """
    Rewrite an absolute path like:
      /.../images_subset_kg/Q34266/05474001.jpg
    to:
      Q34266/05474001.jpg

    Implemented as: if path has >=2 parts, return last2 joined by "/".
    """
    if p is None:
        return None
    s = str(p).strip()
    if not s:
        return p
    # Normalize to posix separators for splitting.
    s2 = s.replace("\\", "/")
    parts = [x for x in s2.split("/") if x]
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return s2


def rewrite_corpus_image_paths(corpus: Dict[int, dict]) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    for doc_id, doc in corpus.items():
        d2 = dict(doc)
        if "image_path" in d2:
            d2["image_path"] = reform_image_path(d2.get("image_path"))
        out[int(doc_id)] = d2
    return out


# ----------------------------
# Writers
# ----------------------------
def write_jsonl_from_mapping(path: Path, mapping: Dict[int, dict], *, key_field: str) -> None:
    """
    Write JSONL by sorting on integer key, ensuring the key_field is consistent.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for k in sorted(mapping.keys()):
            obj = dict(mapping[k])
            obj[key_field] = int(k)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_qrels_tsv(path: Path, qrels: Dict[int, Dict[int, int]]) -> None:
    """
    Write qrels TSV with lines: qid doc_id rel
    Sorted by qid then doc_id.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for qid in sorted(qrels.keys()):
            rels = qrels[qid]
            for doc_id in sorted(rels.keys()):
                f.write(f"{int(qid)}\t{int(doc_id)}\t{int(rels[doc_id])}\n")


def filter_queries(queries: Dict[int, dict], keep_qids: Iterable[int]) -> Dict[int, dict]:
    keep = set(int(x) for x in keep_qids)
    return {int(qid): queries[int(qid)] for qid in keep if int(qid) in queries}


def filter_qrels(qrels: Dict[int, Dict[int, int]], keep_qids: Iterable[int]) -> Dict[int, Dict[int, int]]:
    keep = set(int(x) for x in keep_qids)
    return {int(qid): qrels[int(qid)] for qid in keep if int(qid) in qrels}


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    default_data_dir = Path("/home/xmw5190/KG-MMRAG/MKG_Analogy/out/group_retrieval_no_unused/")
    p.add_argument("--data_dir", type=Path, default=default_data_dir)

    p.add_argument("--mm_queries", type=Path, default=None)
    p.add_argument("--mm_corpus", type=Path, default=None)
    p.add_argument("--mm_qrels", type=Path, default=None)

    p.add_argument("--text_queries", type=Path, default=None)
    p.add_argument("--text_corpus", type=Path, default=None)
    p.add_argument("--text_qrels", type=Path, default=None)

    p.add_argument("--split_seed", type=str, default="markg_v1")

    # ratios (keep defaults 8/1/1)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--valid_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)

    # output folder names
    p.add_argument("--train_dirname", type=str, default="train")
    p.add_argument("--valid_dirname", type=str, default="valid")
    p.add_argument("--test_dirname", type=str, default="test")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    args.mm_queries = args.mm_queries or (args.data_dir / "mm_queries.jsonl")
    args.mm_corpus = args.mm_corpus or (args.data_dir / "mm_corpus.jsonl")
    args.mm_qrels = args.mm_qrels or (args.data_dir / "mm_qrels.tsv")

    args.text_queries = args.text_queries or (args.data_dir / "text_queries.jsonl")
    args.text_corpus = args.text_corpus or (args.data_dir / "text_corpus.jsonl")
    args.text_qrels = args.text_qrels or (args.data_dir / "text_qrels.tsv")

    mm_exists = args.mm_queries.exists() and args.mm_corpus.exists() and args.mm_qrels.exists()
    text_exists = args.text_queries.exists() and args.text_corpus.exists() and args.text_qrels.exists()
    if not (mm_exists and text_exists):
        raise ValueError("Expected BOTH mm and text files to exist. Check --data_dir and file paths.")

    # load originals
    mm_queries = load_queries(args.mm_queries)
    mm_corpus = load_corpus(args.mm_corpus)
    mm_qrels = load_qrels_tsv(args.mm_qrels)

    text_queries = load_queries(args.text_queries)
    text_corpus = load_corpus(args.text_corpus)
    text_qrels = load_qrels_tsv(args.text_qrels)

    # rewrite image paths in corpora (for ALL splits)
    mm_corpus_rw = rewrite_corpus_image_paths(mm_corpus)
    text_corpus_rw = rewrite_corpus_image_paths(text_corpus)

    # build merged space for deterministic splitting across hybrid
    merged_queries, merged_qrels, qid_offset = merge_hybrid_for_splitting(
        mm_queries=mm_queries,
        mm_qrels=mm_qrels,
        text_queries=text_queries,
        text_qrels=text_qrels,
    )

    all_qids_merged = overlapping_qids(merged_queries, merged_qrels)
    if not all_qids_merged:
        raise ValueError("No overlapping qids between merged queries and qrels.")

    tr_m, va_m, te_m = split_qids_deterministic(
        all_qids_merged,
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.valid_ratio),
        test_ratio=float(args.test_ratio),
        seed=str(args.split_seed),
    )

    split2merged = {
        "train": tr_m,
        "valid": va_m,
        "test": te_m,
    }

    # helper: map merged qids back to original mm/text qids
    def merged_to_mm_qids(merged_qids: List[int]) -> List[int]:
        return [int(q) for q in merged_qids if int(q) < qid_offset]

    def merged_to_text_qids(merged_qids: List[int]) -> List[int]:
        return [int(q) - qid_offset for q in merged_qids if int(q) >= qid_offset]

    # write split folders
    name2dir = {
        "train": args.data_dir / str(args.train_dirname),
        "valid": args.data_dir / str(args.valid_dirname),
        "test": args.data_dir / str(args.test_dirname),
    }

    for split_name, merged_qids in split2merged.items():
        out_dir = name2dir[split_name]
        out_dir.mkdir(parents=True, exist_ok=True)

        mm_qids = merged_to_mm_qids(merged_qids)
        text_qids = merged_to_text_qids(merged_qids)

        # subset queries/qrels; keep corpus full
        mm_queries_split = filter_queries(mm_queries, mm_qids)
        mm_qrels_split = filter_qrels(mm_qrels, mm_qids)

        text_queries_split = filter_queries(text_queries, text_qids)
        text_qrels_split = filter_qrels(text_qrels, text_qids)

        # write mm
        write_jsonl_from_mapping(out_dir / "mm_queries.jsonl", mm_queries_split, key_field="qid")
        write_jsonl_from_mapping(out_dir / "mm_corpus.jsonl", mm_corpus_rw, key_field="doc_id")
        write_qrels_tsv(out_dir / "mm_qrels.tsv", mm_qrels_split)

        # write text
        write_jsonl_from_mapping(out_dir / "text_queries.jsonl", text_queries_split, key_field="qid")
        write_jsonl_from_mapping(out_dir / "text_corpus.jsonl", text_corpus_rw, key_field="doc_id")
        write_qrels_tsv(out_dir / "text_qrels.tsv", text_qrels_split)

    # small confirmation to stdout (safe for piping/logging)
    print(
        json.dumps(
            {
                "data_dir": str(args.data_dir),
                "seed": args.split_seed,
                "qid_offset_for_text": qid_offset,
                "counts_merged": {
                    "all": len(all_qids_merged),
                    "train": len(tr_m),
                    "valid": len(va_m),
                    "test": len(te_m),
                },
                "out_folders": {k: str(v) for k, v in name2dir.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
