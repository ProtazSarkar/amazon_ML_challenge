#!/usr/bin/env python3
"""
Enhanced Entity Resolution Inspection Tool (EDA)
==================================================
Compares ACTUAL (Ground Truth) Name & Address against PREDICTED Name & Address
for Source 1 entities in the Amazon ML Challenge entity resolution pipeline.

Usage examples:
    # Inspect 5 random mismatched failure cases (default)
    python student_resource/EDA/inspect_failures.py

    # Inspect specific S1 entity IDs
    python student_resource/EDA/inspect_failures.py --s1-id S1-925783039 S1-773889195

    # Inspect False Positives (over-predictions) specifically
    python student_resource/EDA/inspect_failures.py --failure-type fp --num-samples 10

    # Inspect False Negatives (missed matches) specifically
    python student_resource/EDA/inspect_failures.py --failure-type fn --num-samples 5

    # Custom paths
    python student_resource/EDA/inspect_failures.py --predictions student_resource/output/matching_results.tsv
"""

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


def find_file_path(relative_paths: List[str]) -> Path:
    """Finds the first existing path from a list of candidate relative paths."""
    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent

    search_roots = [
        cwd,
        script_dir,
        script_dir.parent,
        script_dir.parent.parent,
    ]

    for root in search_roots:
        for rel in relative_paths:
            candidate = root / rel
            if candidate.exists():
                return candidate.resolve()

    return Path(relative_paths[0])


def load_mapping_file(filepath: Path) -> Dict[str, List[str]]:
    """Loads a mapping TSV file (source1_entity_id -> list of target matched_entity_ids)."""
    mapping = {}
    if not filepath.exists():
        return mapping

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        header = next(f, None)
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            s1_id = parts[0].strip()
            target_ids_str = parts[1].strip() if len(parts) > 1 else ""
            target_ids = [tid.strip() for tid in target_ids_str.split(',') if tid.strip()]
            mapping[s1_id] = target_ids

    return mapping


def stream_source1_entities(s1_path: Path, target_s1_ids: Set[str] = None) -> Dict[str, Dict[str, str]]:
    """Reads Source 1 entities from TSV, optionally filtering by a set of target S1 IDs."""
    records = {}
    if not s1_path.exists():
        print(f"Warning: Source 1 file not found at {s1_path}")
        return records

    with open(s1_path, 'r', encoding='utf-8', errors='replace') as f:
        header_line = next(f, None)
        if not header_line:
            return records
        headers = [h.strip() for h in header_line.rstrip('\n').split('\t')]

        for line in f:
            parts = [p.strip() for p in line.rstrip('\n').split('\t')]
            if len(parts) < len(headers):
                parts.extend([''] * (len(headers) - len(parts)))
            row_dict = dict(zip(headers, parts))
            s1_id = row_dict.get('entity_id', '')

            if not target_s1_ids or s1_id in target_s1_ids:
                records[s1_id] = row_dict
                if target_s1_ids and len(records) == len(target_s1_ids):
                    break

    return records


def stream_target_entities(
    s2_path: Path, s3_path: Path, target_ids: Set[str]
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    """Streams and retrieves specific target entities from Source 2 and Source 3 files."""
    s2_records = {}
    s3_records = {}
    remaining_ids = set(target_ids)

    if not remaining_ids:
        return s2_records, s3_records

    # Stream Source 2
    if s2_path.exists():
        with open(s2_path, 'r', encoding='utf-8', errors='replace') as f:
            header_line = next(f, None)
            if header_line:
                headers = [h.strip() for h in header_line.rstrip('\n').split('\t')]
                for line in f:
                    parts = [p.strip() for p in line.rstrip('\n').split('\t')]
                    if len(parts) >= len(headers):
                        tid = parts[0].strip()
                        if tid in remaining_ids:
                            s2_records[tid] = dict(zip(headers, parts))
                            remaining_ids.remove(tid)
                            if not remaining_ids:
                                break

    # Stream Source 3 for remaining target IDs
    if remaining_ids and s3_path.exists():
        with open(s3_path, 'r', encoding='utf-8', errors='replace') as f:
            header_line = next(f, None)
            if header_line:
                headers = [h.strip() for h in header_line.rstrip('\n').split('\t')]
                for line in f:
                    parts = [p.strip() for p in line.rstrip('\n').split('\t')]
                    if len(parts) >= len(headers):
                        tid = parts[0].strip()
                        if tid in remaining_ids:
                            s3_records[tid] = dict(zip(headers, parts))
                            remaining_ids.remove(tid)
                            if not remaining_ids:
                                break

    return s2_records, s3_records


def compute_f05(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Calculates Precision, Recall, and F0.5 Score."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    beta_sq = 0.5 ** 2  # 0.25
    f05 = (
        (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f05


def inspect_entity_resolution(args):
    """Main execution function for inspection."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    # Resolve paths
    train_dir = find_file_path(['student_resource/dataset/train', 'dataset/train'])
    s1_path = train_dir / 'train_source1.tsv'
    s2_path = train_dir / 'train_source2.tsv'
    s3_path = train_dir / 'train_source3.tsv'
    gt_path = train_dir / 'train_ground_truth.tsv'

    pred_path = find_file_path([
        args.predictions,
        'student_resource/output/matching_results.tsv',
        'output/matching_results.tsv',
    ])

    print("=" * 100)
    print("                      ENTITY RESOLUTION COMPARISON & INSPECTION TOOL                      ")
    print("=" * 100)
    print(f" Dataset Train Directory : {train_dir}")
    print(f" Predictions File        : {pred_path}")
    print(f" Ground Truth File       : {gt_path}")
    print("-" * 100)

    # Load Ground Truth and Predictions mappings
    print("Loading Ground Truth and Predictions mappings...")
    ground_truth_map = load_mapping_file(gt_path)
    predictions_map = load_mapping_file(pred_path)

    if not ground_truth_map:
        print(f"Error: Ground truth file not found or empty at: {gt_path}")
        return

    if not predictions_map:
        print(f"Warning: Predictions file not found or empty at: {pred_path}")
        print("Note: Will display Ground Truth entities only.")

    # Determine candidate S1 IDs
    # CRITICAL FIX: Filter S1 IDs to those present in the predictions file so we evaluate
    # actual model output instead of un-evaluated entities from full ground truth!
    if predictions_map and not args.eval_all_gt:
        eval_s1_ids = [s1_id for s1_id in ground_truth_map if s1_id in predictions_map]
        print(f"Restricting inspection pool to S1 IDs present in predictions file ({len(eval_s1_ids)} / {len(ground_truth_map)} GT entities).")
    else:
        eval_s1_ids = list(ground_truth_map.keys())

    if args.s1_id:
        target_s1_ids = []
        for raw_id in args.s1_id:
            target_s1_ids.extend([item.strip() for item in raw_id.split(',') if item.strip()])
    else:
        # Categorize available S1 entities in the evaluation pool based on failure type filter
        fp_ids = []
        fn_ids = []
        mismatch_ids = []
        perfect_ids = []

        for s1_id in eval_s1_ids:
            gt_set = set(ground_truth_map.get(s1_id, []))
            pred_set = set(predictions_map.get(s1_id, []))

            has_fp = bool(pred_set - gt_set)
            has_fn = bool(gt_set - pred_set)

            if has_fp:
                fp_ids.append(s1_id)
            if has_fn:
                fn_ids.append(s1_id)
            if has_fp or has_fn:
                mismatch_ids.append(s1_id)
            if gt_set and gt_set == pred_set:
                perfect_ids.append(s1_id)

        if args.failure_type == 'fp':
            candidate_pool = fp_ids if fp_ids else mismatch_ids
            filter_desc = "False Positives (over-predictions)"
        elif args.failure_type == 'fn':
            candidate_pool = fn_ids if fn_ids else mismatch_ids
            filter_desc = "False Negatives (missed ground truth)"
        elif args.failure_type == 'mismatch':
            candidate_pool = mismatch_ids if mismatch_ids else eval_s1_ids
            filter_desc = "Mismatches (False Positives or False Negatives)"
        elif args.failure_type == 'perfect':
            candidate_pool = perfect_ids if perfect_ids else eval_s1_ids
            filter_desc = "Perfect Matches"
        else:
            candidate_pool = eval_s1_ids
            filter_desc = "All / Random"

        print(f"Filter Mode: [{filter_desc}] | Matching Entities Found in Pool: {len(candidate_pool)}")

        random.seed(42)  # Consistent reproducible sampling
        num_to_sample = min(args.num_samples, len(candidate_pool))
        target_s1_ids = random.sample(candidate_pool, num_to_sample) if candidate_pool else []

    if not target_s1_ids:
        print("No S1 entity IDs found to inspect.")
        return

    print(f"Inspecting {len(target_s1_ids)} Source 1 entity record(s)...")

    # Load Source 1 records for target S1 IDs
    s1_records = stream_source1_entities(s1_path, set(target_s1_ids))

    # Collect all needed target entity IDs (S2/S3) from Ground Truth and Predictions
    needed_target_ids = set()
    for s1_id in target_s1_ids:
        needed_target_ids.update(ground_truth_map.get(s1_id, []))
        needed_target_ids.update(predictions_map.get(s1_id, []))

    print(f"Fetching {len(needed_target_ids)} target entities (Source 2 / Source 3)...")
    s2_records, s3_records = stream_target_entities(s2_path, s3_path, needed_target_ids)

    # Global tracking counters for inspected set
    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Display entity comparisons
    for idx, s1_id in enumerate(target_s1_ids, 1):
        s1_data = s1_records.get(s1_id, {})
        gt_ids = ground_truth_map.get(s1_id, [])
        pred_ids = predictions_map.get(s1_id, [])

        gt_set = set(gt_ids)
        pred_set = set(pred_ids)

        tp_set = gt_set.intersection(pred_set)
        fp_set = pred_set - gt_set
        fn_set = gt_set - pred_set

        tp = len(tp_set)
        fp = len(fp_set)
        fn = len(fn_set)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        prec, rec, f05 = compute_f05(tp, fp, fn)

        # Status badge determination
        if not gt_set and not pred_set:
            status_badge = "[🚫 SINGLETON - NO MATCHES]"
        elif tp > 0 and fp == 0 and fn == 0:
            status_badge = "[✅ PERFECT MATCH]"
        elif fp > 0 and fn > 0:
            status_badge = "[❌ OVER-PREDICTION & MISSED MATCH]"
        elif fp > 0:
            status_badge = "[❌ FALSE POSITIVE (OVER-PREDICTION)]"
        elif fn > 0:
            status_badge = "[⚠️ FALSE NEGATIVE (MISSED MATCH)]"
        else:
            status_badge = "[ℹ️ UNKNOWN]"

        print("\n" + "=" * 100)
        print(f"CASE {idx}/{len(target_s1_ids)} | S1 ENTITY ID: {s1_id}  {status_badge}")
        print(f"Metrics for ID: Precision={prec:.2f} | Recall={rec:.2f} | F0.5={f05:.2f} (TP={tp}, FP={fp}, FN={fn})")
        print("=" * 100)

        # ---------------------------------------------------------------------
        # 1. SOURCE 1 QUERY RECORD
        # ---------------------------------------------------------------------
        s1_name = s1_data.get('business_name', 'N/A')
        s1_addr = s1_data.get('business_address', 'N/A')
        s1_country = s1_data.get('country', 'N/A')

        print(f"\n📍 [QUERY] SOURCE 1 RECORD:")
        print(f"   • Entity ID       : {s1_id}")
        print(f"   • Country         : {s1_country}")
        print(f"   • Business Name   : {s1_name}")
        print(f"   • Business Address: {s1_addr}")

        # ---------------------------------------------------------------------
        # 2. ACTUAL GROUND TRUTH MATCHES
        # ---------------------------------------------------------------------
        print(f"\n🎯 [ACTUAL] GROUND TRUTH MATCHES ({len(gt_ids)} record(s)):")
        if not gt_ids:
            print("   (No ground truth matches exist for this entity - Singleton)")
        else:
            for g_idx, tid in enumerate(gt_ids, 1):
                rec_info = s2_records.get(tid) or s3_records.get(tid)
                source_name = "Source 2" if tid in s2_records else ("Source 3" if tid in s3_records else "Unknown")

                if tid in tp_set:
                    match_indicator = "✅ [CORRECTLY PREDICTED BY MODEL]"
                else:
                    match_indicator = "⚠️ [MISSED BY MODEL - FALSE NEGATIVE]"

                print(f"\n   {g_idx}. Target ID: {tid} ({source_name})  -->  {match_indicator}")
                if rec_info:
                    print(f"      • Country         : {rec_info.get('country', 'N/A')}")
                    print(f"      • Actual Name     : {rec_info.get('business_name', 'N/A')}")
                    print(f"      • Actual Address  : {rec_info.get('business_address', 'N/A')}")
                else:
                    print(f"      • Details         : [Record ID present in GT, but not found in S2/S3 files]")

        # ---------------------------------------------------------------------
        # 3. PREDICTED MATCHES
        # ---------------------------------------------------------------------
        print(f"\n🤖 [PREDICTED] MODEL MATCHING RESULTS ({len(pred_ids)} record(s)):")
        if not pred_ids:
            print("   (Model predicted NO matches for this entity)")
        else:
            for p_idx, tid in enumerate(pred_ids, 1):
                rec_info = s2_records.get(tid) or s3_records.get(tid)
                source_name = "Source 2" if tid in s2_records else ("Source 3" if tid in s3_records else "Unknown")

                if tid in gt_set:
                    pred_indicator = "✅ [TRUE POSITIVE - GROUND TRUTH CONFIRMED]"
                else:
                    pred_indicator = "❌ [FALSE POSITIVE - INCORRECT PREDICTION]"

                print(f"\n   {p_idx}. Target ID: {tid} ({source_name})  -->  {pred_indicator}")
                if rec_info:
                    print(f"      • Country         : {rec_info.get('country', 'N/A')}")
                    print(f"      • Predicted Name  : {rec_info.get('business_name', 'N/A')}")
                    print(f"      • Predicted Address: {rec_info.get('business_address', 'N/A')}")
                else:
                    print(f"      • Details         : [Predicted ID not found in S2/S3 files]")

        print("-" * 100)

    # -------------------------------------------------------------------------
    # OVERALL SUMMARY STATISTICS
    # -------------------------------------------------------------------------
    overall_prec, overall_rec, overall_f05 = compute_f05(total_tp, total_fp, total_fn)

    print("\n" + "=" * 100)
    print("                             INSPECTION SUMMARY STATISTICS                             ")
    print("=" * 100)
    print(f" Total Inspected S1 Entities : {len(target_s1_ids)}")
    print(f" Total True Positives  (TP)  : {total_tp}")
    print(f" Total False Positives (FP)  : {total_fp}")
    print(f" Total False Negatives (FN)  : {total_fn}")
    print(f" Aggregate Precision         : {overall_prec:.4f}")
    print(f" Aggregate Recall            : {overall_rec:.4f}")
    print(f" Aggregate F0.5 Score        : {overall_f05:.4f}")
    print("=" * 100 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect Actual vs. Predicted Name & Address for Entity Resolution."
    )
    parser.add_argument(
        '--s1-id',
        nargs='+',
        help="Specific Source 1 entity ID(s) to inspect (e.g. --s1-id S1-925783039 S1-773889195)",
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5,
        help="Number of entities to sample for inspection (default: 5)",
    )
    parser.add_argument(
        '--failure-type',
        choices=['all', 'mismatch', 'fp', 'fn', 'perfect'],
        default='mismatch',
        help="Type of entity resolution cases to sample: mismatch (default), fp (False Positives), fn (False Negatives), perfect, or all",
    )
    parser.add_argument(
        '--predictions',
        type=str,
        default='student_resource/output/matching_results.tsv',
        help="Path to matching results predictions TSV file",
    )
    parser.add_argument(
        '--eval-all-gt',
        action='store_true',
        help="Evaluate all ground truth S1 IDs instead of restricting sampling pool to S1 IDs present in predictions file",
    )

    args = parser.parse_args()
    inspect_entity_resolution(args)


if __name__ == '__main__':
    main()