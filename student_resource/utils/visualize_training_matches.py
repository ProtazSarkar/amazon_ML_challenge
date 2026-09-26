#!/usr/bin/env python3
"""Display training ground-truth matches with decoded business records.

The source files are large, so this script only keeps the records needed for
the selected ground-truth rows in memory. It uses the standard library only.
"""

import argparse
import csv
import os
import sys


Record = dict[str, str]
MatchRow = tuple[str, list[str]]


def read_ground_truth(
    path: str, limit: int | None, requested_s1: set[str] | None
) -> list[MatchRow]:
    """Read selected S1 rows and their comma-separated match IDs."""
    rows: list[MatchRow] = []
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            s1_id = (row.get("source1_entity_id") or "").strip()
            if not s1_id or (requested_s1 and s1_id not in requested_s1):
                continue

            raw_matches = (row.get("matched_entity_ids") or "").strip()
            match_ids = [value.strip() for value in raw_matches.split(",") if value.strip()]
            rows.append((s1_id, match_ids))
            if limit is not None and len(rows) >= limit and not requested_s1:
                break
    return rows


def load_selected_records(path: str, wanted_ids: set[str]) -> dict[str, Record]:
    """Scan one source file and return only records whose IDs are requested."""
    records: dict[str, Record] = {}
    if not wanted_ids:
        return records

    with open(path, "r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            entity_id = (row.get("entity_id") or "").strip()
            if entity_id in wanted_ids:
                records[entity_id] = {
                    "business_name": (row.get("business_name") or "").strip(),
                    "business_address": (row.get("business_address") or "").strip(),
                    "country": (row.get("country") or "").strip(),
                }
                if len(records) == len(wanted_ids):
                    break
    return records


def format_record(entity_id: str, record: Record | None, indent: str = "  ") -> str:
    """Format one decoded entity record for a terminal."""
    if record is None:
        return f"{indent}{entity_id}  [record not found in source files]"
    return "\n".join(
        [
            f"{indent}{entity_id} [{record['country'] or 'country unknown'}]",
            f"{indent}  Name:    {record['business_name'] or '(blank)'}",
            f"{indent}  Address: {record['business_address'] or '(blank)' }",
        ]
    )


def print_matches(rows: list[MatchRow], records: dict[str, Record]) -> None:
    """Print decoded matches in a readable, grouped layout."""
    print("Training entity-resolution matches")
    print("=" * 38)
    print(f"Showing {len(rows)} Source 1 entr{'y' if len(rows) == 1 else 'ies'}")

    for index, (s1_id, match_ids) in enumerate(rows, start=1):
        print(f"\n[{index}] Source 1 reference")
        print(format_record(s1_id, records.get(s1_id)))
        if not match_ids:
            print("  Matches: none (singleton)")
            continue

        print(f"  Matches: {len(match_ids)}")
        for match_id in match_ids:
            print(format_record(match_id, records.get(match_id), indent="    "))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decode training ground-truth IDs into readable business matches."
    )
    parser.add_argument(
        "--train-dir",
        default=os.path.join("dataset", "train"),
        help="Training TSV directory (default: dataset/train)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of ground-truth rows to show (default: 5; ignored with --s1-id)",
    )
    parser.add_argument(
        "--s1-id",
        action="append",
        help="Show this Source 1 ID; repeat the option to show multiple IDs",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args(argv)
    if args.limit < 1:
        print("--limit must be at least 1", file=sys.stderr)
        return 2

    train_dir = args.train_dir
    requested_s1 = set(args.s1_id or [])
    rows = read_ground_truth(
        os.path.join(train_dir, "train_ground_truth.tsv"),
        limit=None if requested_s1 else args.limit,
        requested_s1=requested_s1,
    )
    if not rows:
        print("No matching ground-truth rows found.", file=sys.stderr)
        return 1

    wanted_s1 = {s1_id for s1_id, _ in rows}
    wanted_s2 = {match_id for _, match_ids in rows for match_id in match_ids if match_id.startswith("S2-")}
    wanted_s3 = {match_id for _, match_ids in rows for match_id in match_ids if match_id.startswith("S3-")}

    records = {}
    records.update(load_selected_records(os.path.join(train_dir, "train_source1.tsv"), wanted_s1))
    records.update(load_selected_records(os.path.join(train_dir, "train_source2.tsv"), wanted_s2))
    records.update(load_selected_records(os.path.join(train_dir, "train_source3.tsv"), wanted_s3))
    print_matches(rows, records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())