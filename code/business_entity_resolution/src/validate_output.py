"""Stand-in for the organizers' validator: checks both output files against the
submission rules in the problem statement (stdlib only, streaming).

    python code/business_entity_resolution/src/validate_output.py --out-dir output --test-dir dataset/test
"""
import argparse
import csv
import sys
from pathlib import Path


def read_ids(path: Path) -> set[str]:
    with open(path, encoding="utf-8", newline="") as fh:
        r = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        col = next(r).index("entity_id")
        return {row[col] for row in r}


def rows(path: Path, header: list[str], issues: list[str]):
    """Yield (line_no, s1_id, [ids]) and record format problems."""
    with open(path, encoding="utf-8", newline="") as fh:
        first = fh.readline().rstrip("\n").split("\t")
        if first != header:
            issues.append(f"{path.name}: header {first} != {header}")
        for ln, line in enumerate(fh, 2):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                issues.append(f"{path.name}:{ln}: expected 2 columns, got {len(parts)}")
                continue
            yield ln, parts[0], [x for x in parts[1].split(",") if x] if parts[1] else []


def check_row(name, ln, s1, ids, s1_ids, target_ids, seen, issues):
    if s1 in seen:
        issues.append(f"{name}:{ln}: duplicate row for {s1}")
    seen.add(s1)
    if s1 not in s1_ids:
        issues.append(f"{name}:{ln}: {s1} is not a test Source 1 id")
    if len(set(ids)) != len(ids):
        issues.append(f"{name}:{ln}: duplicate ids in list for {s1}")
    bad = [x for x in ids if not x.startswith(("S2-", "S3-")) or x not in target_ids]
    if bad:
        issues.append(f"{name}:{ln}: invalid ids {bad[:3]} for {s1}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("output"))
    ap.add_argument("--test-dir", type=Path, default=Path("dataset/test"))
    a = ap.parse_args()
    s1_ids = read_ids(a.test_dir / "test_source1.tsv")
    target_ids = read_ids(a.test_dir / "test_source2.tsv") | read_ids(a.test_dir / "test_source3.tsv")

    issues: list[str] = []
    seen_m, seen_c = set(), set()
    n_match = n_cand = empty = not_cand = 0
    mf = a.out_dir / "matching_results.tsv"
    cf = a.out_dir / "candidate_pairs.tsv"
    # Both files are written in the same Source 1 order, so they can be zipped;
    # a mismatch in order is reported and falls back to per-row checks only.
    m_rows = rows(mf, ["source1_entity_id", "matched_entity_ids"], issues)
    c_rows = rows(cf, ["source1_entity_id", "candidate_entity_ids"], issues)
    for (ln, s1, ids), (lc, s1c, cids) in zip(m_rows, c_rows):
        check_row(mf.name, ln, s1, ids, s1_ids, target_ids, seen_m, issues)
        check_row(cf.name, lc, s1c, cids, s1_ids, target_ids, seen_c, issues)
        n_match += len(ids)
        n_cand += len(cids)
        empty += not ids
        if s1 == s1c:
            not_cand += len(set(ids) - set(cids))
        else:
            issues.append(f"line {ln}: files are not in the same Source 1 order ({s1} vs {s1c})")
        if len(issues) > 50:
            issues.append("... stopping after 50 issues")
            break
    for name, seen in ((mf.name, seen_m), (cf.name, seen_c)):
        missing = s1_ids - seen
        if missing:
            issues.append(f"{name}: {len(missing)} Source 1 ids missing, e.g. {sorted(missing)[:3]}")
    if not_cand:
        issues.append(f"{not_cand} matched ids are not in candidate_pairs (pipeline bug)")
    print(f"rows={len(seen_m):,} matched ids={n_match:,} empty rows={empty:,} candidate ids={n_cand:,}")
    if issues:
        print("FAIL")
        for i, msg in enumerate(issues, 1):
            print(f"  {i}. {msg}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
