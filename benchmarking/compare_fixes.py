#!/usr/bin/env python3
"""
Compare two benchmark summary snapshots.

Usage:
    python compare_fixes.py <before.json> <after.json>
    python compare_fixes.py benchmark_summary_before.json benchmark_summary.json

When a file exists in only one snapshot it is reported separately — new or
removed labeled files do NOT pollute the per-file delta table or the totals.
Only files present in BOTH snapshots are compared.
"""
import json, sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def load(path: str) -> dict:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def file_index(data: dict) -> dict:
    """Map basename → per-file stats dict."""
    # Works for both benchmark_summary.json (per_file[].file is basename)
    # and benchmark_result.json (per_file[].file is full path).
    idx = {}
    for r in data.get('per_file', []):
        key = os.path.basename(r.get('file', ''))
        idx[key] = r
    return idx


def get_counts(r: dict):
    total  = r.get('errors') or r.get('total_errors', 0)
    extra  = r.get('extra')  or r.get('extra_errors', 0)
    miss   = r.get('miss')   or r.get('miss_errors', 0)
    flip   = r.get('flip')   or r.get('flip_errors', 0)
    return total, extra, miss, flip


def main():
    if len(sys.argv) == 3:
        before_path, after_path = sys.argv[1], sys.argv[2]
    elif len(sys.argv) == 2:
        # One arg: treat current benchmark_summary.json as "after", arg as "before"
        before_path = sys.argv[1]
        after_path  = 'benchmark_summary.json'
    else:
        before_path = 'benchmark_summary_before.json'
        after_path  = 'benchmark_summary.json'

    before = load(before_path)
    after  = load(after_path)

    b_idx = file_index(before)
    a_idx = file_index(after)

    common   = sorted(set(b_idx) & set(a_idx))
    only_before = sorted(set(b_idx) - set(a_idx))
    only_after  = sorted(set(a_idx) - set(b_idx))

    # ── Per-file delta table (common files only) ──────────────────────────────
    changed = []
    same    = 0
    for fn in common:
        tb, xb, mb, fb = get_counts(b_idx[fn])
        ta, xa, ma, fa = get_counts(a_idx[fn])
        d = ta - tb
        if d != 0:
            changed.append((d, fn, tb, ta, mb, ma, xb, xa, fb, fa))
        else:
            same += 1

    changed.sort()

    print(f"Before : {before_path}")
    print(f"After  : {after_path}")
    print(f"Common files: {len(common)}   only-before: {len(only_before)}   only-after: {len(only_after)}")
    print()

    if changed:
        print(f"{'File':<62} {'bef':>4} {'aft':>4} {'Δ':>4}  miss→  extra→  flip→")
        print("─" * 104)
        for d, fn, tb, ta, mb, ma, xb, xa, fb, fa in changed:
            sign = '+' if d > 0 else ''
            print(f"  {fn[:60]:<60} {tb:>4} {ta:>4} {sign}{d:>3}  "
                  f"{mb}→{ma:<3}  {xb}→{xa:<4}  {fb}→{fa}")
        print(f"  {'(unchanged)':<60} {same:>4} file(s)")
    else:
        print("No per-file changes.")

    # ── Totals over common files only ─────────────────────────────────────────
    tot_b = sum(get_counts(b_idx[fn])[0] for fn in common)
    tot_a = sum(get_counts(a_idx[fn])[0] for fn in common)
    ex_b  = sum(get_counts(b_idx[fn])[1] for fn in common)
    ex_a  = sum(get_counts(a_idx[fn])[1] for fn in common)
    ms_b  = sum(get_counts(b_idx[fn])[2] for fn in common)
    ms_a  = sum(get_counts(a_idx[fn])[2] for fn in common)
    fl_b  = sum(get_counts(b_idx[fn])[3] for fn in common)
    fl_a  = sum(get_counts(a_idx[fn])[3] for fn in common)
    s1_b  = sum((b_idx[fn].get('manual_s1') or b_idx[fn].get('manual_s1_count', 0)) for fn in common)
    s1_a  = sum((a_idx[fn].get('manual_s1') or a_idx[fn].get('manual_s1_count', 0)) for fn in common)

    print()
    print(f"{'':─<104}")
    print(f"  {'Totals (common files only)':<60} {tot_b:>4} {tot_a:>4} {tot_a-tot_b:>+4}  "
          f"{ms_b}→{ms_a:<3}  {ex_b}→{ex_a:<4}  {fl_b}→{fl_a}")
    s1 = s1_b or s1_a or 1
    print(f"  Error rate (common)  before={tot_b/s1*100:.2f}%  after={tot_a/s1*100:.2f}%  "
          f"Δ={( tot_a-tot_b)/s1*100:+.2f}%")

    # ── Files only in one snapshot ─────────────────────────────────────────────
    if only_before:
        print(f"\nFiles removed since before ({len(only_before)}):")
        for fn in only_before:
            tb, xb, mb, fb = get_counts(b_idx[fn])
            print(f"  {fn[:70]}  errors={tb}")
    if only_after:
        print(f"\nNew files in after ({len(only_after)}):")
        for fn in only_after:
            ta, xa, ma, fa = get_counts(a_idx[fn])
            print(f"  {fn[:70]}  errors={ta}")


if __name__ == '__main__':
    main()
