#!/usr/bin/env python3
"""
Severity-weighted error analysis for BPM benchmark suite.

Severity tiers:
  EXTRA:
    tier=0 (trivial): within 50ms of manual S1 → double-detection; algorithm found
                      the right beat but fired twice
    tier=1 (minor):   noisy neighbor (manual context has 'noisy') OR SNR<3 →
                      noise/boundary confusion
    tier=2 (moderate):50-200ms from nearest S1 → timing offset; beat found in wrong
                      phase window
    tier=3 (severe):  >200ms from any S1, no noisy context → true false positive

  MISS:
    tier=0 (trivial): nearest predicted S1 or S2 within 50ms → timing jitter
    tier=1 (minor):   SNR<5 at manual S1 → signal too quiet to expect detection
    tier=2 (moderate):50-150ms from nearest prediction → timing offset
    tier=3 (severe):  >150ms away, SNR>=5 → truly missed clear beat

  PHASE_FLIP: always tier=3 (inverted labels)

Weights: tier0=0.1, tier1=0.3, tier2=0.6, tier3=1.0
"""
import json, sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
with open('benchmark_result.json') as f:
    data = json.load(f)

WEIGHTS = {0: 0.1, 1: 0.3, 2: 0.6, 3: 1.0}
LABELS  = {0: "trivial", 1: "minor", 2: "moderate", 3: "severe"}


def nearest_s1_dist_ms(err):
    """Distance (ms) from error time range to nearest manual S1 context entry."""
    t, dur = err['time_sec'], err.get('duration_sec', 0)
    s1s = [c for c in err.get('manual_context', []) if c['state'] == 'S1']
    if not s1s:
        return 9999
    dists = []
    for c in s1s:
        s, e = c['start_sec'], c['end_sec']
        overlap_start = max(t, s)
        overlap_end   = min(t + dur, e)
        if overlap_start < overlap_end:
            dists.append(0)
        else:
            dists.append(min(abs(t - e), abs(t + dur - s)) * 1000)
    return min(dists)


def nearest_pred_dist_ms(err):
    """Distance (ms) from manual S1 time to nearest predicted S1 context entry."""
    t, dur = err['time_sec'], err.get('duration_sec', 0)
    preds = [c for c in err.get('predicted_context', []) if c['state'] == 'S1']
    if not preds:
        return 9999
    dists = []
    for c in preds:
        s, e = c['start_sec'], c['end_sec']
        overlap_start = max(t, s)
        overlap_end   = min(t + dur, e)
        if overlap_start < overlap_end:
            dists.append(0)
        else:
            dists.append(min(abs(t - e), abs(t + dur - s)) * 1000)
    return min(dists)


def has_noisy_context(err):
    return any(c['state'] == 'noisy' for c in err.get('manual_context', []))


def classify_extra(err):
    dist = nearest_s1_dist_ms(err)
    noisy = has_noisy_context(err)
    snr = err.get('snr') or 0

    if dist < 50:
        return 0, "double_detection"        # found the beat, fired twice
    if noisy or snr < 3:
        return 1, "noisy_region"            # noise/boundary confusion
    if dist < 200:
        return 2, "timing_offset"           # beat region found, wrong timing
    return 3, "true_extra"                  # genuine false positive


def classify_miss(err):
    pred_dist = nearest_pred_dist_ms(err)
    snr = err.get('snr') or 0

    if pred_dist < 50:
        return 0, "timing_jitter"           # found it, slightly off
    if snr < 5:
        return 1, "low_snr"                 # too quiet to reliably detect
    if pred_dist < 150:
        return 2, "timing_offset"           # close but not close enough
    return 3, "true_miss"                   # genuinely missed clear beat


def classify_flip(err):
    return 3, "phase_flip"


# ------------------------------------------------------------------
# Build enriched error list
# ------------------------------------------------------------------
all_errors = []
for fd in data['errors_by_file']:
    fname = fd['file']
    for e in fd['errors']:
        t = e['type']
        if t == 'extra':
            tier, label = classify_extra(e)
        elif t == 'miss':
            tier, label = classify_miss(e)
        else:
            tier, label = classify_flip(e)
        all_errors.append({**e, 'file': fname, 'tier': tier, 'category': label,
                           'weight': WEIGHTS[tier]})

total_raw    = len(all_errors)
total_weight = sum(e['weight'] for e in all_errors)

print(f"Total errors: {total_raw}  (weighted total: {total_weight:.1f})")
print(f"Error rate: {data['error_rate_pct']:.2f}%  "
      f"Weighted rate: {total_weight/data['total_manual_s1']*100:.2f}%\n")

# ------------------------------------------------------------------
# By tier
# ------------------------------------------------------------------
print("=== BY SEVERITY TIER ===")
for tier in range(4):
    errs = [e for e in all_errors if e['tier'] == tier]
    w = sum(e['weight'] for e in errs)
    print(f"  Tier {tier} ({LABELS[tier]:8s}): {len(errs):4d} raw  {w:6.1f} weighted")

# ------------------------------------------------------------------
# By category
# ------------------------------------------------------------------
from collections import Counter, defaultdict
print("\n=== BY CATEGORY ===")
cats = Counter(e['category'] for e in all_errors)
cat_weight = defaultdict(float)
for e in all_errors:
    cat_weight[e['category']] += e['weight']
for cat, cnt in cats.most_common():
    t = [e['tier'] for e in all_errors if e['category'] == cat][0]
    print(f"  {cat:20s} (tier {t}): {cnt:4d} raw  {cat_weight[cat]:6.1f} weighted")

# ------------------------------------------------------------------
# By error type × tier
# ------------------------------------------------------------------
print("\n=== ERROR TYPE BREAKDOWN ===")
for etype in ('extra', 'miss', 'phase_flip'):
    errs = [e for e in all_errors if e['type'] == etype]
    by_tier = Counter(e['tier'] for e in errs)
    w = sum(e['weight'] for e in errs)
    parts = "  ".join(f"t{t}:{by_tier.get(t,0)}" for t in range(4))
    print(f"  {etype:12s}: {len(errs):4d} raw  {w:6.1f} weighted  [{parts}]")

# ------------------------------------------------------------------
# Top files by weighted error
# ------------------------------------------------------------------
print("\n=== TOP FILES BY WEIGHTED ERROR COUNT ===")
file_stats = defaultdict(lambda: {'raw': 0, 'weighted': 0.0, 'tiers': Counter()})
for e in all_errors:
    fn = os.path.basename(e['file'])[:70]
    file_stats[fn]['raw'] += 1
    file_stats[fn]['weighted'] += e['weight']
    file_stats[fn]['tiers'][e['tier']] += 1

for fn, s in sorted(file_stats.items(), key=lambda x: -x[1]['weighted'])[:15]:
    tiers = "  ".join(f"t{t}:{s['tiers'].get(t,0)}" for t in range(4))
    print(f"  raw={s['raw']:4d}  wtd={s['weighted']:6.1f}  [{tiers}]  {fn[:60]}")

# ------------------------------------------------------------------
# Severe (tier 3) extras detail — the ones that matter most
# ------------------------------------------------------------------
severe_extras = [e for e in all_errors if e['type'] == 'extra' and e['tier'] == 3]
print(f"\n=== TIER-3 EXTRAS ({len(severe_extras)}) — breakdown by manual context ===")
near_s1_dist = [nearest_s1_dist_ms(e) for e in severe_extras]
bpms = [e.get('bpm_at_time', 0) for e in severe_extras]
snrs = [e.get('snr') or 0 for e in severe_extras]
import numpy as np
print(f"  dist to nearest S1: mean={np.mean(near_s1_dist):.0f}ms  "
      f"median={np.median(near_s1_dist):.0f}ms  max={max(near_s1_dist):.0f}ms")
print(f"  BPM: mean={np.mean(bpms):.0f}  min={min(bpms):.0f}  max={max(bpms):.0f}")
print(f"  SNR: mean={np.mean(snrs):.1f}  min={min(snrs):.1f}  max={max(snrs):.1f}")

# What manual state sits around these severe extras?
surrounding_states = Counter()
for e in severe_extras:
    for c in e.get('manual_context', []):
        if abs(c['offset']) == 1:
            surrounding_states[c['state']] += 1
print(f"  Surrounding manual states (offset ±1): {dict(surrounding_states.most_common())}")

# ------------------------------------------------------------------
# Severe misses detail
# ------------------------------------------------------------------
severe_misses = [e for e in all_errors if e['type'] == 'miss' and e['tier'] == 3]
print(f"\n=== TIER-3 MISSES ({len(severe_misses)}) ===")
miss_snrs = [e.get('snr') or 0 for e in severe_misses]
print(f"  SNR: mean={np.mean(miss_snrs):.1f}  min={min(miss_snrs):.1f}  max={max(miss_snrs):.1f}")
# Top files for severe misses
miss_files = Counter(os.path.basename(e['file'])[:60] for e in severe_misses)
for fn, cnt in miss_files.most_common(5):
    print(f"  {cnt:3d}  {fn}")
