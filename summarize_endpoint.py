#!/usr/bin/env python3
"""Aggregate completed endpoint experiments using only Python standard library."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
summary = {}
for name in ['control', 'native', 'refine', 'validate']:
    path = ROOT / f'endpoint_{name}_stats.json'
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    result = {'complete': data.get('complete', False), 'methods': {}}
    for method in sorted({r['method'] for r in data['rows']}):
        rows = [r for r in data['rows'] if r['method'] == method]
        metrics = {}
        for key in ['target_rel_error', 'outside_leak_rel', 'achieved_target_gain', 'pixel_target_mse', 'pixel_outside_max_error', 'seconds', 'feedback_seconds']:
            values = [r[key] for r in rows if key in r]
            if values:
                metrics[key] = {'mean': statistics.mean(values), 'n': len(values)}
        result['methods'][method] = metrics
    if name == 'validate':
        paired = []
        for seed in data['protocol']['seeds']:
            rows = {r['method']: r for r in data['rows'] if r['seed'] == seed}
            if not {'hard_direct', 'fixed_feedback', 'safeguarded_feedback'} <= rows.keys():
                continue
            base, fixed, safe = [rows[k] for k in ['hard_direct', 'fixed_feedback', 'safeguarded_feedback']]
            paired.append({'seed': seed, 'curve': base['curve'], 'direct': base['target_rel_error'], 'fixed': fixed['target_rel_error'], 'safe': safe['target_rel_error'], 'safe_error_reduction_pct': 100 * (1 - safe['target_rel_error'] / base['target_rel_error']), 'safe_pixel_reduction_pct': 100 * (1 - safe['pixel_target_mse'] / base['pixel_target_mse']), 'accepted_updates': sum(t['accepted'] for t in safe['trials'])})
        result['paired'] = paired
        if paired:
            result['wins_vs_direct'] = sum(p['safe'] < p['direct'] for p in paired)
            result['mean_paired_error_reduction_pct'] = statistics.mean(p['safe_error_reduction_pct'] for p in paired)
            result['ratio_of_mean_error_reduction_pct'] = 100 * (1 - statistics.mean(p['safe'] for p in paired) / statistics.mean(p['direct'] for p in paired))
            differences = [p['direct'] - p['safe'] for p in paired]
            n = len(differences)
            mean = statistics.mean(differences)
            result['paired_absolute_improvement_mean'] = mean
            if n == 4:
                half = 3.182446 * statistics.stdev(differences) / n ** 0.5
                result['exploratory_t95_interval_n4'] = [mean - half, mean + half]
                result['interval_warning'] = 'Only four seed-level pairs in a synthetic task; no broad statistical significance claim.'
    summary[name] = result
(ROOT / 'endpoint_summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
