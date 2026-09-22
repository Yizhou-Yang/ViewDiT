#!/usr/bin/env python3
"""Aggregate complete groups only; never silently summarize partial runs."""
import json
import statistics as s
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parent
NAMES = ['wan_boundary_frontier_pilot', 'wan_feasible_control', 'wan_local_rank', 'cog_locality_probe', 'wan_transition_budget', 'cog_norm_replay', 'wan_bridge_convergence_pilot', 'cog_norm_holdout', 'cog_affine_replay', 'cog_native_moment_projection']


def mean(rows, key):
    return s.mean(r[key] for r in rows)


def ranges(rows, keys):
    return {k: {'min': min(r[k] for r in rows), 'max': max(r[k] for r in rows), 'mean': mean(rows, k)} for k in keys}


def groups(rows, key):
    out = defaultdict(list)
    for r in rows:
        out[str(r[key])].append(r)
    return out


def main():
    data = {}
    for name in NAMES:
        d = json.loads((ROOT / f'{name}_stats.json').read_text())
        assert d['complete'], name
        data[name] = d['rows']
    out = {'status': {name: {'complete': True, 'rows': len(rows)} for name, rows in data.items()}, 'total_rows': sum(map(len, data.values()))}
    out['feasible'] = {k: {'n': len(v), **ranges(v, ['suffix_reduction_pct', 'relative_core_rmse', 'seconds', 'forwards']), 'reduction_ratio_of_means_pct': 100*(1-mean(v,'final_suffix_mse')/mean(v,'initial_suffix_mse'))} for k,v in groups(data['wan_feasible_control'], 'method').items()}
    out['bridge'] = {k: {'passes': sum(r['passes_budget'] for r in v), 'n': len(v), **ranges(v, ['relative_protected_rms', 'reencoded_fade_core_relative_rmse', 'reencoded_fade_protected_relative_rms'])} for k,v in groups(data['wan_transition_budget'], 'transition_frames').items()}
    out['rank'] = data['wan_local_rank']
    out['normalization_pilot'] = ranges(data['cog_norm_replay'], ['raw_earlier_max','identity_max','replay_earlier_max','edited_chunk_vs_raw_relative_rmse','edit_amplitude_ratio'])
    r = data['cog_norm_holdout']
    out['normalization_holdout'] = {'n':len(r),'passes':sum(x['passes_all'] for x in r), **ranges(r, ['raw_prefix_max','replay_prefix_max','identity_max','edit_relative_rmse_vs_raw','edit_amplitude_ratio','capture_seconds','raw_decode_seconds','replay_decode_seconds'])}
    out['affine'] = ranges(data['cog_affine_replay'], ['raw_seconds','hook_seconds','affine_seconds','capture_seconds','compile_seconds','affine_prefix_max','affine_noop_max','affine_vs_hook_max','moment_scalars','runtime_coefficient_scalars'])
    out['native_projection'] = {f'{k}:{weight}': {'n':len(rows), **ranges(rows,['prefix_mse_reduction_pct','prefix_max','edit_relative_rmse','seconds'])} for k,v in groups(data['cog_native_moment_projection'],'method').items() for weight,rows in groups(v,'edit_weight').items()}
    out['convergence'] = [{k:v for k,v in r.items() if k!='curve'} for r in data['wan_bridge_convergence_pilot']]
    out['caveats'] = ['No semantic editor or downstream DiT evaluation.', 'Four original contents are exploratory, not independent held-out data.', 'Four new contents validate the frozen replay method only; other follow-ups reusing them are not new holdouts.', 'All float values use model pixel range [-1,1].', 'Exact prefix preservation is relative to replay zero-edit output, not bitwise identical to native decoding.', 'Pixel compositing already gives exact preservation; no overall superiority to it is demonstrated.', 'Latency from native projection, bridge, and VAE decode workloads is not interchangeable.', 'Finite-difference rank is a numerical local result on two low-resolution samples, not a global impossibility theorem.']
    (ROOT / 'boundary_round_summary.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
