#!/usr/bin/env python3
"""Summarize complete V3 evidence; label queued semantic work explicitly."""
import json
import statistics
import random
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parent


def interval(values):
    rng=random.Random(20260921)
    samples=sorted(sum(rng.choice(values) for _ in values)/len(values) for _ in range(10000))
    return [samples[249],samples[9749]]


def main():
    report={'groups':{},'warning':'Bootstrap by content, not frames; small sample, diagnostic differences, not overall perceptual quality significance.'}
    for group in ['cog_boundary_moment_transport','cog_online_boundary_norm','cog_transport_validation','cog_context_capsule','cog_context_capsule480','cog_capsule_compositor','cog_exact_memory','cog_semantic_smoke','cog_semantic_bear49','cog_semantic_capsule_restore','cog_semantic_bear49_strong']:
        path=ROOT/f'{group}_stats.json'
        if not path.exists():
            report['groups'][group]={'complete':False,'status':'not downloaded or not produced'}
            continue
        d=json.loads(path.read_text());report['groups'][group]={'complete':d['complete'],'rows':len(d['rows'])}
        if not d['complete']:continue
        if group=='cog_transport_validation':
            pairs=defaultdict(dict)
            for row in d['rows']:pairs[(row['clip'],row['edit'])][row['method']]=row
            splits={}
            for edit in ['geometric1','encoded_brightness']:
                values=[v for (c,e),v in pairs.items() if e==edit]
                diff=[v['reference']['edit_relative_rmse']-v['dual']['edit_relative_rmse'] for v in values]
                splits[edit]={'contents':len(values),'reference_mean':statistics.mean(v['reference']['edit_relative_rmse'] for v in values),'dual_mean':statistics.mean(v['dual']['edit_relative_rmse'] for v in values),'paired_delta_mean':statistics.mean(diff),'paired_bootstrap95':interval(diff),'dual_better':sum(x>0 for x in diff),'max_prefix_vs_noop':max(v['dual']['prefix_max_vs_noop'] for v in values),'max_prefix_vs_standard':max(v['dual']['prefix_max_vs_standard'] for v in values),'dual_min_amplitude':min(v['dual']['edit_amplitude_ratio'] for v in values),'dual_max_amplitude':max(v['dual']['edit_amplitude_ratio'] for v in values)}
            report['validation']=splits
        elif group=='cog_context_capsule':
            methods=defaultdict(list)
            for row in d['rows']:methods[row['method']].append(row)
            report['capsule']={k:{'payload_bytes':statistics.mean(r['payload_bytes'] for r in v),'zlib_bytes_mean':statistics.mean(r['zlib_payload_bytes'] for r in v),'max_prefix_error':max(r['prefix_max'] for r in v),'mean_edit_relative_rmse':statistics.mean(r['edit_relative_rmse'] for r in v)} for k,v in methods.items()}
        elif group.startswith('cog_semantic_'):
            report[group]=d['rows']
    (ROOT/'v3_research_summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
