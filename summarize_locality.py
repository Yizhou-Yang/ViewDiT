#!/usr/bin/env python3
import json
import statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
summary={}

def mean(rows,key):
    return statistics.mean(r[key] for r in rows)

for name in ['endpoint_stress','wan_state_probe','wan_tail_compensate','wan_real_pilot','wan_baseline_audit']:
    p=ROOT/f'{name}_stats.json'
    if not p.exists():continue
    d=json.loads(p.read_text());s={'complete':d['complete'],'n_rows':len(d['rows'])}
    if name=='endpoint_stress':
        a=[r for r in d['rows'] if r['task']=='ablation'];s['reject_only_error']=mean(a,'target_rel_error')
        b=[r for r in d['rows'] if r['task']=='bridge'];s['bridge']={m:{k:mean([r for r in b if r['method']==m],k) for k in ['core_pixel_mse','edit_residual_boundary_jump_proxy']} for m in sorted(set(r['method'] for r in b))}
    if name=='wan_state_probe':
        s['methods']={m:{k:mean([r for r in d['rows'] if r['method']==m],k) for k in ['suffix_rms','suffix_leak_rel','edit_preserved_rel_error']} for m in sorted(set(r['method'] for r in d['rows']))}
        s['full_reset_equals_splice_all']=all(c['full_state_vs_pixel_splice_max']==0 for c in d['clips'])
    if name=='wan_tail_compensate':
        s['mean_reduction_pct']=mean(d['rows'],'suffix_mse_reduction_pct');s['mean_seconds']=mean(d['rows'],'seconds')
        s['all_core_exact']=all(r['prefix_and_core_max_change']==0 for r in d['rows'])
    if name=='wan_real_pilot':
        s['groups']={}
        for kind in sorted(set(r['edit_type'] for r in d['rows'])):
            rows=[r for r in d['rows'] if r['edit_type']==kind]
            s['groups'][kind]={'n':len(rows),'initial_mse':mean(rows,'initial_suffix_mse'),'final_mse':mean(rows,'final_suffix_mse'),'hybrid_mse':mean(rows,'hybrid_reencode_suffix_mse'),'mean_paired_reduction_pct':mean(rows,'suffix_mse_reduction_pct'),'ratio_mean_reduction_pct':100*(1-mean(rows,'final_suffix_mse')/mean(rows,'initial_suffix_mse')),'wins_vs_no_compensation':sum(r['final_suffix_mse']<r['initial_suffix_mse'] for r in rows),'wins_vs_hybrid':sum(r['final_suffix_mse']<r['hybrid_reencode_suffix_mse'] for r in rows),'seconds':mean(rows,'seconds'),'all_core_exact':all(r['prefix_core_max_change']==0 for r in rows)}
        s['cases']=[{k:r[k] for k in ['clip','edit_type','initial_suffix_mse','final_suffix_mse','suffix_mse_reduction_pct','hybrid_reencode_suffix_mse','prefix_core_max_change','seconds','boundary_jump_initial','boundary_jump_final']} for r in d['rows']]
    if name=='wan_baseline_audit':
        s['methods']={}
        for m in sorted(set(r['method'] for r in d['rows'])):
            rows=[r for r in d['rows'] if r['method']==m]
            keys=['suffix_mse','prefix_core_mse','prefix_core_max','seconds','preceding_frames_vs_prior_max']
            s['methods'][m]={k:mean(rows,k) for k in keys if k in rows[0]}
        s['cases']=d['rows']
    summary[name]=s
(ROOT/'locality_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
