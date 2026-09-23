#!/usr/bin/env python3
"""First-order block-residual extrapolation baseline, not official TaylorSeer."""
import argparse
import gc
import json
import time
import types
from pathlib import Path
import torch
from diffusers import CogVideoXPipeline, CogVideoXTransformer3DModel, AutoencoderKLCogVideoX, CogVideoXDDIMScheduler
from dit_compute_allocation_pilot import CASES, GROUPS, DevicePipeline

class Forecast:
    def __init__(self, model):
        self.model = model
        self.original = [b.forward for b in model.transformer_blocks]
        self.handle = model.register_forward_pre_hook(self.increment)
        self.step = -1
        for i,b in enumerate(model.transformer_blocks):
            def forward(block,hidden_states,encoder_hidden_states,temb,image_rotary_emb=None,idx=i):
                if idx in self.selected and self.step in range(5,26,2):
                    latest, previous = self.hist[idx][-1],self.hist[idx][-2]
                    ratio = (self.step-latest[0])/(latest[0]-previous[0])
                    dh = latest[1] + ratio*(latest[1]-previous[1])
                    de = latest[2] + ratio*(latest[2]-previous[2])
                    self.reused += 1
                    return hidden_states+dh,encoder_hidden_states+de
                output=self.original[idx](hidden_states,encoder_hidden_states,temb,image_rotary_emb)
                self.calls += 1
                if idx in self.selected:
                    h=self.hist.setdefault(idx,[])
                    h.append((self.step,(output[0]-hidden_states).detach(),(output[1]-encoder_hidden_states).detach()))
                    if len(h)>2:
                        h.pop(0)
                return output
            b.forward=types.MethodType(forward,b)
    def increment(self,*args):
        self.step += 1
    def reset(self,groups):
        self.selected=set(i for g in groups for i in GROUPS[g])
        self.hist={}
        self.step=-1
        self.calls=self.reused=0
    def close(self):
        self.handle.remove()
        for b,f in zip(self.model.transformer_blocks,self.original):
            b.forward=f
        self.hist.clear()

@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--model',type=Path,default=Path('/root/viewdit/weights/CogVideoX-2b'))
    a=p.parse_args()
    assert json.loads((a.root/'stats.json').read_text())['complete']
    out=a.root/'forecast_baseline'
    out.mkdir(exist_ok=False)
    analysis=json.loads((a.root/'analysis.json').read_text())
    chosen=analysis['chosen_diagonal']
    embeds=torch.load(a.root/'embeddings.pt',weights_only=True,map_location='cpu')
    torch.set_num_threads(2)
    vae=AutoencoderKLCogVideoX.from_pretrained(a.model/'vae',torch_dtype=torch.float32,local_files_only=True).eval()
    dit=CogVideoXTransformer3DModel.from_pretrained(a.model/'transformer',torch_dtype=torch.float16,local_files_only=True).cuda().eval().requires_grad_(False)
    sched=CogVideoXDDIMScheduler.from_pretrained(a.model/'scheduler',local_files_only=True)
    pipe=DevicePipeline(vae=vae,transformer=dit,scheduler=sched,tokenizer=None,text_encoder=None)
    pipe.set_progress_bar_config(disable=True)
    hook=Forecast(dit)
    report={'complete':False,'description':'first order residual extrapolation, standard baseline not official TaylorSeer','chosen_from_calibration':chosen,'runs':[]}
    (out/'stats.json').write_text(json.dumps(report,indent=2))
    for case,prompt,seed in CASES:
        initial=torch.load(a.root/f'{case}_initial.pt',weights_only=True,map_location='cuda')
        reference=torch.load(a.root/f'{case}_outputs.pt',weights_only=True,map_location='cpu')['full'].float()
        outputs={}
        for name,groups in [('linear_selected',chosen),('linear_all4',[0,1,2,3])]:
            hook.reset(groups)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start=time.perf_counter()
            z=pipe(latents=initial.clone(),prompt_embeds=embeds[prompt].cuda(),negative_prompt_embeds=embeds[''].cuda(),height=480,width=720,num_frames=17,num_inference_steps=30,guidance_scale=6.,eta=0.,generator=torch.Generator(device='cuda').manual_seed(seed),output_type='latent').frames
            torch.cuda.synchronize()
            seconds=time.perf_counter()-start
            assert torch.isfinite(z).all()
            outputs[name]=z.cpu()
            row={'case':case,'method':name,'seconds_denoising_only':seconds,'actual_block_calls':hook.calls,'reused_block_calls':hook.reused,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'latent_mse_to_full':float((z.cpu().float()-reference).square().mean()),'finite':True}
            report['runs'].append(row)
            torch.save(outputs,out/f'{case}_outputs.pt')
            (out/'stats.json').write_text(json.dumps(report,indent=2))
            print(json.dumps(row),flush=True)
            del z
        gc.collect()
    hook.close()
    report['complete']=True
    (out/'stats.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
