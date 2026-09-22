#!/usr/bin/env python3
"""Fixed four-clip real-video pilot; localized geometric/color edits, not text edits."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from wan_vae_official import WanVAE
from wan_state_probe import stream
from wan_tail_compensate import suffix

OUT=Path('/root/viewdit/results/video/wan_real_pilot')
DATA=Path('/root/viewdit/data/davis_locality_pilot/JPEGImages')
CLIPS=['bear','blackswan','car-roundabout','camel']


def main():
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    vae=WanVAE(vae_pth='/root/viewdit/weights/WanVAE/Wan2.1_VAE.pth',dtype=torch.float32,device='cuda')
    report={'complete':False,'protocol':{'clips':CLIPS,'frames':17,'size':[128,224],'iterations':24,'lr':.03,'source':'DAVIS-Edit original video frames, first 17; no outcome filtering','edit_types':['latent_geometric_impulse','pixel_color_patch_encoded'],'core':'frames 5:9, 0-based','protected_suffix':'frames 9:17','objective':'suffix MSE/initial MSE + 0.001 normalized correction MSE','output':'fixed last iterate; reusable latent, standard unmodified decoder','warning':'Four real clips, not full semantic benchmark. Core retained exactly is a causal structural property, not learned semantic quality.'},'rows':[]}
    def save(row=None):
        if row is not None:report['rows'].append(row);print('ROW',json.dumps(row),flush=True)
        (OUT/'stats.json').write_text(json.dumps(report,indent=2))
    for clip in CLIPS:
        a=[np.array(Image.open(p).convert('RGB').resize((224,128),Image.Resampling.LANCZOS),copy=True) for p in sorted((DATA/clip).glob('*.jpg'))[:17]]
        assert len(a)==17
        original=torch.from_numpy(np.stack(a)).permute(3,0,1,2).float().cuda()/127.5-1
        with torch.no_grad():
            z=vae.model.encode(original.unsqueeze(0),vae.scale)[0].float();base,_=stream(vae,z)
        for kind in report['protocol']['edit_types']:
            with torch.no_grad():
                ze=z.clone()
                if kind=='latent_geometric_impulse':
                    field=(torch.roll(z,1,-1)-torch.roll(z,-1,-1))*.5
                    field=field/(field.square().mean().sqrt()+1e-8)*z.square().mean().sqrt()
                    ze[:,2]+=.4*field[:,2]
                else:
                    altered=original.clone();altered[0,5:9,32:96,56:168]=(altered[0,5:9,32:96,56:168]+.35).clamp(-1,1)
                    encoded=vae.model.encode(altered.unsqueeze(0),vae.scale)[0].float()
                    ze[:,2]=encoded[:,2]
                edited,states=stream(vae,ze);state=states[3];future=ze[:,3:].clone()
                target=base[:,9:];initial=suffix(vae,future,state)
                replay=float((initial-edited[:,9:]).abs().max());assert replay<1e-5
                denom=(initial-target).square().mean().clamp(min=1e-12)
                splice=edited.clone();splice[:,9:]=base[:,9:]
                reencoded=vae.model.encode(splice.unsqueeze(0),vae.scale)[0].float()
                hybrid=ze.clone();hybrid[:,3:]=reencoded[:,3:]
                hybrid_img,_=stream(vae,hybrid)
            correction=torch.zeros_like(future,requires_grad=True);opt=torch.optim.Adam([correction],lr=.03)
            curve=[];tic=time.time();torch.cuda.reset_peak_memory_stats()
            for k in range(24):
                opt.zero_grad(set_to_none=True)
                prediction=suffix(vae,future+correction,state)
                mse=(prediction-target).square().mean()
                objective=mse/denom+.001*correction.square().mean()/(future.square().mean()+1e-8)
                assert torch.isfinite(objective)
                objective.backward();opt.step();curve.append(float(mse.detach()/denom))
            with torch.no_grad():
                final=ze.clone();final[:,3:]=future+correction.detach()
                result,_=stream(vae,final)
                mse=float((result[:,9:]-target).square().mean());hmse=float((hybrid_img[:,9:]-target).square().mean())
                row={'clip':clip,'edit_type':kind,'initial_suffix_mse':float(denom),'final_suffix_mse':mse,'suffix_mse_reduction_pct':100*(1-mse/float(denom)),'hybrid_reencode_suffix_mse':hmse,'hybrid_core_max_change':float((hybrid_img[:,:9]-edited[:,:9]).abs().max()),'prefix_core_max_change':float((result[:,:9]-edited[:,:9]).abs().max()),'edit_core_rms':float((edited[:,5:9]-base[:,5:9]).square().mean().sqrt()),'initial_suffix_rms':float(denom.sqrt()),'final_suffix_rms':mse**.5,'relative_correction_norm':float(correction.norm()/future.norm()),'seconds':time.time()-tic,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'relative_mse_curve':curve,'reconstruction_mse':float((base-original).square().mean()),'suffix_replay_max':replay,'boundary_jump_initial':float(((edited-base)[:,9]-(edited-base)[:,8]).square().mean().sqrt()),'boundary_jump_final':float(((result-base)[:,9]-(result-base)[:,8]).square().mean().sqrt()),'boundary_jump_splice':float(((splice-base)[:,9]-(splice-base)[:,8]).square().mean().sqrt())}
                save(row)
                videos={'base':base.cpu(),'edit_latent_only':edited.cpu(),'tail_compensated':result.cpu(),'hybrid_reencode':hybrid_img.cpu(),'pixel_splice':splice.cpu()}
                torch.save({'videos':videos,'edited_z':ze.cpu(),'compensated_z':final.cpu()},OUT/f'{clip}_{kind}.pt')
                keys=list(videos);frames=[]
                for i in range(17):
                    strip=torch.cat([(videos[k][:,i]+1)/2 for k in keys],dim=2)
                    arr=(strip.clamp(0,1).permute(1,2,0).numpy()*255).astype('uint8')
                    canvas=Image.new('RGB',(1120,156),'white');canvas.paste(Image.fromarray(arr),(0,28));draw=ImageDraw.Draw(canvas)
                    for j,k in enumerate(keys):draw.text((224*j+4,6),k,fill='black')
                    frames.append(canvas)
                frames[0].save(OUT/f'{clip}_{kind}.gif',save_all=True,append_images=frames[1:],duration=125,loop=0)
                panel=Image.new('RGB',(1120,624),'white')
                for j,i in enumerate([7,8,9,10]):panel.paste(frames[i],(0,156*j))
                panel.save(OUT/f'{clip}_{kind}.png')
    report['complete']=True;save();print('COMPLETE',flush=True)

if __name__=='__main__':main()
