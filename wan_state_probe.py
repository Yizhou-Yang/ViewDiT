#!/usr/bin/env python3
"""Causal decoder-state intervention audit. Not a semantic editing system."""
import json
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from wan_vae_official import WanVAE

OUT=Path('/root/viewdit/results/video/wan_state_probe')
WEIGHT='/root/viewdit/weights/WanVAE/Wan2.1_VAE.pth'

def clone_cache(cache):
    return [v.clone() if torch.is_tensor(v) else v for v in cache]

def blend_cache(current, reference, fraction):
    return [(1-fraction)*v+fraction*r if torch.is_tensor(v) and torch.is_tensor(r) else r for v,r in zip(current,reference)]

@torch.no_grad()
def stream(vae,z,references=None,reset_at=None,fraction=1.,partial=False):
    m=vae.model;m.clear_cache()
    z=z.unsqueeze(0)/vae.scale[1].view(1,16,1,1,1)+vae.scale[0].view(1,16,1,1,1)
    x=m.conv2(z);outs=[];cache=[]
    for i in range(x.shape[2]):
        if references is not None and reset_at==i:
            if partial:
                m._feat_map[0]=references[i][0].clone()
            else:m._feat_map=blend_cache(m._feat_map,references[i],fraction)
        cache.append(clone_cache(m._feat_map))
        m._conv_idx=[0]
        outs.append(m.decoder(x[:,:,i:i+1],feat_cache=m._feat_map,feat_idx=m._conv_idx))
    y=torch.cat(outs,2)[0].float().clamp(-1,1)
    m.clear_cache()
    return y,cache

@torch.no_grad()
def main():
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    vae=WanVAE(vae_pth=WEIGHT,dtype=torch.float32,device='cuda')
    report={'complete':False,'protocol':{'seeds':[40,41,42,43],'model':'official Wan2.1 VAE only','dtype':'FP32','frames':17,'resolution':128,'edit_latent_index':2,'nominal_edited_frames':[5,6,7,8],'protected_suffix_start':9,'source':'previous Latte generated clips, final frame repeated to get 17; not real-video semantic benchmark','scale':0.4,'warning':'full cache reset can be equivalent to pixel splice and is not itself a novel contribution'},'rows':[]}
    def save(row=None):
        if row is not None:report['rows'].append(row);print('ROW',json.dumps(row),flush=True)
        (OUT/'stats.json').write_text(json.dumps(report,indent=2))
    for seed in [40,41,42,43]:
        payload=torch.load(f'/root/viewdit/results/video/endpoint_refine/seed{seed}.pt',map_location='cpu',weights_only=True)
        frames=payload['videos']['base'];frames=torch.cat([frames,frames[-1:]],0)
        frames=F.interpolate(frames,size=(128,128),mode='bilinear',align_corners=False)
        x=frames.permute(1,0,2,3).cuda()
        torch.cuda.reset_peak_memory_stats();tic=time.time()
        z=vae.model.encode(x.unsqueeze(0),vae.scale)[0].float()
        base,cache=stream(vae,z)
        official=vae.model.decode(z.unsqueeze(0),vae.scale)[0].float().clamp(-1,1)
        replay=float((base-official).abs().max());assert replay==0
        assert torch.isfinite(base).all()
        field=(torch.roll(z,1,-1)-torch.roll(z,-1,-1))*.5
        field=field/(field.square().mean().sqrt()+1e-8)*z.square().mean().sqrt()
        impulse=[]
        for j in range(z.shape[1]):
            zz=z.clone();zz[:,j]+=.1*field[:,j]
            yy,_=stream(vae,zz)
            impulse.append((yy-base).square().mean((0,2,3)).sqrt().cpu().tolist())
        zedit=z.clone();zedit[:,2]+=.4*field[:,2]
        edited,_=stream(vae,zedit)
        vids={'source':x.cpu(),'reconstruction':base.cpu(),'latent_only':edited.cpu()}
        editnorm=(edited[:,5:9]-base[:,5:9]).norm().clamp(min=1e-8)
        def record(name,y):
            d=y-base
            row={'seed':seed,'method':name,'prefix_max':float(d[:,:5].abs().max()),'suffix_max':float(d[:,9:].abs().max()),'suffix_rms':float(d[:,9:].square().mean().sqrt()),'suffix_leak_rel':float(d[:,9:].norm()/editnorm),'edit_preserved_rel_error':float((y[:,5:9]-edited[:,5:9]).norm()/editnorm),'per_frame_rms':d.square().mean((0,2,3)).sqrt().cpu().tolist(),'boundary_residual_jump':float((d[:,9]-d[:,8]).square().mean().sqrt()),'suffix_vs_pixel_splice_max':float((y[:,9:]-base[:,9:]).abs().max())}
            vids[name]=y.cpu();save(row)
        record('latent_only',edited)
        reset,_=stream(vae,zedit,cache,3,1.)
        record('full_state_reset',reset)
        partial,_=stream(vae,zedit,cache,3,1.,True)
        record('first_layer_reset',partial)
        for rho in [.25,.5,.75]:
            yy,_=stream(vae,zedit,cache,3,rho);record(f'state_blend_{rho}',yy)
        splice=edited.clone();splice[:,9:]=base[:,9:];record('pixel_splice',splice)
        print('SPLICE_EQUIVALENCE',seed,float((reset-splice).abs().max()),flush=True)
        report.setdefault('clips',[]).append({'seed':seed,'latent_shape':list(z.shape),'stream_identity_max':replay,'impulse_rms':impulse,'full_state_vs_pixel_splice_max':float((reset-splice).abs().max()),'seconds':time.time()-tic,'peak_memory_allocated':torch.cuda.max_memory_allocated(),'reconstruction_mse':float((base-x).square().mean())});save()
        torch.save({'videos':vids,'z':z.cpu()},OUT/f'seed{seed}.pt')
        keys=['reconstruction','latent_only','first_layer_reset','state_blend_0.5','full_state_reset']
        fs=[]
        for i in range(17):
            strip=torch.cat([(vids[k][:,i]+1)/2 for k in keys],dim=2)
            arr=(strip.clamp(0,1).permute(1,2,0).numpy()*255).astype('uint8')
            canvas=Image.new('RGB',(640,154),'white');canvas.paste(Image.fromarray(arr),(0,26));draw=ImageDraw.Draw(canvas)
            for j,k in enumerate(keys):draw.text((j*128+2,4),k[:19],fill='black')
            fs.append(canvas)
        fs[0].save(OUT/f'seed{seed}.gif',save_all=True,append_images=fs[1:],duration=125,loop=0)
        panel=Image.new('RGB',(640,616),'white')
        for j,i in enumerate([7,8,9,10]):panel.paste(fs[i],(0,j*154))
        panel.save(OUT/f'seed{seed}.png')
    report['complete']=True;save();print('COMPLETE',flush=True)

if __name__=='__main__':main()
