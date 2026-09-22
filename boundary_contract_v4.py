#!/usr/bin/env python3
"""Audited CogVideoX boundary contract, not a generic graph certificate.

Independent implementation of reference GroupNorm replay (known prior art).
New experimental scope: cut-aware state selection, fail-closed reuse guards,
zero-state chunk boundaries, and protected-render replay with exact composition.
Run in the existing V100 diffusers0.31.0 environment. No training/downloads.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
import diffusers
from diffusers import AutoencoderKLCogVideoX


def digest_tensor(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def identity(vae, z):
    if diffusers.__version__ != '0.31.0':
        raise ValueError('Unreviewed diffusers version')
    if vae.use_tiling or vae.num_latent_frames_batch_size != 2:
        raise ValueError('Only untiled two-latent batching was reviewed')
    if z.dtype != torch.float32 or z.shape[0] != 1 or z.shape[2] % 2 != 1:
        raise ValueError('Only FP32 batch1 odd latent lengths supported')
    return {'version':diffusers.__version__, 'shape':list(z.shape),
            'batch':2, 'dtype':str(z.dtype),
            'execution':hashlib.sha256(inspect.getsource(type(vae)._decode).encode()).hexdigest()}


def weights_digest(vae):
    h = hashlib.sha256()
    for name, tensor in vae.state_dict().items():
        h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class Replay:
    def __init__(self, decoder, call_id, moments=None):
        self.call_id = call_id;self.moments = {} if moments is None else moments
        self.capture = moments is None;self.count = {};self.handles = [];self.seen = set()
        for name, module in decoder.named_modules():
            if isinstance(module, torch.nn.GroupNorm):
                self.handles.append(module.register_forward_hook(self.hook(name)))

    def hook(self, name):
        def apply(m, args, out):
            index = self.count.get(name, 0);self.count[name] = index + 1
            if index != self.call_id:
                return out
            x = args[0]
            if x.ndim != 5:
                raise ValueError('Unexpected normalization rank')
            self.seen.add(name)
            flat = x.reshape(x.shape[0], m.num_groups, -1)
            if self.capture:
                self.moments[name] = (flat.mean(-1, keepdim=True).detach().clone(),
                                     (flat.var(-1, unbiased=False, keepdim=True)+m.eps).rsqrt().detach().clone())
                return out
            mu, rs = self.moments[name]
            y = ((flat-mu)*rs).reshape_as(x)
            if m.weight is not None:y = y*m.weight.view(1,-1,1,1,1)
            if m.bias is not None:y = y+m.bias.view(1,-1,1,1,1)
            return y
        return apply

    def close(self):
        for handle in self.handles:handle.remove()


class BoundaryContract:
    """State is tied to a reference prefix, weights and execution signature.

Weight hash is supplied by the caller once per immutable model session.
It must be recomputed after any model replacement or mutation.
"""
    @classmethod
    def commit(cls, vae, z, start_latent, model_hash):
        self = cls();self.signature = identity(vae,z);self.model_hash = model_hash
        if not 3 <= start_latent < z.shape[2]:
            raise ValueError('Cut must satisfy 3 <= b < latent length')
        self.b = start_latent;self.p = 1+4*(start_latent-1)
        self.prefix_hash = digest_tensor(z[:,:,:start_latent])
        self.call_id = None if start_latent % 2 else (start_latent-2)//2
        capture = Replay(vae.decoder,self.call_id)
        try:
            reference = vae.decode(z).sample
        finally:capture.close()
        self.moments = capture.moments
        self.payload_bytes = sum(t.numel()*t.element_size() for values in self.moments.values() for t in values)
        return self, reference

    def render(self, vae, ze, model_hash, raw=None):
        if identity(vae,ze) != self.signature or model_hash != self.model_hash:
            raise ValueError('Stale model/execution contract')
        if digest_tensor(ze[:,:,:self.b]) != self.prefix_hash:
            raise ValueError('Protected latent ancestors changed; recommit required')
        if raw is None:raw = vae.decode(ze).sample
        if self.call_id is None:
            return raw.clone()
        replay = Replay(vae.decoder,self.call_id,self.moments)
        try:
            restored = vae.decode(ze).sample
            if replay.seen != set(self.moments):
                raise ValueError('Normalization layout changed')
        finally:replay.close()
        result = raw.clone();result[:,:,:self.p] = restored[:,:,:self.p]
        return result


def math_audit():
    torch.manual_seed(20260922)
    x = torch.randn(12,dtype=torch.float64,device='cuda',requires_grad=True)
    norm = lambda v: (v-v.mean())/(v.var(unbiased=False)+1e-5).sqrt()
    jac = torch.autograd.functional.jacobian(norm,x)
    sub = jac[:5,5:]
    mu = x.mean();r = (x.var(unbiased=False)+1e-5).rsqrt();n = x.numel()
    analytic = -r/n-torch.outer(x[:5]-mu,x[5:]-mu)*r**3/n
    sv = torch.linalg.svdvals(sub)
    a = torch.tensor([-1.,-1.,1.,1.],device='cuda',dtype=torch.float64)
    b = torch.tensor([-2.**.5,0.,0.,2.**.5],device='cuda',dtype=torch.float64)
    return {'cross_jacobian_formula_max_error':float((sub-analytic).abs().max()),
            'cross_jacobian_numerical_rank':int((sv>1e-10).sum()),
            'equal_input_mean':bool(torch.allclose(a.mean(),b.mean())),
            'equal_input_variance':bool(torch.allclose(a.var(unbiased=False),b.var(unbiased=False))),
            'relu_mean_a':float(a.relu().mean()),'relu_mean_b':float(b.relu().mean()),
            'warning':'Toy derivative verification, not a new theorem or video benchmark'}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path('/root/viewdit/results/video/boundary_contract_v4'))
    args = parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    vae = AutoencoderKLCogVideoX.from_pretrained('/root/viewdit/weights/CogVAE',local_files_only=True,torch_dtype=torch.float32).cuda().eval().requires_grad_(False)
    model_hash = weights_digest(vae)
    report = {'complete':False,'protocol':{'clips':['car-shadow','cows','elephant','goat'],
              'frames':33,'size':[128,224],'cuts':[3,4,5,6,7,8],'trials':2,
              'seed':20260922,'amplitude':.4,'dtype':'FP32','device':torch.cuda.get_device_name(),
              'warning':'Reused V3 contents; mechanism development only. Gaussian suffix stress, not semantic editing. Full decode restoration, no speed claim. No general graph compiler or multi-commit claim.'},
              'rows':[],'guards':{},'model_hash':model_hash}
    def save():
        tmp=args.out/'stats.tmp';tmp.write_text(json.dumps(report,indent=2));tmp.replace(args.out/'stats.json')
    with torch.enable_grad():report['math_audit']=math_audit()
    save()
    generator=torch.Generator(device='cuda').manual_seed(20260922)
    for clip in report['protocol']['clips']:
        root=Path('/root/viewdit/data/v3_validation_data/JPEGImages')/clip
        files=sorted(p for p in root.glob('*.jpg') if not p.name.startswith('.'))[:33]
        if len(files)!=33:raise ValueError('Missing frames')
        x=torch.from_numpy(np.stack([np.asarray(Image.open(p).convert('RGB').resize((224,128)),dtype=np.float32)/127.5-1 for p in files])).permute(3,0,1,2).unsqueeze(0).cuda()
        z=vae.encode(x).latent_dist.mode();del x
        for b in report['protocol']['cuts']:
            contract,base=BoundaryContract.commit(vae,z,b,model_hash)
            if clip==report['protocol']['clips'][0] and b==4:
                stale=z.clone();stale[:,:,0,0,0]+=.01
                try:contract.render(vae,stale,model_hash)
                except ValueError:report['guards']['changed_prefix_rejected']=True
                else:raise AssertionError('Stale prefix was accepted')
                try:contract.render(vae,z,model_hash+'invalid')
                except ValueError:report['guards']['changed_model_rejected']=True
                else:raise AssertionError('Stale model was accepted')
                old=vae.num_latent_frames_batch_size;vae.num_latent_frames_batch_size=1
                try:
                    try:contract.render(vae,z,model_hash)
                    except ValueError:report['guards']['changed_batch_rejected']=True
                    else:raise AssertionError('Changed batch accepted')
                finally:vae.num_latent_frames_batch_size=old
            for trial in range(2):
                ze=z.clone();ze[:,:,b:]+=.4*z.square().mean().sqrt()*torch.randn(ze[:,:,b:].shape,device='cuda',generator=generator)
                raw=vae.decode(ze).sample
                torch.cuda.synchronize();tic=time.perf_counter()
                y=contract.render(vae,ze,model_hash,raw)
                torch.cuda.synchronize();elapsed=time.perf_counter()-tic
                p=contract.p
                row={'clip':clip,'cut_latent':b,'protected_frames':p,'trial':trial,
                     'selected_call':contract.call_id,'state_payload_bytes':contract.payload_bytes,
                     'raw_prefix_max':float((raw[:,:,:p]-base[:,:,:p]).abs().max()),
                     'restored_prefix_max':float((y[:,:,:p]-base[:,:,:p]).abs().max()),
                     'edited_max':float((y[:,:,p:]-raw[:,:,p:]).abs().max()),
                     'restore_seconds_excluding_raw':elapsed}
                report['rows'].append(row);save();print('ROW',json.dumps(row),flush=True)
    report['complete']=True;save();print('COMPLETE',flush=True)


if __name__=='__main__':main()
