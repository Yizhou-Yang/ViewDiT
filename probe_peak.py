import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch
from pathlib import Path
from diffusers import AutoencoderKLCogVideoX
from cog_boundary_moment_transport import capture
from cog_exact_memory import NoUnusedTemporalCache
SRC = Path('/root/viewdit/results/video/cog_semantic_bear49')
torch.set_num_threads(2)
d = torch.load(SRC / 'bear_round1.pt', map_location='cpu', weights_only=True)
ze = d['latent']
z = d['source_latent']
vae = AutoencoderKLCogVideoX.from_pretrained(
    '/root/viewdit/weights/CogVAE', local_files_only=True,
    torch_dtype=torch.float32).cuda().eval().requires_grad_(False)
patch = NoUnusedTemporalCache(vae)
torch.cuda.reset_peak_memory_stats()
_, ref = capture(vae, z[:, :, :5].cuda())
p1 = torch.cuda.max_memory_allocated()
del ref
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
_, ed = capture(vae, ze.cuda())
p2 = torch.cuda.max_memory_allocated()
print('PREFIX_PEAK_GB', round(p1 / 1e9, 2), 'EDIT_PEAK_GB', round(p2 / 1e9, 2), flush=True)
print('PREFIX_SHAPE', tuple(z[:, :, :5].shape), 'EDIT_SHAPE', tuple(ze.shape), flush=True)
patch.close()