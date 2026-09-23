#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import torch
from PIL import Image, ImageDraw
from interaction_gpu_pilot import Editor, scene, evaluate, means


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--test-offset', type=int, default=200000)
    args = p.parse_args()
    torch.set_num_threads(2)
    info = json.loads((args.root / 'stats.json').read_text())
    assert info['complete']
    n = info['protocol']['test_scenes']
    data = [scene(args.test_offset + i) for i in range(n)]
    x, y = torch.stack([d[0] for d in data]), torch.stack([d[1] for d in data])
    del data
    class Identity(torch.nn.Module):
        def forward(self, z):
            return z[:, None, :3].expand(-1, 3, -1, -1, -1, -1)
    result = {'identity': means(evaluate(Identity(), x, y)), 'trained_head_mapping': {}}
    signal = y[:, 2] - y[:, 0] - y[:, 1] + x[:, :3]
    present = signal.abs().flatten(1).amax(1) > .005
    result['interaction_present_scenes'] = int(present.sum())
    result['interaction_subset_descriptive_only'] = {}
    for kind in ['direct', 'interaction']:
        rows = [row for seed in [11, 29, 47] for i, row in enumerate(info['selected'][f'{seed}_{kind}']['per_scene']) if bool(present[i])]
        result['interaction_subset_descriptive_only'][kind] = means(rows)
    for seed in [11, 29, 47]:
        structured = Editor('interaction').cuda().eval()
        structured.load_state_dict(torch.load(args.root / f'{seed}_interaction.pt', weights_only=True, map_location='cuda'))
        direct = Editor('direct').cuda().eval()
        state = {k: v.clone() for k, v in structured.state_dict().items()}
        for key in ['head.weight', 'head.bias']:
            state[key][6:9] += state[key][0:3] + state[key][3:6]
        direct.load_state_dict(state)
        maximum = 0.
        with torch.no_grad():
            for i in range(n):
                maximum = max(maximum, float((direct(x[i:i+1]) - structured(x[i:i+1])).abs().max()))
        result['trained_head_mapping'][str(seed)] = {'maximum_output_difference': maximum, 'scope': 'same backbone, linear output head, binary operations, no nonlinear branch restrictions'}
        assert maximum < 1e-5
    e = torch.load(args.root / 'test_examples.pt', weights_only=True)
    pd = torch.load(args.root / '11_direct_predictions.pt', weights_only=True)
    ps = torch.load(args.root / '11_interaction_predictions.pt', weights_only=True)
    canvas = Image.new('RGB', (6 * 192, 4 * 164), 'white')
    draw = ImageDraw.Draw(canvas)
    labels = ['source', 'GT remove A', 'direct remove A', 'interaction remove A', 'GT remove B', 'GT remove both']
    for i in range(4):
        tiles = [e['x'][i, :3, 2], e['y'][i, 0, :, 2], pd[i, 0, :, 2], ps[i, 0, :, 2], e['y'][i, 1, :, 2], e['y'][i, 2, :, 2]]
        for j, tile in enumerate(tiles):
            a = (tile.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype('uint8')
            canvas.paste(Image.fromarray(a).resize((192, 144)), (j * 192, i * 164 + 20))
            draw.text((j * 192 + 2, i * 164 + 3), labels[j], fill='black')
    canvas.save(args.root / 'contact_sheet.png')
    (args.root / 'inspection.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
