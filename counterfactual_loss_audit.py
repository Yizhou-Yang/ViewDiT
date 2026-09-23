#!/usr/bin/env python3
"""Audit algebraic redundancy, not video editing performance. Requires torch."""
import argparse
import json
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(20260922)
    dtype = torch.float64
    h = torch.tensor([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]], device=device, dtype=dtype) / 2
    e = torch.randn(4, 8192, device=device, dtype=dtype, requires_grad=True)
    original = e.square().sum()
    transformed = (h @ e).square().sum()
    g0 = torch.autograd.grad(original, e, retain_graph=True)[0]
    g1 = torch.autograd.grad(transformed, e, retain_graph=True)[0]
    contrast = torch.tensor([1., -1., -1., 1.], device=device, dtype=dtype)
    contrast_loss = (contrast @ e).square().sum()
    pure_common = torch.ones_like(e)
    common_penalty = (contrast @ pure_common).square().sum()
    metric = torch.eye(4, device=device, dtype=dtype) + torch.outer(contrast, contrast)
    augmented = original + contrast_loss
    explicit_quadratic = (e * (metric @ e)).sum()
    result = {
        'device': device,
        'dtype': str(dtype),
        'orthogonal_loss_relative_error': float((original - transformed).abs() / original),
        'orthogonal_gradient_max_error': float((g0 - g1).abs().max()),
        'interaction_only_common_error_penalty': float(common_penalty),
        'paired_plus_interaction_quadratic_relative_error': float((augmented - explicit_quadratic).abs() / augmented),
        'conclusions': [
            'Equal-weight complete orthonormal contrasts have exactly the paired squared loss and gradients in exact arithmetic.',
            'An added squared interaction contrast is a quadratic reweighting of joint prediction errors, not new target information.',
            'Interaction contrast alone leaves arbitrary common-mode prediction error unpenalized.',
            'This does not refute useful inductive biases, different model parameterizations, partial-label settings, sampling changes, or nonlinear losses.'
        ]
    }
    assert result['orthogonal_loss_relative_error'] < 1e-12
    assert result['orthogonal_gradient_max_error'] < 1e-12
    assert result['interaction_only_common_error_penalty'] == 0
    assert result['paired_plus_interaction_quadratic_relative_error'] < 1e-12
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
