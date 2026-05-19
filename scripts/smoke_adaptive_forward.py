"""Smoke-test adaptive ablation forward shapes.

Run from repo root:
    conda run -n star python scripts/smoke_adaptive_forward.py
"""

from argparse import Namespace
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.star import STAR


def main():
    base = dict(
        obs_length=8,
        pred_length=12,
        sample_num=3,
        model_dim=32,
        num_heads=4,
        rt_layers=1,
        num_virtual_nodes=2,
        spatial_prior_mix=0.6,
        spatial_sigma=2.0,
        aip_top_p=0.75,
        router_top_p=0.5,
        dropout=0.0,
        motion_gate_bias=1.0,
        spatial_gate_bias=-1.0,
    )

    seq_length = 20
    num_ped = 5
    nodes_abs = torch.randn(seq_length, num_ped, 2)
    nodes_norm = nodes_abs - nodes_abs[7:8]
    shift_value = torch.zeros_like(nodes_abs)
    seq_list = torch.ones(seq_length, num_ped)
    nei_lists = torch.ones(seq_length, num_ped, num_ped)
    nei_lists[:, torch.arange(num_ped), torch.arange(num_ped)] = 0
    nei_num = nei_lists.sum(-1)
    batch_pednum = torch.tensor([3, 2])
    inputs = (nodes_abs, nodes_norm, shift_value, seq_list, nei_lists, nei_num, batch_pednum)

    for mode in ("adaptive", "adaptive_no_motion", "adaptive_no_spatial", "adaptive_motion_only"):
        args = Namespace(**base, ablation=mode)
        model = STAR(args).eval()
        with torch.no_grad():
            pred, aux = model(inputs)
        assert tuple(pred.shape) == (3, 12, num_ped, 2), (mode, pred.shape)
        assert torch.isfinite(pred).all(), mode
        print(
            mode,
            tuple(pred.shape),
            "motion_gate={:.4f}".format(float(aux["motion_gate_mean"])),
            "spatial_gate={:.4f}".format(float(aux["spatial_gate_mean"])),
        )


if __name__ == "__main__":
    main()
