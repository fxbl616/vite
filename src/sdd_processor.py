import os

import torch

from .processor import processor as BaseProcessor
from .sdd_dataloader import SDDTrajectoryDataloader
from .star import STAR


class processor(BaseProcessor):
    """Processor for SDD pickle data.

    The train/test loop is inherited from the ETH/UCY processor because SDD
    scenes already contain only target agents and use the same STAR-style
    7-tuple input after adaptation.
    """

    def __init__(self, args):
        self.args = args

        self.dataloader = SDDTrajectoryDataloader(args)
        self.net = STAR(args)

        self.set_optimizer()

        self.device = torch.device("cuda" if self.args.using_cuda and torch.cuda.is_available() else "cpu")
        self.args.using_cuda = self.device.type == "cuda"
        self.net = self.net.to(self.device)

        if not os.path.isdir(self.args.model_dir):
            os.makedirs(self.args.model_dir)

        self.net_file = open(os.path.join(self.args.model_dir, "net.txt"), "a+")
        self.net_file.write(str(self.net))
        self.net_file.close()
        self.log_file_curve = open(os.path.join(self.args.model_dir, "log_curve.txt"), "a+")

        self.best_ade = 100
        self.best_fde = 100
        self.best_epoch = -1

    def model_run_args(self):
        args = super().model_run_args()
        args.update({
            "dataset": "sdd",
            "sdd_data_root": getattr(self.args, "sdd_data_root", "./data/stanford"),
            "sdd_unit_scale": float(getattr(self.args, "sdd_unit_scale", 1.0)),
            "sdd_neighbor_mode": getattr(self.args, "sdd_neighbor_mode", "full"),
            "sdd_neighbor_thred": float(getattr(self.args, "sdd_neighbor_thred", 10000.0)),
        })
        return args
