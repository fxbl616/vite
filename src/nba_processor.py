import os
import time

import torch
import torch.nn as nn
from tqdm import tqdm

from .nba_dataloader import NBATrajectoryDataloader
from .processor import processor as BaseProcessor
from .star import STAR
from .utils import L2forTestK, getFutureLossMask, maskedBestOfKLoss, trajectoryDiversityLoss


class processor(BaseProcessor):
    def __init__(self, args):
        self.args = args

        self.dataloader = NBATrajectoryDataloader(args)
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
            "dataset": "nba",
            "nba_format": getattr(self.args, "nba_format", "mart_npy"),
            "nba_neighbor_mode": getattr(self.args, "nba_neighbor_mode", "full"),
            "nba_target_groups": getattr(self.args, "nba_target_groups", "PLAYER"),
            "nba_context_groups": getattr(self.args, "nba_context_groups", "PLAYER,BALL"),
            "nba_target_indices": getattr(self.args, "nba_target_indices", None),
            "nba_unit_scale": getattr(self.args, "nba_unit_scale", None),
        })
        return args

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.learning_rate)
        self.criterion = nn.MSELoss(reduction="none")

    def _to_device_tuple(self, inputs):
        return tuple(torch.as_tensor(item, dtype=torch.float32, device=self.device) for item in inputs)

    def _masked_future_loss(self, seq_list, target_mask):
        lossmask, num = getFutureLossMask(
            seq_list,
            self.args.obs_length,
            self.args.pred_length,
            using_cuda=self.args.using_cuda,
        )
        lossmask = lossmask * target_mask.view(1, -1).type_as(lossmask)
        return lossmask, num

    def train_epoch(self, epoch):
        self.dataloader.reset_batch_pointer(set="train", valid=False)
        loss_epoch = 0

        for batch in range(self.dataloader.trainbatchnums):
            start = time.time()
            inputs, target_mask, batch_id = self.dataloader.get_train_batch(batch)
            inputs = self._to_device_tuple(inputs)
            target_mask = torch.as_tensor(target_mask, dtype=torch.float32, device=self.device)

            batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs
            inputs_forward = (
                batch_abs[:-1],
                batch_norm[:-1],
                shift_value[:-1],
                seq_list[:-1],
                nei_list[:-1],
                nei_num[:-1],
                batch_pednum,
            )

            self.net.zero_grad()
            outputs, aux = self.net.forward(inputs_forward, iftest=False)

            lossmask, num = self._masked_future_loss(seq_list, target_mask)
            target = batch_norm[self.args.obs_length:self.args.obs_length + self.args.pred_length, :, :2]
            traj_loss, best_ade, best_fde = maskedBestOfKLoss(
                outputs,
                target,
                lossmask,
                fde_weight=self.args.fde_weight,
            )
            diversity_loss = trajectoryDiversityLoss(
                outputs,
                lossmask,
                margin=self.args.diversity_margin,
            )
            loss = (
                traj_loss
                + self.args.moe_weight * aux["router_loss"]
                + self.args.diversity_weight * diversity_loss
            )
            loss_epoch += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip)
            self.optimizer.step()

            end = time.time()
            if batch % self.args.show_step == 0 and self.args.ifshow_detail:
                print(
                    "nba-train-{}/{} (epoch {}), train_loss = {:.5f}, time/batch = {:.5f}".format(
                        batch,
                        self.dataloader.trainbatchnums,
                        epoch,
                        loss.item(),
                        end - start,
                    )
                )

        return loss_epoch / max(self.dataloader.trainbatchnums, 1)

    @torch.no_grad()
    def test_epoch(self):
        self.dataloader.reset_batch_pointer(set="test")
        error_epoch, final_error_epoch = 0, 0
        error_cnt_epoch, final_error_cnt_epoch = 1e-5, 1e-5

        for batch in tqdm(range(self.dataloader.testbatchnums)):
            inputs, target_mask, batch_id = self.dataloader.get_test_batch(batch)
            inputs = self._to_device_tuple(inputs)
            target_mask = torch.as_tensor(target_mask, dtype=torch.float32, device=self.device)

            batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs
            inputs_forward = (
                batch_abs[:-1],
                batch_norm[:-1],
                shift_value[:-1],
                seq_list[:-1],
                nei_list[:-1],
                nei_num[:-1],
                batch_pednum,
            )

            outputs_infer, aux = self.net.forward(inputs_forward, iftest=True)
            self.net.zero_grad()

            lossmask, num = self._masked_future_loss(seq_list, target_mask)
            target = batch_norm[self.args.obs_length:self.args.obs_length + self.args.pred_length, :, :2]
            error, error_cnt, final_error, final_error_cnt = L2forTestK(outputs_infer, target, lossmask)

            error_epoch += error
            error_cnt_epoch += error_cnt
            final_error_epoch += final_error
            final_error_cnt_epoch += final_error_cnt

        return error_epoch / error_cnt_epoch, final_error_epoch / final_error_cnt_epoch
