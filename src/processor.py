import os
import time

import torch
import torch.nn as nn

from .star import STAR
from .utils import *

from tqdm import tqdm


class processor(object):
    def __init__(self, args):

        self.args = args

        self.dataloader = Trajectory_Dataloader(args)
        self.net = STAR(args)

        self.set_optimizer()

        self.device = torch.device("cuda" if self.args.using_cuda and torch.cuda.is_available() else "cpu")
        self.args.using_cuda = self.device.type == "cuda"
        self.net = self.net.to(self.device)

        if not os.path.isdir(self.args.model_dir):
            os.mkdir(self.args.model_dir)

        self.net_file = open(os.path.join(self.args.model_dir, 'net.txt'), 'a+')
        self.net_file.write(str(self.net))
        self.net_file.close()
        self.log_file_curve = open(os.path.join(self.args.model_dir, 'log_curve.txt'), 'a+')

        self.best_ade = 100
        self.best_fde = 100
        self.best_epoch = -1

    def model_shape_args(self):
        model_dim = int(getattr(self.args, "model_dim", 64))
        return {
            "obs_length": int(getattr(self.args, "obs_length", 8)),
            "pred_length": int(getattr(self.args, "pred_length", 12)),
            "sample_num": int(getattr(self.args, "sample_num", 20)),
            "model_dim": model_dim,
            "num_heads": int(getattr(self.args, "num_heads", 4)),
            "rt_layers": int(getattr(self.args, "rt_layers", 1)),
            "num_virtual_nodes": int(getattr(self.args, "num_virtual_nodes", 3)),
            "decoder_hidden_dim": int(getattr(self.args, "decoder_hidden_dim", model_dim * 2)),
            "ablation": str(getattr(self.args, "ablation", "full")),
        }

    def model_run_args(self):
        args = self.model_shape_args()
        args.update({
            "aip_top_p": float(getattr(self.args, "aip_top_p", 0.75)),
            "router_top_p": float(getattr(self.args, "router_top_p", 0.7)),
            "spatial_prior_mix": float(getattr(self.args, "spatial_prior_mix", 0.6)),
            "spatial_sigma": float(getattr(self.args, "spatial_sigma", 2.0)),
            "moe_weight": float(getattr(self.args, "moe_weight", 0.01)),
            "fde_weight": float(getattr(self.args, "fde_weight", 0.5)),
            "diversity_weight": float(getattr(self.args, "diversity_weight", 0.02)),
            "diversity_margin": float(getattr(self.args, "diversity_margin", 0.2)),
            "motion_gate_bias": float(getattr(self.args, "motion_gate_bias", 1.0)),
            "spatial_gate_bias": float(getattr(self.args, "spatial_gate_bias", -1.0)),
        })
        return args

    def validate_checkpoint_args(self, checkpoint):
        saved_args = checkpoint.get("model_args")
        if saved_args is None:
            print("Warning: checkpoint has no model_args metadata; loading state_dict directly.")
            return

        current_args = self.model_shape_args()
        mismatches = []
        for key, current_value in current_args.items():
            if key in saved_args and saved_args[key] != current_value:
                mismatches.append((key, saved_args[key], current_value))

        if mismatches:
            mismatch_text = ", ".join(
                ["{}: checkpoint={} current={}".format(key, old, new) for key, old, new in mismatches]
            )
            raise ValueError(
                "Checkpoint architecture mismatch. Use the same model shape args when testing/loading. "
                + mismatch_text
            )

    def _checkpoint_dir(self):
        return os.path.join(self.args.save_dir, self.args.train_model)

    def _checkpoint_path(self, load_model):
        model_dir = self._checkpoint_dir()
        load_model = str(load_model)
        if load_model.lower() in ("best", "best_fde"):
            return os.path.join(model_dir, self.args.train_model + "_best.tar")
        return os.path.join(model_dir, self.args.train_model + "_" + load_model + ".tar")

    def _remove_old_checkpoints(self, keep_path):
        model_dir = self._checkpoint_dir()
        keep_name = os.path.basename(keep_path)
        prefix = self.args.train_model + "_"
        for filename in os.listdir(model_dir):
            if not filename.startswith(prefix) or not filename.endswith(".tar"):
                continue
            if filename == keep_name:
                continue
            os.remove(os.path.join(model_dir, filename))

    def save_model(self, epoch, test_error=None, test_final_error=None):

        model_dir = self._checkpoint_dir()
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, self.args.train_model + "_best.tar")
        torch.save({
            'epoch': epoch,
            'state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_args': self.model_run_args(),
            'best_ade': test_error,
            'best_fde': test_final_error,
        }, model_path)
        self._remove_old_checkpoints(model_path)

    def load_model(self):

        if self.args.load_model is not None:
            self.args.model_save_path = self._checkpoint_path(self.args.load_model)
            print(self.args.model_save_path)
            if os.path.isfile(self.args.model_save_path):
                print('Loading checkpoint')
                checkpoint = torch.load(self.args.model_save_path, map_location=self.device)
                model_epoch = checkpoint['epoch']
                self.validate_checkpoint_args(checkpoint)
                try:
                    load_result = self.net.load_state_dict(checkpoint['state_dict'], strict=False)
                except RuntimeError as error:
                    raise RuntimeError(
                        "Failed to load checkpoint state_dict. This usually means sample_num/model_dim/num_heads "
                        "or another shape-changing argument differs from the training run."
                    ) from error
                if load_result.missing_keys:
                    print("Missing checkpoint keys initialized from current model:", load_result.missing_keys)
                if load_result.unexpected_keys:
                    print("Unexpected checkpoint keys ignored:", load_result.unexpected_keys)
                print('Loaded checkpoint at epoch', model_epoch)

    def set_optimizer(self):

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.learning_rate)
        self.criterion = nn.MSELoss(reduction='none')

    def test(self):

        print('Testing begin')
        self.load_model()
        self.net.eval()
        test_error, test_final_error = self.test_epoch()
        print('Set: {}, epoch: {},test_error: {} test_final_error: {}'.format(self.args.test_set,
                                                                                          self.args.load_model,
                                                                                       test_error, test_final_error))
    def train(self):

        print('Training begin')
        test_error, test_final_error = 0, 0
        for epoch in range(self.args.num_epochs):

            self.net.train()
            train_loss = self.train_epoch(epoch)

            if epoch >= self.args.start_test:
                self.net.eval()
                test_error, test_final_error = self.test_epoch()
                if test_final_error < self.best_fde:
                    self.best_ade = test_error
                    self.best_epoch = epoch
                    self.best_fde = test_final_error
                    self.save_model(epoch, test_error, test_final_error)

            self.log_file_curve.write(
                str(epoch) + ',' + str(train_loss) + ',' + str(test_error) + ',' + str(test_final_error) + ',' + str(
                    self.args.learning_rate) + '\n')

            if epoch % 10 == 0:
                self.log_file_curve.close()
                self.log_file_curve = open(os.path.join(self.args.model_dir, 'log_curve.txt'), 'a+')

            if epoch >= self.args.start_test:
                print(
                    '----epoch {}, train_loss={:.5f}, ADE={:.3f}, FDE={:.3f}, Best_ADE={:.3f}, Best_FDE={:.3f} at Epoch {}'
                        .format(epoch, train_loss, test_error, test_final_error, self.best_ade, self.best_fde,
                                self.best_epoch))
            else:
                print('----epoch {}, train_loss={:.5f}'
                      .format(epoch, train_loss))

    def train_epoch(self, epoch):

        self.dataloader.reset_batch_pointer(set='train', valid=False)
        loss_epoch = 0

        for batch in range(self.dataloader.trainbatchnums):

            start = time.time()
            inputs, batch_id = self.dataloader.get_train_batch(batch)
            inputs = tuple([torch.Tensor(i).to(self.device) for i in inputs])

            batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs
            inputs_forward = batch_abs[:-1], batch_norm[:-1], shift_value[:-1], seq_list[:-1], nei_list[:-1], nei_num[
                                                                                                              :-1], batch_pednum

            self.net.zero_grad()

            outputs, aux = self.net.forward(inputs_forward, iftest=False)

            lossmask, num = getFutureLossMask(
                seq_list,
                self.args.obs_length,
                self.args.pred_length,
                using_cuda=self.args.using_cuda,
            )
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
                    'train-{}/{} (epoch {}), train_loss = {:.5f}, time/batch = {:.5f} '.format(batch,
                                                                                               self.dataloader.trainbatchnums,
                                                                                               epoch, loss.item(),
                                                                                               end - start))

        train_loss_epoch = loss_epoch / self.dataloader.trainbatchnums
        return train_loss_epoch

    @torch.no_grad()
    def test_epoch(self):
        self.dataloader.reset_batch_pointer(set='test')
        error_epoch, final_error_epoch = 0, 0,
        error_cnt_epoch, final_error_cnt_epoch = 1e-5, 1e-5

        for batch in tqdm(range(self.dataloader.testbatchnums)):

            inputs, batch_id = self.dataloader.get_test_batch(batch)
            inputs = tuple([torch.Tensor(i).to(self.device) for i in inputs])

            batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs

            inputs_forward = batch_abs[:-1], batch_norm[:-1], shift_value[:-1], seq_list[:-1], nei_list[:-1], nei_num[
                                                                                                              :-1], batch_pednum

            outputs_infer, aux = self.net.forward(inputs_forward, iftest=True)
            self.net.zero_grad()

            lossmask, num = getFutureLossMask(
                seq_list,
                self.args.obs_length,
                self.args.pred_length,
                using_cuda=self.args.using_cuda,
            )
            target = batch_norm[self.args.obs_length:self.args.obs_length + self.args.pred_length, :, :2]
            error, error_cnt, final_error, final_error_cnt = L2forTestK(outputs_infer, target, lossmask)

            error_epoch += error
            error_cnt_epoch += error_cnt
            final_error_epoch += final_error
            final_error_cnt_epoch += final_error_cnt

        return error_epoch / error_cnt_epoch, final_error_epoch / final_error_cnt_epoch
