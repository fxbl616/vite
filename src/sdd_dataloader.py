import os
import pickle
import random

import numpy as np


class SDDTrajectoryDataloader(object):
    """
    SDD adapter for local Stanford Drone Dataset pickle files.

    Expected pickle format:
        A list of numpy arrays. Each array is one scene/window and is usually
        shaped [N, T, 2] where N is the number of agents and T=20.

    The adapter converts every scene to the STAR-compatible batch tuple:
        nodes_abs, nodes_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum
    """

    def __init__(self, args):
        self.args = args
        self.obs_length = int(getattr(args, "obs_length", 8))
        self.pred_length = int(getattr(args, "pred_length", 12))
        self.seq_length = self.obs_length + self.pred_length
        self.args.seq_length = self.seq_length

        self.data_root = getattr(args, "sdd_data_root", "./data/stanford")
        self.train_file = getattr(args, "sdd_train_file", None) or os.path.join(self.data_root, "sdd_train.pkl")
        self.test_file = getattr(args, "sdd_test_file", None) or os.path.join(self.data_root, "sdd_test.pkl")
        self.unit_scale = float(getattr(args, "sdd_unit_scale", 1.0))
        self.batch_size = max(1, int(getattr(args, "batch_size", 1)))
        self.test_batch_size = max(1, int(getattr(args, "test_batch_size", self.batch_size)))
        self.neighbor_mode = str(getattr(args, "sdd_neighbor_mode", "full")).lower()
        self.neighbor_thred = float(getattr(args, "sdd_neighbor_thred", 10000.0))
        self.random_rotate = bool(getattr(args, "sdd_random_rotate", False))

        print("Preparing SDD data from:", self.data_root)
        self.train_scenes = self._load_pickle(self.train_file)
        self.test_scenes = self._load_pickle(self.test_file)

        train_limit = int(getattr(args, "sdd_train_limit", 0) or 0)
        test_limit = int(getattr(args, "sdd_test_limit", 0) or 0)
        if train_limit > 0:
            self.train_scenes = self.train_scenes[:train_limit]
        if test_limit > 0:
            self.test_scenes = self.test_scenes[:test_limit]

        if len(self.train_scenes) == 0:
            raise ValueError("SDD train split is empty. Check --sdd_data_root or --sdd_train_file.")
        if len(self.test_scenes) == 0:
            raise ValueError("SDD test split is empty. Check --sdd_data_root or --sdd_test_file.")

        self.train_batches = []
        self.test_batches = []
        self.reset_batch_pointer(set="train", valid=False)
        self.reset_batch_pointer(set="test", valid=False)

        print("Total number of SDD training scenes:", len(self.train_scenes))
        print("Total number of SDD test scenes:", len(self.test_scenes))
        print("Total number of SDD training batches:", self.trainbatchnums)
        print("Total number of SDD test batches:", self.testbatchnums)

    def _load_pickle(self, file_path):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(
                "Missing SDD pickle file: {}. Expected sdd_train.pkl/sdd_test.pkl or pass explicit "
                "--sdd_train_file/--sdd_test_file paths.".format(file_path)
            )

        with open(file_path, "rb") as handle:
            raw_scenes = pickle.load(handle)

        scenes = []
        for index, raw_scene in enumerate(raw_scenes):
            nodes = self._standardize_scene(raw_scene, file_path, index)
            if nodes.shape[0] < self.seq_length:
                continue
            nodes = nodes[:self.seq_length] / max(self.unit_scale, 1e-8)
            seq_list = np.ones((self.seq_length, nodes.shape[1]), dtype=np.float32)
            scenes.append({
                "nodes": nodes.astype(np.float32),
                "seq_list": seq_list,
                "scene_id": "{}:{}".format(os.path.basename(file_path), index),
            })
        return scenes

    def _standardize_scene(self, scene, file_path, index):
        scene = np.asarray(scene, dtype=np.float32)
        if scene.ndim != 3 or scene.shape[-1] != 2:
            raise ValueError(
                "{} scene {} must have shape [N,T,2] or [T,N,2], got {}.".format(
                    file_path, index, scene.shape
                )
            )

        if scene.shape[0] == self.seq_length:
            return scene
        if scene.shape[1] == self.seq_length:
            return np.transpose(scene, (1, 0, 2))

        if scene.shape[0] > scene.shape[1]:
            return scene
        return np.transpose(scene, (1, 0, 2))

    def _neighbor_inputs(self, nodes, seq_list):
        time_steps, num_agents, _ = nodes.shape
        nei_list = np.zeros((time_steps, num_agents, num_agents), dtype=np.float32)

        if self.neighbor_mode == "none":
            return nei_list, np.zeros((time_steps, num_agents), dtype=np.float32)

        for time_index in range(time_steps):
            present = seq_list[time_index] > 0
            valid_pair = present[:, None] & present[None, :]
            np.fill_diagonal(valid_pair, False)

            if self.neighbor_mode == "full":
                nei_list[time_index] = valid_pair.astype(np.float32)
            elif self.neighbor_mode == "distance":
                diff = nodes[time_index, :, None, :] - nodes[time_index, None, :, :]
                dist = np.linalg.norm(diff, axis=-1)
                nei_list[time_index] = (valid_pair & (dist <= self.neighbor_thred)).astype(np.float32)
            else:
                raise ValueError("Unsupported --sdd_neighbor_mode: {}".format(self.neighbor_mode))

        nei_num = np.sum(nei_list, axis=-1).astype(np.float32)
        return nei_list, nei_num

    def _batch_from_scenes(self, scenes):
        total_agents = sum(scene["nodes"].shape[1] for scene in scenes)
        nodes = np.zeros((self.seq_length, total_agents, 2), dtype=np.float32)
        seq_list = np.zeros((self.seq_length, total_agents), dtype=np.float32)
        nei_list = np.zeros((self.seq_length, total_agents, total_agents), dtype=np.float32)
        nei_num = np.zeros((self.seq_length, total_agents), dtype=np.float32)
        batch_pednum = []
        batch_id = []

        start = 0
        for scene in scenes:
            num_agents = scene["nodes"].shape[1]
            end = start + num_agents
            scene_nei_list, scene_nei_num = self._neighbor_inputs(scene["nodes"], scene["seq_list"])
            nodes[:, start:end] = scene["nodes"]
            seq_list[:, start:end] = scene["seq_list"]
            nei_list[:, start:end, start:end] = scene_nei_list
            nei_num[:, start:end] = scene_nei_num
            batch_pednum.append(num_agents)
            batch_id.append(scene["scene_id"])
            start = end

        batch_pednum = np.asarray(batch_pednum, dtype=np.float32)
        return (nodes, seq_list, nei_list, nei_num, batch_pednum), batch_id

    def _make_batches(self, scenes, batch_size, shuffle):
        indices = list(range(len(scenes)))
        if shuffle:
            random.shuffle(indices)
        batches = []
        for start in range(0, len(indices), batch_size):
            selected = [scenes[index] for index in indices[start:start + batch_size]]
            if selected:
                batches.append(self._batch_from_scenes(selected))
        return batches

    def rotate_shift_batch(self, batch_data, ifrotate=False):
        nodes, seq_list, nei_list, nei_num, batch_pednum = batch_data
        nodes = nodes.copy()

        if ifrotate:
            theta = random.random() * np.pi
            original = nodes.copy()
            nodes[:, :, 0] = original[:, :, 0] * np.cos(theta) - original[:, :, 1] * np.sin(theta)
            nodes[:, :, 1] = original[:, :, 0] * np.sin(theta) + original[:, :, 1] * np.cos(theta)

        shift_value = np.repeat(nodes[self.obs_length - 1].reshape((1, -1, 2)), self.seq_length, axis=0)
        return (
            nodes,
            nodes - shift_value,
            shift_value,
            seq_list,
            nei_list,
            nei_num,
            batch_pednum,
        )

    def get_train_batch(self, idx):
        batch_data, batch_id = self.train_batches[idx]
        return self.rotate_shift_batch(batch_data, ifrotate=self.random_rotate), batch_id

    def get_test_batch(self, idx):
        batch_data, batch_id = self.test_batches[idx]
        return self.rotate_shift_batch(batch_data, ifrotate=False), batch_id

    def reset_batch_pointer(self, set, valid=False):
        if set == "train":
            self.train_batches = self._make_batches(self.train_scenes, self.batch_size, shuffle=True)
            self.trainbatchnums = len(self.train_batches)
        else:
            self.test_batches = self._make_batches(self.test_scenes, self.test_batch_size, shuffle=False)
            self.testbatchnums = len(self.test_batches)
