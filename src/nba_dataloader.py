import os
import random

import numpy as np


def _parse_group_list(value, default=None):
    if value is None:
        return default
    if isinstance(value, (list, tuple, set)):
        groups = value
    else:
        groups = str(value).replace(";", ",").split(",")
    groups = {str(group).strip().upper() for group in groups if str(group).strip()}
    return groups if groups else default


def _parse_index_list(value):
    if value is None or str(value).strip() == "":
        return None
    indices = []
    for item in str(value).replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            indices.extend(range(int(start), int(end) + 1))
        else:
            indices.append(int(item))
    return sorted(set(indices))


def _agent_sort_key(agent_id, group, target_groups):
    target_rank = 0 if group in target_groups else 1
    try:
        numeric_id = int(agent_id)
    except ValueError:
        numeric_id = agent_id
    return target_rank, group, numeric_id


class NBATrajectoryDataloader(object):
    """
    NBA adapter that converts common basketball trajectory formats into the
    STAR batch tuple:

    nodes_abs, nodes_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum

    A separate target_mask marks which agents should contribute to loss.
    Context agents such as the ball can still participate in the graph.
    """

    def __init__(self, args):
        self.args = args
        self.obs_length = int(getattr(args, "obs_length", 8))
        self.pred_length = int(getattr(args, "pred_length", 12))
        self.seq_length = self.obs_length + self.pred_length
        self.args.seq_length = self.seq_length

        self.data_format = str(getattr(args, "nba_format", "mart_npy")).lower()
        self.data_root = getattr(args, "nba_data_root", "./data/nba")
        self.batch_size = max(1, int(getattr(args, "batch_size", 1)))
        self.test_batch_size = max(1, int(getattr(args, "test_batch_size", self.batch_size)))
        self.stride = max(1, int(getattr(args, "nba_stride", 1)))
        self.random_rotate = bool(getattr(args, "nba_random_rotate", False))
        self.neighbor_mode = str(getattr(args, "nba_neighbor_mode", "full")).lower()
        self.neighbor_thred = float(getattr(args, "nba_neighbor_thred", 10000.0))
        self.target_groups = _parse_group_list(getattr(args, "nba_target_groups", "PLAYER"), {"PLAYER"})
        self.context_groups = _parse_group_list(getattr(args, "nba_context_groups", "PLAYER,BALL"), None)
        self.target_indices = _parse_index_list(getattr(args, "nba_target_indices", None))
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

        print("Preparing NBA data with format:", self.data_format)
        self.train_scenes = self._load_split("train")
        self.test_scenes = self._load_split("test")

        train_limit = int(getattr(args, "nba_train_limit", 0) or 0)
        test_limit = int(getattr(args, "nba_test_limit", 0) or 0)
        if train_limit > 0:
            self.train_scenes = self.train_scenes[:train_limit]
        if test_limit > 0:
            self.test_scenes = self.test_scenes[:test_limit]

        if len(self.train_scenes) == 0:
            raise ValueError("NBA train split is empty. Check --nba_data_root or --nba_train_file.")
        if len(self.test_scenes) == 0:
            raise ValueError("NBA test split is empty. Check --nba_data_root or --nba_test_file.")

        self.train_batches = []
        self.test_batches = []
        self.reset_batch_pointer(set="train", valid=False)
        self.reset_batch_pointer(set="test", valid=False)

        print("Total number of NBA training scenes:", len(self.train_scenes))
        print("Total number of NBA test scenes:", len(self.test_scenes))
        print("Total number of NBA training batches:", self.trainbatchnums)
        print("Total number of NBA test batches:", self.testbatchnums)

    def _unit_scale(self):
        scale = getattr(self.args, "nba_unit_scale", None)
        if scale is not None:
            return float(scale)
        if self.data_format in ("mart", "mart_npy"):
            # MART divides NBA court coordinates by 94/28 to convert feet to meters.
            return 94.0 / 28.0
        # SocialVAE reports NBA results in feet, so keep feet unless the user opts in.
        return 1.0

    def _load_split(self, split):
        if self.data_format in ("mart", "mart_npy"):
            return self._load_mart_npy(split)
        if self.data_format in ("socialvae", "socialvae_txt", "txt"):
            return self._load_socialvae_txt(split)
        raise ValueError("Unsupported --nba_format: {}".format(self.data_format))

    def _split_path(self, split, suffix):
        specific = getattr(self.args, "nba_{}_{}".format(split, suffix), None)
        if specific:
            return specific
        return None

    def _candidate_mart_roots(self):
        roots = [
            self.data_root,
            os.path.join(self.project_root, "data", "nba"),
            os.path.join(self.project_root, "external", "MART", "datasets", "nba"),
            os.path.join(self.project_root, os.pardir, "MART", "datasets", "nba"),
            os.path.join(self.project_root, os.pardir, "MART", "datasets", "nba_data"),
            os.path.join(self.project_root, os.pardir, "LED", "datasets", "nba"),
        ]
        normalized = []
        for root in roots:
            abs_root = os.path.abspath(root)
            if abs_root not in normalized:
                normalized.append(abs_root)
        return normalized

    def _resolve_mart_file(self, split):
        expected_name = "nba_{}.npy".format(split)
        for root in self._candidate_mart_roots():
            file_path = os.path.join(root, expected_name)
            if os.path.isfile(file_path):
                return file_path
        return os.path.join(self.data_root, expected_name)

    def _missing_mart_message(self, split, file_path):
        candidates = [
            os.path.join(root, "nba_{}.npy".format(split))
            for root in self._candidate_mart_roots()
        ]
        candidate_text = "\n  - ".join(candidates)
        return (
            "Missing NBA MART npy file: {missing}\n"
            "Expected a file named nba_{split}.npy. Checked:\n"
            "  - {candidates}\n\n"
            "Fix options:\n"
            "  1. Put MART NBA files at STAR/data/nba/nba_train.npy and STAR/data/nba/nba_test.npy.\n"
            "  2. Or pass explicit paths with --nba_train_file and --nba_test_file.\n"
            "  3. Or clone/download MART so files exist at ../MART/datasets/nba/.\n\n"
            "Do not use output/synthetic_nba_mart for real experiments; it is only a smoke-test dataset."
        ).format(missing=file_path, split=split, candidates=candidate_text)

    def _load_mart_npy(self, split):
        file_arg = self._split_path(split, "file")
        file_path = file_arg or self._resolve_mart_file(split)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(self._missing_mart_message(split, file_path))

        data = np.load(file_path).astype(np.float32)
        data = self._standardize_mart_array(data, file_path)
        if data.shape[1] < self.seq_length:
            raise ValueError(
                "{} has only {} frames, but obs_length + pred_length = {}.".format(
                    file_path, data.shape[1], self.seq_length
                )
            )

        scale = self._unit_scale()
        data = data[:, :self.seq_length, :, :] / max(scale, 1e-8)
        scenes = []
        for index, nodes in enumerate(data):
            seq_list = np.ones((self.seq_length, nodes.shape[1]), dtype=np.float32)
            target_mask = self._mart_target_mask(nodes.shape[1])
            scenes.append({
                "nodes": nodes,
                "seq_list": seq_list,
                "target_mask": target_mask,
                "groups": ["AGENT"] * nodes.shape[1],
                "scene_id": "{}:{}".format(os.path.basename(file_path), index),
            })
        return scenes

    def _mart_target_mask(self, num_agents):
        if self.target_indices is None:
            return np.ones(num_agents, dtype=np.float32)
        target_mask = np.zeros(num_agents, dtype=np.float32)
        for index in self.target_indices:
            if index < 0 or index >= num_agents:
                raise ValueError(
                    "--nba_target_indices contains {}, but this MART scene has {} agents.".format(
                        index, num_agents
                    )
                )
            target_mask[index] = 1.0
        return target_mask

    def _standardize_mart_array(self, data, file_path):
        if data.ndim != 4 or data.shape[-1] != 2:
            raise ValueError("{} must have shape [S,T,N,2] or [S,N,T,2].".format(file_path))

        axis1_can_be_time = data.shape[1] >= self.seq_length
        axis2_can_be_time = data.shape[2] >= self.seq_length
        if axis1_can_be_time and not axis2_can_be_time:
            return data
        if axis2_can_be_time and not axis1_can_be_time:
            return np.transpose(data, (0, 2, 1, 3))

        # Ambiguous case: NBA scenes usually have far fewer agents than frames.
        if data.shape[1] <= 32 and data.shape[2] >= self.seq_length:
            return np.transpose(data, (0, 2, 1, 3))
        return data

    def _load_socialvae_txt(self, split):
        dir_arg = self._split_path(split, "dir")
        folder = dir_arg or os.path.join(self.data_root, split)
        if not os.path.isdir(folder):
            raise FileNotFoundError(
                "Missing NBA SocialVAE txt folder: {}. Expected txt scenes or pass "
                "--nba_{}_dir.".format(folder, split)
            )

        files = [
            os.path.join(folder, name)
            for name in sorted(os.listdir(folder))
            if name.lower().endswith((".txt", ".csv"))
        ]
        scenes = []
        for file_path in files:
            scenes.extend(self._read_socialvae_file(file_path))
        return scenes

    def _read_socialvae_file(self, file_path):
        frame_map = {}
        agent_group = {}
        with open(file_path, "r") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.replace(",", " ").split()
                if len(parts) < 4:
                    raise ValueError("{}:{} has fewer than 4 columns.".format(file_path, line_number))
                frame = int(float(parts[0]))
                agent_id = str(parts[1])
                x_pos = float(parts[2])
                y_pos = float(parts[3])
                group = parts[4].upper() if len(parts) >= 5 else "PLAYER"
                if self.context_groups is not None and group not in self.context_groups:
                    continue
                frame_map.setdefault(frame, {})[agent_id] = (x_pos, y_pos)
                agent_group[agent_id] = group

        frames = sorted(frame_map.keys())
        if len(frames) < self.seq_length:
            return []

        scale = self._unit_scale()
        scenes = []
        for start in range(0, len(frames) - self.seq_length + 1, self.stride):
            window_frames = frames[start:start + self.seq_length]
            agents = [
                agent_id for agent_id, group in agent_group.items()
                if any(agent_id in frame_map[frame] for frame in window_frames)
                and (self.context_groups is None or group in self.context_groups)
            ]
            agents = sorted(
                agents,
                key=lambda item: _agent_sort_key(item, agent_group[item], self.target_groups),
            )
            if not agents:
                continue

            nodes = np.zeros((self.seq_length, len(agents), 2), dtype=np.float32)
            seq_list = np.zeros((self.seq_length, len(agents)), dtype=np.float32)
            target_mask = np.zeros(len(agents), dtype=np.float32)
            groups = []

            for agent_index, agent_id in enumerate(agents):
                group = agent_group[agent_id]
                groups.append(group)
                target_mask[agent_index] = 1.0 if group in self.target_groups else 0.0
                for time_index, frame in enumerate(window_frames):
                    if agent_id in frame_map[frame]:
                        nodes[time_index, agent_index] = frame_map[frame][agent_id]
                        seq_list[time_index, agent_index] = 1.0

            nodes = nodes / max(scale, 1e-8)
            obs_full = seq_list[:self.obs_length].all(axis=0)
            if not bool(np.any((target_mask > 0) & obs_full)):
                continue

            scenes.append({
                "nodes": nodes,
                "seq_list": seq_list,
                "target_mask": target_mask,
                "groups": groups,
                "scene_id": "{}:{}-{}".format(os.path.basename(file_path), window_frames[0], window_frames[-1]),
            })
        return scenes

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
                raise ValueError("Unsupported --nba_neighbor_mode: {}".format(self.neighbor_mode))

        nei_num = np.sum(nei_list, axis=-1).astype(np.float32)
        return nei_list, nei_num

    def _batch_from_scenes(self, scenes):
        total_agents = sum(scene["nodes"].shape[1] for scene in scenes)
        nodes = np.zeros((self.seq_length, total_agents, 2), dtype=np.float32)
        seq_list = np.zeros((self.seq_length, total_agents), dtype=np.float32)
        nei_list = np.zeros((self.seq_length, total_agents, total_agents), dtype=np.float32)
        nei_num = np.zeros((self.seq_length, total_agents), dtype=np.float32)
        target_mask = np.zeros(total_agents, dtype=np.float32)
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
            target_mask[start:end] = scene["target_mask"]
            batch_pednum.append(num_agents)
            batch_id.append(scene["scene_id"])
            start = end

        batch_pednum = np.asarray(batch_pednum, dtype=np.float32)
        batch_data = (nodes, seq_list, nei_list, nei_num, batch_pednum, target_mask)
        return batch_data, batch_id

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
        nodes, seq_list, nei_list, nei_num, batch_pednum, target_mask = batch_data
        nodes = nodes.copy()

        if ifrotate:
            theta = random.random() * np.pi
            original = nodes.copy()
            nodes[:, :, 0] = original[:, :, 0] * np.cos(theta) - original[:, :, 1] * np.sin(theta)
            nodes[:, :, 1] = original[:, :, 0] * np.sin(theta) + original[:, :, 1] * np.cos(theta)

        shift_value = np.repeat(nodes[self.obs_length - 1].reshape((1, -1, 2)), self.seq_length, axis=0)
        inputs = (
            nodes,
            nodes - shift_value,
            shift_value,
            seq_list,
            nei_list,
            nei_num,
            batch_pednum,
        )
        return inputs, target_mask

    def get_train_batch(self, idx):
        batch_data, batch_id = self.train_batches[idx]
        inputs, target_mask = self.rotate_shift_batch(batch_data, ifrotate=self.random_rotate)
        return inputs, target_mask, batch_id

    def get_test_batch(self, idx):
        batch_data, batch_id = self.test_batches[idx]
        inputs, target_mask = self.rotate_shift_batch(batch_data, ifrotate=False)
        return inputs, target_mask, batch_id

    def reset_batch_pointer(self, set, valid=False):
        if set == "train":
            self.train_batches = self._make_batches(self.train_scenes, self.batch_size, shuffle=True)
            self.trainbatchnums = len(self.train_batches)
        else:
            self.test_batches = self._make_batches(self.test_scenes, self.test_batch_size, shuffle=False)
            self.testbatchnums = len(self.test_batches)
