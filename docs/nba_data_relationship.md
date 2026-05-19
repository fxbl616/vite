# NBA 数据关系与 STAR 适配说明

本文档说明当前项目如何把 NBA 轨迹数据整理成 STAR-ART-ViTE v2 能直接使用的 batch。目标是隔离 NBA 改动，不影响 ETH/UCY 的原始数据链路。

## 1. 为什么单独加 NBA dataloader

ETH/UCY 的 STAR 数据链路默认是行人场景：

```text
true_pos_.csv
-> Trajectory_Dataloader
-> nodes_abs / nodes_norm / seq_list / nei_list / batch_pednum
-> STAR.forward
```

NBA 数据不是普通行人场景。它通常包含球员、篮球、固定球场坐标和不同时间采样设置。因此当前实现新增：

```text
src/nba_dataloader.py
src/nba_processor.py
```

而不是把 NBA 逻辑塞进 `src/utils.py` 和 `src/processor.py`。这样 ETH/UCY 的训练和测试路径保持原样。

## 2. 支持的两种 NBA 来源

当前 NBA adapter 支持两种常见格式。

### 2.1 MART-style npy

MART 的 NBA loader 使用：

```text
nba_train.npy
nba_test.npy
```

常见形状为：

```text
[num_scene, seq_len, num_agent, 2]
```

或者：

```text
[num_scene, num_agent, seq_len, 2]
```

当前 adapter 会自动判断时间轴，并统一转成：

```text
[num_scene, seq_len, num_agent, 2]
```

MART 代码中会把 NBA 坐标除以：

```text
94 / 28
```

这相当于把 NBA 球场的 feet 坐标近似转成 meter。当前 adapter 对 `mart_npy` 默认也使用这个 scale：

```text
nodes = nodes / (94 / 28)
```

如果想改单位，可以传：

```bash
--nba_unit_scale <scale>
```

### 2.2 SocialVAE-style txt

SocialVAE 的 NBA 文本格式更透明，每一行是：

```text
frame_ID agent_ID pos_x pos_y group
```

其中：

```text
group = PLAYER 或 BALL
```

当前 adapter 会按时间滑窗生成 scene：

```text
obs_length + pred_length = seq_length
```

默认：

```text
--nba_target_groups PLAYER
--nba_context_groups PLAYER,BALL
```

也就是说：

```text
PLAYER 参与预测 loss
PLAYER 和 BALL 都进入交互图
```

篮球作为 context agent 使用，不默认作为预测目标。

## 3. 统一后的内部 scene 表示

不管原始数据来自 MART npy 还是 SocialVAE txt，都会先整理成统一 scene：

```text
nodes:       [T, N, 2]
seq_list:    [T, N]
target_mask: [N]
groups:      List[str]
scene_id:    str
```

含义如下：

| 字段 | 含义 |
|---|---|
| `nodes` | 原始坐标，长度为 `obs_length + pred_length` |
| `seq_list` | agent 在每一帧是否存在 |
| `target_mask` | 哪些 agent 参与 loss / ADE / FDE |
| `groups` | SocialVAE txt 中的 `PLAYER` / `BALL`；MART npy 没有 group，默认全是 `AGENT` |
| `scene_id` | 调试用的场景来源标识 |

## 4. 从 scene 到 STAR-style batch

多个 NBA scene 可以合并成一个 batch。合并时沿 agent 维拼接：

```text
scene_1: [T, N1, 2]
scene_2: [T, N2, 2]

batch:   [T, N1 + N2, 2]
```

为了避免不同 NBA scene 之间发生交互，`nei_list` 会做成 block diagonal：

```text
scene_1 agents 只连 scene_1
scene_2 agents 只连 scene_2
跨 scene 的边为 0
```

`batch_pednum` 记录每个 scene 的 agent 数：

```text
batch_pednum = [N1, N2, ...]
```

这和 STAR-ART-ViTE v2 的 `scene_slices` 完全对应。

## 5. STAR.forward 看到的输入

NBA processor 最终仍然构造 STAR 原本的输入 tuple：

```text
nodes_abs:     [T, N, 2]
nodes_norm:    [T, N, 2]
shift_value:   [T, N, 2]
seq_list:      [T, N]
nei_list:      [T, N, N]
nei_num:       [T, N]
batch_pednum:  [num_scene]
```

其中 normalized coordinate 和 ETH/UCY 一样，以最后观测帧为中心：

```text
shift_value[t, i] = nodes_abs[obs_length - 1, i]
nodes_norm = nodes_abs - shift_value
```

所以模型主体不需要知道数据来自 ETH/UCY 还是 NBA。

## 6. target_mask 的作用

NBA 和 ETH/UCY 最大的差别之一是：有些 agent 只应该作为 context，不应该作为预测目标。

例如 SocialVAE txt 中：

```text
PLAYER -> target + context
BALL   -> context only
```

因此 NBA processor 会在 loss 前做：

```text
lossmask = future_presence_mask * target_mask
```

这样：

```text
BALL 参与 ART relation graph / AIP / expert routing
BALL 不参与 best-of-K loss
BALL 不参与 minADE_K / minFDE_K
```

MART npy 没有 group 信息，当前默认：

```text
target_mask = all ones
```

也就是所有 agent 都参与预测和评价。

如果发现 MART/LED 风格的 `.npy` 里包含篮球，而且篮球不应该参与 ADE/FDE，可以手动指定 target agent 的下标：

```bash
--nba_target_indices 0-9
```

上面表示只让第 0 到第 9 个 agent 参与 loss 和 evaluation，其余 agent 仍可作为 context 进入图。

## 7. NBA 邻接先验

ETH/UCY 的原 STAR 依赖 `neighbor_thred` 生成局部邻居。NBA 上这个逻辑不应直接照搬，因为篮球场景中远距离队友、防守人和篮球也可能有关系。

当前 NBA adapter 支持三种基础邻接：

```text
--nba_neighbor_mode full
--nba_neighbor_mode distance
--nba_neighbor_mode none
```

默认是：

```text
full
```

这表示同一个 scene 内所有同时存在的 agent 都先作为候选邻居，再交给模型里的 ART relation graph 和 AIP top-p pruning 选择重要边。

如果使用：

```text
distance
```

则会根据：

```text
--nba_neighbor_thred
```

做距离阈值邻居。

## 8. 推荐命令

### MART npy

MART NBA 常见设置是观测 10 帧、预测 20 帧：

```bash
python trainval.py --dataset nba --phase train --train_model star_art_vite_v2_nba_mart --nba_format mart_npy --nba_data_root ./data/nba --obs_length 10 --pred_length 20 --seq_length 30 --sample_num 20 --spatial_prior_mix 0.3 --spatial_sigma 2.0 --nba_neighbor_mode full
```

如果数据文件不在 `./data/nba/nba_train.npy` 和 `./data/nba/nba_test.npy`：

```bash
python trainval.py --dataset nba --nba_format mart_npy --nba_train_file <path/to/nba_train.npy> --nba_test_file <path/to/nba_test.npy> --obs_length 10 --pred_length 20 --seq_length 30
```

### SocialVAE txt

SocialVAE 风格可以保留 8/12 设置：

```bash
python trainval.py --dataset nba --phase train --train_model star_art_vite_v2_nba_socialvae --nba_format socialvae_txt --nba_data_root ./data/nba/rebound --obs_length 8 --pred_length 12 --seq_length 20 --nba_target_groups PLAYER --nba_context_groups PLAYER,BALL --nba_neighbor_mode full
```

目录结构默认应类似：

```text
data/nba/rebound/train/*.txt
data/nba/rebound/test/*.txt
```

也可以显式传：

```bash
--nba_train_dir <train_txt_dir> --nba_test_dir <test_txt_dir>
```

## 9. ETH/UCY 是否受影响

默认命令仍然是：

```bash
python trainval.py --dataset eth5 --test_set eth ...
```

此时仍然使用：

```text
src/processor.py
src/utils.py::Trajectory_Dataloader
```

只有当：

```bash
--dataset nba
```

时，才会使用：

```text
src/nba_processor.py
src/nba_dataloader.py
```

因此 NBA 适配不会改变 ETH/UCY 的 dataloader、loss mask 或测试循环。

## 10. 当前要特别注意的实验公平性

1. MART npy 和 SocialVAE txt 可能不是同一个 split，不能混着和论文表格硬比。
2. MART 默认把 feet 转成 meter，SocialVAE 论文可能报告 feet，需要确认单位后再比较。
3. `obs_length/pred_length` 必须和 baseline 一致。MART 常见 NBA 是 10/20，SocialVAE 风格常见可用 8/12。
4. 如果篮球只作为 context，需要报告清楚 `BALL` 不参与 ADE/FDE。
5. NBA 上 `spatial_prior_mix=0.6` 未必合适，建议从 `0.0/0.2/0.4/0.6` 做小网格。
