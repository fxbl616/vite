# Adaptive-Prior ETH/UCY Ablation Experiments

当前结论已经从“ART + AIP + ViTE 硬融合”转向“以 ViTE-style expert routing 为强主干，学习何时使用运动先验和空间先验”。命令默认在项目根目录 `D:\program\teststar\STAR` 下执行，并且默认你已经在 CMD 中执行过：

```cmd
conda activate star
```

## 1. 新消融组

| Mode | 目的 |
| --- | --- |
| `adaptive` | 新主方法：ViTE experts + learnable spatial gate + learnable motion gate |
| `adaptive_no_motion` | 去掉 motion gate，固定使用常速度 motion prior，判断运动门控是否有用 |
| `adaptive_no_spatial` | 去掉 spatial gate，只用 uniform scene adjacency，判断空间门控是否有用 |
| `adaptive_motion_only` | 不用专家交互，只保留 temporal encoder + motion gate，判断门控自身是否有用 |
| `vite_only` | 当前最强旧基线：ViTE-style experts + uniform adjacency + 固定 motion prior |
| `motion_only` | 简单运动基线 |
| `spatial_only` | 简单空间先验基线 |

旧的 `full / art_only / art_aip / no_motion_prior` 仍然保留在代码里，但不再作为默认新实验主线。

## 2. 模型变化

`adaptive` 不再使用 ART relation、AIP 和 relational Transformer。它的路径是：

```text
observed [x,y,vx,vy]
-> temporal encoder
-> uniform scene graph + optional spatial-prior gate
-> one-hop expert + virtual high-order expert
-> router fusion
-> K decoder heads
-> adaptive motion gate scales constant-velocity baseline
```

新增两个可学习门控：

- `spatial_gate`: 每个行人一标量，决定 one-hop graph 更接近 uniform adjacency 还是 STAR spatial prior。
- `motion_gate`: 每个行人一标量，决定 decoder baseline 更接近静止 extrapolation 还是常速度 extrapolation。

门控输入包含节点特征和运动上下文：

```text
[speed, acceleration, turning, scene_density]
```

## 3. 运行命令

补跑 `univ / zara1 / zara2`：

```cmd
scripts\run_eth_ucy_ablation_remaining.cmd
```

单独跑某个数据集：

```cmd
scripts\run_eth_ucy_ablation_remaining.cmd univ
scripts\run_eth_ucy_ablation_remaining.cmd zara1
scripts\run_eth_ucy_ablation_remaining.cmd zara2
```

先检查命令，不启动训练：

```cmd
scripts\run_eth_ucy_ablation_remaining.cmd --dry-run univ
```

每组输出目录：

```text
output\<dataset>\ab_<dataset>_<mode>\
```

主要文件：

```text
train_stdout.txt
test_stdout_best.txt
log_curve.txt
<train_model>_best.tar
```

脚本会在每次训练前删除旧 `config_train.yaml`，测试前删除旧 `config_test.yaml`，确保命令行里的新参数生效。

## 4. 按数据集超参数

Top-P Threshold 对应 `--router_top_p`，Expert Router Layers 对应 `--rt_layers`。

| Dataset | `--learning_rate` | `--num_epochs` | `--router_top_p` | `--rt_layers` | `--num_virtual_nodes` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `eth` | 0.001 | 300 | 0.5 | 3 | 3 |
| `hotel` | 0.0018 | 300 | 0.6 | 3 | 3 |
| `univ` | 0.001 | 300 | 0.5 | 3 | 3 |
| `zara1` | 0.0012 | 300 | 0.6 | 3 | 3 |
| `zara2` | 0.0012 | 300 | 0.5 | 3 | 3 |

公共参数：

```cmd
--dataset eth5
--start_test 10
--sample_num 20
--fde_weight 0.5
--spatial_prior_mix 0.6
--spatial_sigma 2.0
--diversity_weight 0.02
--diversity_margin 0.2
--motion_gate_bias 1.0
--spatial_gate_bias -1.0
```

## 5. 单条命令模板

以 `univ + adaptive` 为例：

```cmd
mkdir output\univ\ab_univ_adaptive 2>nul & del /q output\univ\ab_univ_adaptive\config_train.yaml 2>nul & python trainval.py --phase train --test_set univ --train_model ab_univ_adaptive --ablation adaptive --dataset eth5 --num_epochs 300 --start_test 10 --sample_num 20 --fde_weight 0.5 --spatial_prior_mix 0.6 --spatial_sigma 2.0 --diversity_weight 0.02 --diversity_margin 0.2 --learning_rate 0.001 --router_top_p 0.5 --rt_layers 3 --num_virtual_nodes 3 --motion_gate_bias 1.0 --spatial_gate_bias -1.0 > output\univ\ab_univ_adaptive\train_stdout.txt 2>&1
```

测试：

```cmd
mkdir output\univ\ab_univ_adaptive 2>nul & del /q output\univ\ab_univ_adaptive\config_test.yaml 2>nul & python trainval.py --phase test --test_set univ --train_model ab_univ_adaptive --ablation adaptive --load_model best --dataset eth5 --num_epochs 300 --start_test 10 --sample_num 20 --fde_weight 0.5 --spatial_prior_mix 0.6 --spatial_sigma 2.0 --diversity_weight 0.02 --diversity_margin 0.2 --learning_rate 0.001 --router_top_p 0.5 --rt_layers 3 --num_virtual_nodes 3 --motion_gate_bias 1.0 --spatial_gate_bias -1.0 > output\univ\ab_univ_adaptive\test_stdout_best.txt 2>&1
```

## 6. 汇总结果

训练完成后运行：

```cmd
python scripts\summarize_eth_ucy_ablations.py --output docs\eth_ucy_ablation_summary.md
```

重点看：

- `adaptive - vite_only`: 新方法是否超过当前最强旧基线。
- `adaptive - adaptive_no_motion`: motion gate 是否有贡献。
- `adaptive - adaptive_no_spatial`: spatial gate 是否有贡献。
- `adaptive - motion_only`: 新交互主干是否超过简单运动模型。

delta 使用 FDE，负数表示左边方法更好。
