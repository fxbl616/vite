# STAR

Code for [Spatio-Temporal Graph Transformer Networks for Pedestrian Trajectory Prediction](https://arxiv.org/abs/2005.08514)

### Environment

```bash
pip install numpy==1.18.1
pip install torch==1.7.0
pip install pyyaml=5.3.1
pip install tqdm=4.45.0
```

### Train

The Default settings are to train on ETH-univ dataset.

Data cache and models will be stored in the subdirectory "./output/eth/" by default. Notice that for this repo, we only provide implementation on GPU.

```
git clone https://github.com/Majiker/STAR.git
cd STAR
python trainval.py --test_set <dataset to evaluate> --start_test <epoch to start test>
```

Configuration files are also created after the first run, arguments could be modified through configuration files or command line.
Priority: command line \> configuration files \> default values in script.

The datasets are selected on arguments '--test_set'. Five datasets in ETH/UCY including
[**eth, hotel, zara1, zara2, univ**].

### Example

This command is to train model for ETH-hotel and start test at epoch 10. For different dataset, change 'hotel' to other datasets named in the last section.

```
python trainval.py --test_set hotel --start_test 50
```

During training, the model for Best FDE on the corresponding test dataset would be record.

### Cite STAR

If you find this repo useful, please consider citing our paper

```bibtex
@inproceedings{
    YuMa2020Spatio,
    title={Spatio-Temporal Graph Transformer Networks for Pedestrian Trajectory Prediction},
    author={Cunjun Yu and Xiao Ma and Jiawei Ren and Haiyu Zhao and Shuai Yi},
    booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
    month = {August},
    year={2020}
}
```

### Reference

The code base heavily borrows from [SR-LSTM](https://github.com/zhangpur/SR-LSTM)

python trainval.py --phase train --test_set eth --train_model star_art_vite_v2 --num_epochs 300 --start_test 10 --sample_num 20 --fde_weight 0.5 --spatial_prior_mix 0.6 --spatial_sigma 2.0 --diversity_weight 0.02 --diversity_margin 0.2

----epoch 299, train_loss=0.34837, ADE=0.130, FDE=0.229, Best_ADE=0.128, Best_FDE=0.224 at Epoch 292 zara2

### NBA adapter

NBA support is isolated from the ETH/UCY path. Use `--dataset nba` to enable the NBA dataloader and processor.

Before training, place MART-style NBA files at:

```text
data/nba/nba_train.npy
data/nba/nba_test.npy
```

Or check whether an existing local MART/LED copy can be found:

```bash
python scripts/prepare_nba_data.py
```

MART-style npy training:

```bash
conda run -n star python trainval.py --dataset nba --phase train --train_model star_art_vite_v2_nba_mart --nba_format mart_npy --nba_data_root ./data/nba --obs_length 10 --pred_length 20 --seq_length 30 --sample_num 20 --num_epochs 300 --start_test 10 --spatial_prior_mix 0.3 --spatial_sigma 2.0 --nba_neighbor_mode full
```

### SDD adapter

The local SDD adapter expects:

```text
data/stanford/sdd_train.pkl
data/stanford/sdd_test.pkl
```

SDD training:

```bash
conda run -n star python trainval.py --dataset sdd --phase train --train_model star_art_vite_v2_sdd --sdd_data_root ./data/stanford --obs_length 8 --pred_length 12 --seq_length 20 --sample_num 20 --num_epochs 300 --start_test 10 --spatial_prior_mix 0.3 --spatial_sigma 100.0 --sdd_neighbor_mode full --router_top_p 0.5 --learning_rate 0.001 --rt_layers 1 --num_virtual_nodes 1
```

SDD testing:

```bash
conda run -n star python trainval.py --dataset sdd --phase test --train_model star_art_vite_v2_sdd --load_model 44 --sdd_data_root ./data/stanford --obs_length 8 --pred_length 12 --seq_length 20 --sample_num 20 --spatial_prior_mix 0.3 --spatial_sigma 100.0 --sdd_neighbor_mode full
```

### Ablation experiments

消融实验已经统一接入 `--ablation` 参数，支持 full、motion_only、spatial_only、art_only、art_aip、vite_only、no_motion_prior。

完整命令见 docs/ablation_experiments.md。

```bash
python trainval.py --dataset nba --phase train --train_model star_art_vite_v2_nba_mart --nba_format mart_npy --nba_data_root ./data/nba --obs_length 10 --pred_length 20 --seq_length 30 --sample_num 20 --num_epochs 100 --start_test 10 --spatial_prior_mix 0.3 --spatial_sigma 2.0 --nba_neighbor_mode full --router_top_p 0.5 --learning_rate 0.0005 --rt_layers 2 --num_virtual_nodes 2

python trainval.py --dataset eth5 --phase train --test_set eth --train_model star_art_vite_v2 --num_epochs 300 --start_test 10 --sample_num 20 --fde_weight 0.5 --spatial_prior_mix 0.6 --spatial_sigma 2.0 --diversity_weight 0.02 --diversity_margin 0.2
--router_top_p 0.5 --learning_rate 0.0005 --rt_layers 3 --num_virtual_nodes 3

conda run -n star python trainval.py --dataset eth5 --phase test --test_set eth --train_model star_art_vite_v2 --load_model 44 --sample_num 20 --fde_weight 0.5 --spatial_prior_mix 0.6 --spatial_sigma 2.0 --diversity_weight 0.02 --diversity_margin 0.2
```
