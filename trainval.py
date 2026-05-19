import argparse
import ast
import os

import torch
import yaml

# Use Deterministic mode and set random seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)


def get_parser():
    parser = argparse.ArgumentParser(
        description='STAR')
    parser.add_argument('--dataset', default='eth5')
    parser.add_argument('--save_dir')
    parser.add_argument('--model_dir')
    parser.add_argument('--config')
    parser.add_argument('--using_cuda', default=True, type=ast.literal_eval)
    parser.add_argument('--test_set', default='eth', type=str,
                        help='Set this value to [eth, hotel, zara1, zara2, univ] for ETH-univ, ETH-hotel, UCY-zara01, UCY-zara02, UCY-univ')
    parser.add_argument('--base_dir', default='.', help='Base directory including these scripts.')
    parser.add_argument('--save_base_dir', default='./output/', help='Directory for saving caches and models.')
    parser.add_argument('--phase', default='train', help='Set this value to \'train\' or \'test\'')
    parser.add_argument('--train_model', default='star', help='Your model name')
    parser.add_argument('--load_model', default=None, type=str, help="load pretrained model for test or training")
    parser.add_argument('--model', default='star.STAR')
    parser.add_argument('--seq_length', default=20, type=int)
    parser.add_argument('--obs_length', default=8, type=int)
    parser.add_argument('--pred_length', default=12, type=int)
    parser.add_argument('--batch_around_ped', default=256, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--test_batch_size', default=4, type=int)
    parser.add_argument('--show_step', default=100, type=int)
    parser.add_argument('--start_test', default=10, type=int)
    parser.add_argument('--sample_num', default=20, type=int)
    parser.add_argument('--num_epochs', default=300, type=int)
    parser.add_argument('--ifshow_detail', default=True, type=ast.literal_eval)
    parser.add_argument('--ifsave_results', default=False, type=ast.literal_eval)
    parser.add_argument('--randomRotate', default=True, type=ast.literal_eval,
                        help="=True:random rotation of each trajectory fragment")
    parser.add_argument('--neighbor_thred', default=10, type=int)
    parser.add_argument('--learning_rate', default=0.0015, type=float)
    parser.add_argument('--clip', default=1, type=int)
    parser.add_argument('--model_dim', default=64, type=int)
    parser.add_argument('--num_heads', default=4, type=int)
    parser.add_argument('--rt_layers', default=1, type=int)
    parser.add_argument('--aip_top_p', default=0.75, type=float)
    parser.add_argument('--router_top_p', default=0.7, type=float)
    parser.add_argument('--num_virtual_nodes', default=3, type=int)
    parser.add_argument('--moe_weight', default=0.01, type=float)
    parser.add_argument('--fde_weight', default=0.5, type=float)
    parser.add_argument('--spatial_prior_mix', default=0.6, type=float)
    parser.add_argument('--spatial_sigma', default=2.0, type=float)
    parser.add_argument('--diversity_weight', default=0.02, type=float)
    parser.add_argument('--diversity_margin', default=0.2, type=float)
    parser.add_argument(
        '--ablation',
        default='full',
        choices=[
            'full',
            'motion_only',
            'spatial_only',
            'art_only',
            'art_aip',
            'vite_only',
            'no_motion_prior',
            'adaptive',
            'adaptive_no_motion',
            'adaptive_no_spatial',
            'adaptive_motion_only',
        ],
        help='Ablation switch. adaptive is the new ViTE-style adaptive-prior model.',
    )
    parser.add_argument('--motion_gate_bias', default=1.0, type=float,
                        help='Initial bias for adaptive motion gate. Positive starts closer to constant velocity.')
    parser.add_argument('--spatial_gate_bias', default=-1.0, type=float,
                        help='Initial bias for adaptive spatial gate. Negative starts closer to uniform ViTE adjacency.')
    parser.add_argument('--nba_format', default='mart_npy',
                        help='NBA data format: mart_npy or socialvae_txt')
    parser.add_argument('--nba_data_root', default='./data/nba',
                        help='Root for NBA data. MART expects nba_train.npy/nba_test.npy; SocialVAE expects train/test txt folders.')
    parser.add_argument('--nba_train_file', default=None,
                        help='Optional MART-style train npy file path.')
    parser.add_argument('--nba_test_file', default=None,
                        help='Optional MART-style test npy file path.')
    parser.add_argument('--nba_train_dir', default=None,
                        help='Optional SocialVAE-style train txt directory.')
    parser.add_argument('--nba_test_dir', default=None,
                        help='Optional SocialVAE-style test txt directory.')
    parser.add_argument('--nba_unit_scale', default=None, type=float,
                        help='Divide NBA coordinates by this scale. Default: MART uses 94/28 feet-to-meter scale; SocialVAE keeps feet.')
    parser.add_argument('--nba_neighbor_mode', default='full',
                        help='NBA neighbor graph before model pruning: full, distance, or none.')
    parser.add_argument('--nba_neighbor_thred', default=10000.0, type=float)
    parser.add_argument('--nba_stride', default=1, type=int,
                        help='Sliding-window stride for SocialVAE txt scenes.')
    parser.add_argument('--nba_train_limit', default=0, type=int,
                        help='Optional cap for NBA train scenes, useful for smoke tests.')
    parser.add_argument('--nba_test_limit', default=0, type=int,
                        help='Optional cap for NBA test scenes, useful for smoke tests.')
    parser.add_argument('--nba_target_groups', default='PLAYER',
                        help='Comma-separated SocialVAE groups used as prediction targets.')
    parser.add_argument('--nba_context_groups', default='PLAYER,BALL',
                        help='Comma-separated SocialVAE groups kept in the interaction graph.')
    parser.add_argument('--nba_target_indices', default=None,
                        help='Optional comma/range list of MART npy agent indices used as prediction targets, e.g. 0-9.')
    parser.add_argument('--nba_random_rotate', default=False, type=ast.literal_eval,
                        help='NBA-specific rotation augmentation. Default False to preserve court orientation.')
    parser.add_argument('--sdd_data_root', default='./data/stanford',
                        help='Root for SDD pkl data. Expected sdd_train.pkl and sdd_test.pkl.')
    parser.add_argument('--sdd_train_file', default=None,
                        help='Optional explicit SDD train pkl file path.')
    parser.add_argument('--sdd_test_file', default=None,
                        help='Optional explicit SDD test pkl file path.')
    parser.add_argument('--sdd_unit_scale', default=1.0, type=float,
                        help='Divide SDD coordinates by this scale. Default keeps local pkl pixel coordinates.')
    parser.add_argument('--sdd_neighbor_mode', default='full',
                        help='SDD neighbor graph before model pruning: full, distance, or none.')
    parser.add_argument('--sdd_neighbor_thred', default=10000.0, type=float)
    parser.add_argument('--sdd_train_limit', default=0, type=int,
                        help='Optional cap for SDD train scenes, useful for smoke tests.')
    parser.add_argument('--sdd_test_limit', default=0, type=int,
                        help='Optional cap for SDD test scenes, useful for smoke tests.')
    parser.add_argument('--sdd_random_rotate', default=False, type=ast.literal_eval,
                        help='SDD-specific rotation augmentation. Default False to preserve camera orientation.')

    return parser


def load_arg(p):
    # save arg
    if os.path.exists(p.config):
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f) or {}
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                try:
                    assert (k in key)
                except:
                    s = 1
        parser.set_defaults(**default_arg)
        return parser.parse_args()
    else:
        return False


def save_arg(args):
    # save arg
    arg_dict = vars(args)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    with open(args.config, 'w') as f:
        yaml.dump(arg_dict, f)


if __name__ == '__main__':
    parser = get_parser()
    p = parser.parse_args()

    if str(p.dataset).lower() in ('nba', 'sdd'):
        output_key = str(p.dataset).lower()
    else:
        output_key = str(p.test_set)
    p.save_dir = p.save_base_dir + output_key + '/'
    p.model_dir = p.save_base_dir + output_key + '/' + p.train_model + '/'
    p.config = p.model_dir + '/config_' + p.phase + '.yaml'

    if not load_arg(p):
        save_arg(p)

    args = load_arg(p)

    if args.using_cuda and torch.cuda.is_available():
        torch.cuda.set_device(0)
    else:
        args.using_cuda = False

    if str(args.dataset).lower() == 'nba':
        from src.nba_processor import processor
    elif str(args.dataset).lower() == 'sdd':
        from src.sdd_processor import processor
    else:
        from src.processor import processor

    trainer = processor(args)

    if args.phase == 'test':
        trainer.test()
    else:
        trainer.train()
