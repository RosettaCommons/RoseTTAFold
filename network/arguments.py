import argparse
import data_loader

TRUNK_PARAMS = ['n_module', 'n_module_str', 'n_layer', 'd_msa', 'd_pair', 'd_templ',\
                'n_head_msa', 'n_head_pair', 'n_head_templ', 'd_hidden',\
                'r_ff', 'n_resblock', 'p_drop']

SE3_PARAMS = ['num_layers', 'num_channels', 'num_degrees', 'n_heads', 'div', 
              'l0_in_features', 'l0_out_features', 'l1_in_features', 'l1_out_features',
              'num_edge_features']

def get_args():
    parser = argparse.ArgumentParser()

    # training parameters
    train_group = parser.add_argument_group("training parameters")
    train_group.add_argument("-model_name", default=None,
            help="model name for saving")
    train_group.add_argument('-batch_size', type=int, default=1,
            help="Batch size [1]")
    train_group.add_argument('-lr', type=float, default=5.0e-4, 
            help="Learning rate [5.0e-4]")
    train_group.add_argument('-num_epochs', type=int, default=300,
            help="Number of epochs [300]")
    train_group.add_argument("-step_lr", type=int, default=300,
            help="Parameter for Step LR scheduler [300]")
    train_group.add_argument("-port", type=int, default=12319,
            help="PORT for ddp training, should be randomized [12319]")
    train_group.add_argument("-accum", type=int, default=1,
            help="Gradient accumulation when it's > 1 [1]")

    # data-loading parameters
    data_group = parser.add_argument_group("data loading parameters")
    data_group.add_argument('-maxseq', type=int, default=1000,
            help="Maximum depth of subsampled MSA [1000]")
    data_group.add_argument('-maxtoken', type=int, default=2**16,
            help="Maximum depth of subsampled MSA [2**16]")
    data_group.add_argument("-lmin", type=int, default=100,
            help="Lower limit of crop size [100]")
    data_group.add_argument("-lmax", type=int, default=260,
            help="Upper limit of crop size [260]")
    data_group.add_argument("-rescut", type=float, default=3.5,
            help="Resolution cutoff [3.5]")
    data_group.add_argument("-slice", type=str, default="CONT",
            help="How to make crops [CONT (default) / DISCONT]")
    data_group.add_argument("-subsmp", type=str, default="UNI",
            help="How to subsample MSAs [UNI (default) / LOG / CONST]")
    data_group.add_argument('-mintplt', type=int, default=0,
            help="Minimum number of templates to select [0]")
    data_group.add_argument('-maxtplt', type=int, default=10,
            help="maximum number of templates to select [10]")
    data_group.add_argument('-seqid', type=float, default=80.0,
            help="maximum sequence identity cutoff for template selection [80.0]")

    # Trunk module properties
    trunk_group = parser.add_argument_group("Trunk module parameters")
    trunk_group.add_argument('-n_module', type=int, default=4,
            help="Number of iteration blocks without structure [4]")
    trunk_group.add_argument('-n_module_str', type=int, default=4,
            help="Number of iteration blocks with structure [4]")
    trunk_group.add_argument('-n_layer', type=int, default=1,
            help="Number of attention layer for each transformer encoder [1]")
    trunk_group.add_argument('-d_msa', type=int, default=384,
            help="Number of MSA features [384]")
    trunk_group.add_argument('-d_pair', type=int, default=288,
            help="Number of pair features [288]")
    trunk_group.add_argument('-d_templ', type=int, default=64,
            help="Number of templ features [64]")
    trunk_group.add_argument('-n_head_msa', type=int, default=12,
            help="Number of attention heads for MSA2MSA [12]")
    trunk_group.add_argument('-n_head_pair', type=int, default=8,
            help="Number of attention heads for Pair2Pair [8]")
    trunk_group.add_argument('-n_head_templ', type=int, default=4,
            help="Number of attention heads for template [4]")
    trunk_group.add_argument("-d_hidden", type=int, default=64,
            help="Number of hidden features for initial structure builder [64]")
    trunk_group.add_argument("-r_ff", type=int, default=4,
            help="ratio for feed-forward network in transformer encoder [4]")
    trunk_group.add_argument("-n_resblock", type=int, default=1,
            help="Number of residual blocks for MSA2Pair [1]")
    trunk_group.add_argument("-p_drop", type=float, default=0.1,
            help="Dropout ratio [0.1]")
    trunk_group.add_argument("-not_use_perf", action="store_true", default=False,
            help="Use performer or not [False]")

    # Structure module properties
    str_group = parser.add_argument_group("structure module parameters")
    str_group.add_argument('-num_layers', type=int, default=2,
            help="Number of equivariant layers in structure module block [2]")
    str_group.add_argument('-num_channels', type=int, default=32,
            help="Number of channels [32]")
    str_group.add_argument('-num_degrees', type=int, default=2,
            help="Number of degrees for SE(3) network [2]")
    str_group.add_argument('-l0_in_features', type=int, default=32,
            help="Number of type 0 input features [32]")
    str_group.add_argument('-l0_out_features', type=int, default=8,
            help="Number of type 0 output features [8]")
    str_group.add_argument('-l1_in_features', type=int, default=3,
            help="Number of type 1 input features [3]")
    str_group.add_argument('-l1_out_features', type=int, default=3,
            help="Number of type 1 output features [3]")
    str_group.add_argument('-num_edge_features', type=int, default=32,
            help="Number of edge features [32]")
    str_group.add_argument('-n_heads', type=int, default=4,
            help="Number of attention heads for SE3-Transformer [4]")
    str_group.add_argument("-div", type=int, default=4,
            help="Div parameter for SE3-Transformer [4]")

    # Loss function parameters
    loss_group = parser.add_argument_group("loss parameters")
    loss_group.add_argument('-w_dist', type=float, default=1.0,
            help="Weight on distd in loss function [1.0]")
    loss_group.add_argument('-w_str', type=float, default=1.0,
            help="Weight on strd in loss function [1.0]")
    loss_group.add_argument('-w_rms', type=float, default=1.0,
            help="Weight on rmsd in loss function [1.0]")
    loss_group.add_argument('-w_lddt', type=float, default=1.0,
            help="Weight on predicted lddt loss [1.0]")
    loss_group.add_argument('-w_blen', type=float, default=0.1,
            help="Weight on predicted blen loss [0.1]")
    loss_group.add_argument('-w_bang', type=float, default=0.1,
            help="Weight on predicted bang loss [0.1]")

    # parse arguments
    args = parser.parse_args()

    # Setup dataloader parameters:
    loader_param = data_loader.set_data_loader_params(args)

    # make dictionary for each parameters
    trunk_param = {}
    for param in TRUNK_PARAMS:
        trunk_param[param] = getattr(args, param)
    if not args.not_use_perf:
        trunk_param["performer_N_opts"] = {"nb_features": 64, "feature_redraw_interval": 10000}
        trunk_param["performer_L_opts"] = {"nb_features": 64, "feature_redraw_interval": 10000}
    SE3_param = {}
    for param in SE3_PARAMS:
        if hasattr(args, param):
            SE3_param[param] = getattr(args, param)
    trunk_param['SE3_param'] = SE3_param 
    loss_param = {}
    for param in ['w_dist', 'w_str', 'w_rms', 'w_lddt', 'w_blen', 'w_bang']:
        loss_param[param] = getattr(args, param)

    return args, trunk_param, loader_param, loss_param
