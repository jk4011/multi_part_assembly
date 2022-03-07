from yacs.config import CfgNode as CN

# Miscellaneous configs
_C = CN()

# Experiment related
_C.exp = CN()
_C.exp.name = ''
_C.exp.ckp_dir = 'checkpoint/'
_C.exp.weight_file = ''
_C.exp.gpus = [
    0,
]
_C.exp.num_workers = 4
_C.exp.batch_size = 16
_C.exp.num_epochs = 200
_C.exp.val_every = 5  # evaluate model every n training epochs
_C.exp.val_sample_vis = 5  # sample visualizations

# Model related
_C.model = CN()
_C.model.encoder = 'pointnet'  # 'dgcnn', 'pointnet2_ssg', 'pointnet2_msg'
_C.model.pc_feat_dim = 512
_C.model.transformer_feat_dim = 1024
_C.model.transformer_heads = 4
_C.model.transformer_layers = 1
_C.model.transformer_pre_ln = True
_C.model.noise_dim = 32  # stochastic PoseRegressor

# Loss related
_C.loss = CN()
_C.loss.sample_iter = 5  # MoN loss sampling
_C.loss.trans_loss_w = 1.
_C.loss.rot_loss = 'l2'  # 'cosine', ''
_C.loss.rot_loss_w = 1.
_C.loss.use_rot_pt_l2_loss = False
_C.loss.rot_pt_l2_loss_w = 1.
_C.loss.use_rot_pt_cd_loss = False
_C.loss.rot_pt_cd_loss_w = 1.
_C.loss.use_transform_pt_cd_loss = False
_C.loss.transform_pt_cd_loss_w = 1.

# Data related
_C.data = CN()
_C.data.data_dir = '../Generative-3D-Part-Assembly/prepare_data'
_C.data.data_fn = 'Chair.{}.npy'
_C.data.data_keys = ('part_ids', 'instance_label', 'match_ids')
_C.data.num_pc_points = 1000
_C.data.max_num_part = 20

# Optimizer related
_C.optimizer = CN()
_C.optimizer.lr = 1e-3
_C.optimizer.weight_decay = 0.
_C.optimizer.warmup_ratio = 0.05
_C.optimizer.clip_grad = None


def get_cfg_defaults():
    return _C.clone()
