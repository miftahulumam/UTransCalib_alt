from dotwiz import DotWiz

config_UTransCalib_resnet_oct4 = {
    'model_name': 'UTransCalib_resnet_oct4',
    'activation': 'nn.ReLU(inplace=True)',
    'init_weights': True
}

config_UTransCalib_lite_oct4 = {
    'model_name': 'UTransCalib_lite_oct4',
    'activation': 'nn.SiLU(inplace=True)',
    'init_weights': True
}

config_UTransCalib_densenet_oct20 = {
    'model_name': 'UTransCalib_densenet_oct20',
    'activation': 'nn.SiLU(inplace=True)',
    'init_weights': True
}

config_UTransCalib_mobilenet_oct24 = {
    'model_name': 'UTransCalib_mobilenet_oct24',

    'rgb_activation': 'nn.SiLU(inplace=True)',
    'depth_activation': 'nn.LeakyReLU(negative_slope=0.1, inplace=True)',
    'fusion_activation': 'nn.SiLU(inplace=True)',
    'regression_activation': 'nn.ReLU(inplace=True)',

    'decoder_drop_rate': 0.1,
    'fusion_reduction': 4,

    'branch_attn_repeat': 1,
    'fusion_attn_repeat': 2,
    
    'head_drop_rate': 0.3,
    
    'init_weights': False
}

config_UTransCalib_mobilenet_nov3_a = {
    'model_name': 'UTransCalib_mobilenet_nov3_a',

    'rgb_activation': 'nn.SiLU(inplace=True)',
    'depth_activation': 'nn.LeakyReLU(negative_slope=0.1, inplace=True)',
    'fusion_activation': 'nn.SiLU(inplace=True)',
    'regression_activation': 'nn.ReLU(inplace=True)',

    'decoder_drop_rate': 0.1,
    'fusion_reduction': 4,

    'branch_attn_repeat': 0,
    'fusion_attn_repeat': 2,

    'head_drop_rate': 0.3,

    'init_weights': False
}

config_UTransCalib_mobilenet_nov3_b = {
    'model_name': 'UTransCalib_mobilenet_nov3_a',

    'rgb_activation': 'nn.SiLU(inplace=True)',
    'depth_activation': 'nn.LeakyReLU(negative_slope=0.1, inplace=True)',
    'fusion_activation': 'nn.SiLU(inplace=True)',
    'regression_activation': 'nn.ReLU(inplace=True)',

    'decoder_drop_rate': 0.1,
    'fusion_reduction': 4,

    'branch_attn_repeat': 0,
    'fusion_attn_repeat': 1,

    'head_drop_rate': 0.3,

    'init_weights': False
}

# ablation
config_UTransCalib_mobilenet_ablation_3s = {
    'model_name': 'UTransCalib_mobilenet_ablation_3s',

    'rgb_activation': 'nn.SiLU(inplace=True)',
    'depth_activation': 'nn.LeakyReLU(negative_slope=0.1, inplace=True)',
    'fusion_activation': 'nn.SiLU(inplace=True)',
    'regression_activation': 'nn.ReLU(inplace=True)',

    'decoder_drop_rate': 0.1,
    'fusion_reduction': 4,

    'branch_attn_repeat': 1,
    'fusion_attn_repeat': 2,

    'head_drop_rate': 0.3,

    'init_weights': False
}

config_UTransCalib_mobilenet_ablation_2s = {
    'model_name': 'UTransCalib_mobilenet_ablation_2s',

    'rgb_activation': 'nn.SiLU(inplace=True)',
    'depth_activation': 'nn.LeakyReLU(negative_slope=0.1, inplace=True)',
    'fusion_activation': 'nn.SiLU(inplace=True)',
    'regression_activation': 'nn.ReLU(inplace=True)',

    'decoder_drop_rate': 0.1,
    'fusion_reduction': 4,

    'branch_attn_repeat': 1,
    'fusion_attn_repeat': 2,

    'head_drop_rate': 0.3,

    'init_weights': False
}

# KICS
config_UTransCalib_KICS = {
    'model_name': 'UTransCalib_KICS_Jan5',

    'input_size': (256, 512),

    'rgb_activation': 'nn.SiLU(inplace=True)',
    'depth_activation': 'nn.LeakyReLU(negative_slope=0.1, inplace=True)',
    'fusion_activation': 'nn.LeakyReLU(negative_slope=0.1, inplace=True)',
    'regression_activation': 'nn.LeakyReLU(negative_slope=0.1, inplace=True)',

    'fusion_patch_size': 9,
    'fusion_channel_stages': [128, 64, 32],
    'fusion_output_reduction': 128,

    'decoder_drop_rate': 0.1,

    'branch_attn_repeat': 1,

    'head_drop_rate': 0.3,

    'init_weights': False
}

config_UTransCalib_KICS_2 = {
    'model_name': 'UTransCalib_KICS_2_Jan8',

    'input_size': (256, 512),

    'rgb_activation': 'nn.SiLU(inplace=True)',
    'depth_activation': 'nn.LeakyReLU(negative_slope=0.1, inplace=True)',
    'fusion_activation': 'nn.LeakyReLU(negative_slope=0.1, inplace=True)',
    'regression_activation': 'nn.LeakyReLU(negative_slope=0.1, inplace=True)',

    'fusion_patch_size': 20,

    'decoder_drop_rate': 0.1,

    'branch_attn_repeat': 0,
    'fusion_attn_repeat': 2,

    'head_drop_rate': 0.3,

    'init_weights': False
}

config_UTransCalib_KICS_rangga = {
    'model_name': 'UTransCalib_KICS_rangga_Jan8',

    'rgb_activation': 'nn.SiLU(inplace=True)',
    'depth_activation': 'nn.LeakyReLU(negative_slope=0.1, inplace=True)',

    'decoder_drop_rate': 0.1,
    
    'md': 4,
}