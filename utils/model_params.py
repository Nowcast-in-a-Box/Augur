UNET_KWARGS = {
    "input_channel": 6,
    "size": 128,
}

DGMR_KWARGS = {
    "forecast_steps": 6,
    "input_channels": 1,
    "output_shape": 128,
    "gen_lr": 5e-5,
    "disc_lr": 2e-4,
    "num_samples": 2,
    "generation_steps": 1,
    "num_context_steps": 6,
    "precip_weight_cap": 5,
}

NOWCASTNET_KWARGS = {
    "input_length": 6,
    "total_length": 12,
    "img_height": 128,
    "img_width": 128,
    "ngf": 32,
    "input_channels": 1,
}

FOURCASTNET_KWARGS = {
    "img_size": (128, 128),
    "in_chans": 6,
    "out_chans": 6,
}

EXPRECAST_KWARGS = {
    "input_frames": 6,
    "output_frames": 6,
    "in_chans": 1,
    "out_chans": 1,
    "depths": [2, 6, 2, 2],
    "learing_rate": 1e-3,
}

PHYDNET_KWARGS = {
    "input_channels": 6,
    "input_height": 128,
    "input_width": 128,
    "learning_rate": 1e-3,
}

MAU_KWARGS = {
    "input_length": 6,
    "total_length": 12,
    "lr": 1e-3,
    "num_layers": 4,
    "delay_interval": 1000,
    "num_hidden": 64,
    "sampling_start_value": 1.0,
    "sampling_stop_iter": 6700,
}

SIMVP_KWARGS = {
    "shape_in": (6, 1, 128, 128),
    "shape_out": (6, 1, 128, 128),
    "lr": 5e-3,
}

CONVLSTM_KWARGS = {
    "nf": 64,
    "in_chan": 1,
    "T": 6,
}

EARTHFARSEER_KWARGS = {
    "shape_in": (6, 1, 128, 128),
    "lr": 1e-3,
}

ALPHAPRE_KWARGS = {
    "aweight_stop_steps": 5000,
    "img_size": 128,
    "T_in": 6,
    "T_out": 6,
}

WADEPRE_KWARGS = {
    "timesteps": 6,
    "spatial_size": 128,
    "loss_a_stop_step": 3000,
    "lr": 1.5e-4,
    "wavelet_level": 3,
    "detail_layer_channels": [64, 128, 256],
    "detail_num_blocks": 4,
    "loss_a_weight": 0.1,
    "loss_a_constant_weight": 0.01,
    "loss_d_weight": 0.05,
    "loss_recon_mean_weight": 0.005,
    "detail_idr_dim": 64,
    "detail_feature_channel": 128,
    "refine_hidden_dim": 6 * 96,
    "approx_hidden_size": 512,
    "approx_cells": 3,
    "dropout_rate": 0.1,
}
