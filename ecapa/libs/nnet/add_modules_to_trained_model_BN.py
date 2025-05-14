
import sys
sys.path.insert(0, 'subtools/pytorch')

import torch
from model.ecapa_tdnn_xivector_uncertainty import ECAPA_TDNN_XiVec_Uncer
from model.ecapa_tdnn_xivector_uncertainty_prior import ECAPA_TDNN_XiVec_Uncer_Prior

in_model_file = "exp/ecapaxiuncer_c1024_pytorchBN/69.params"
out_model_file = "exp/ecapaxiuncer_c1024_pytorchBN/69.prior.params"

model_params = {
    "aug_dropout": 0., "tail_dropout": 0.,
    "training": True, "extracted_embedding": "near",
    "ecapa_params": {"channels": 1024,
                     "embd_dim": 512,
                     "mfa_conv": 1536,
                     "bn_params": {"momentum": 0.5, "affine": True, "track_running_stats": True}},

    # statistics, lde, attentive, multi-head, multi-resolution
    "pooling": "xi-postmean-softplus2-uncer",
    "pooling_params": {
        "num_nodes":1536,
        "num_head":16,
        "share":True,
        "affine_layers":1,
        "hidden_size":256,
        "context":[0],
        "temperature":False, 
        "fixed":True
    },
    "fc1": True,
    "fc1_params": {
        "nonlinearity": '', "nonlinearity_params": {"inplace": True},
        "bn-relu": False,
        "bn": False,
        "bn_params": {"momentum": 0.5, "affine": False, "track_running_stats": True}},

    "fc2_params": {
        "nonlinearity": '', "nonlinearity_params": {"inplace": True},
        "bn-relu": False,
        "bn": True,
        "bn_params": {"momentum": 0.5, "affine": False, "track_running_stats": True}},

    "margin_loss": True,
    "margin_loss_params": {
        "method": "aam", "m": 0.2, "feature_normalize": True,
        "s": 30, "mhe_loss": False, "mhe_w": 0.01},

    "use_step": False,
    "step_params": {
        "margin_warm":False,
        "margin_warm_conf":{"start_epoch":1,"end_epoch":1,"offset_margin":-0.0,"init_lambda":1.0},
        "T": None,
        "m": True, "lambda_0": 0, "lambda_b": 1000, "alpha": 5, "gamma": 1e-4,
        "s": False, "s_tuple": (30, 12), "s_list": None,
        "t": False, "t_tuple": (0.5, 1.2),
        "p": False, "p_tuple": (0.5, 0.1)}
}


model_in = ECAPA_TDNN_XiVec_Uncer(80, 7205, **model_params)

model_out = ECAPA_TDNN_XiVec_Uncer_Prior(80, 7205, **model_params)

model_in.load_state_dict(torch.load(in_model_file, map_location="cpu"))

model_out_dict = model_out.state_dict()

for k, v in model_in.named_parameters():
    if k in model_out_dict:
        print(k)
# 1. filter out unnecessary keys
model_out_in_dict = {k: v for k, v in model_in.named_parameters() if k in model_out_dict}
# 2. overwrite entries in the existing state dict
model_out_dict.update(model_out_in_dict) 
# 3. load the new state dict
model_out.load_state_dict(model_out_dict)

torch.save(model_out.state_dict(), out_model_file)