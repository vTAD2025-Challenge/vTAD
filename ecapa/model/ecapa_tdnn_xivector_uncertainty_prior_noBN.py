
# Copyright xmuspeech (Author: Leo 2022-05-27)
# refs:
# 1.  ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification
#           https://arxiv.org/abs/2005.07143

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
sys.path.insert(0, 'subtools/pytorch')
import libs.support.utils as utils
from libs.nnet import *


class Res2NetBlock(torch.nn.Module):
    """An implementation of Res2NetBlock w/ dilation.

    Arguments
    ---------
    in_channels : int
        The number of channels expected in the input.
    out_channels : int
        The number of output channels.
    scale : int
        The scale of the Res2Net block.
    kernel_size: int
        The kernel size of the Res2Net block.
    dilation : int
        The dilation of the Res2Net block.

    Example
    -------
    >>> inp_tensor = torch.rand([8, 120, 64]).transpose(1, 2)
    >>> layer = Res2NetBlock(64, 64, scale=4, dilation=3)
    >>> out_tensor = layer(inp_tensor).transpose(1, 2)
    >>> out_tensor.shape
    torch.Size([8, 120, 64]) 
    """

    def __init__(
        self, in_channels, out_channels, scale=8, kernel_size=3, dilation=1, bn_params={}
    ):
        super(Res2NetBlock, self).__init__()
        assert in_channels % scale == 0
        assert out_channels % scale == 0
        assert scale>1
        in_channel = in_channels // scale
        hidden_channel = out_channels // scale
        context = [i for i in range(-(kernel_size//2) *
                                    dilation, (kernel_size//2)*dilation+1, dilation)]
        self.blocks = nn.ModuleList(
            [
                ReluBatchNormTdnnLayer(in_channel, hidden_channel, context,**bn_params) for i in range(scale - 1)
            ]
        )
 
        self.scale = scale

    def forward(self, x):
        y = []
        spx = torch.chunk(x,self.scale,dim=1)
        sp = spx[0]
        y.append(sp)
        for i,block in enumerate(self.blocks):
            if i == 0:
                sp = spx[i+1]
            if i>=1:
                sp = sp + spx[i+1]
            sp = block(sp)
            y.append(sp)

        y = torch.cat(y, dim=1)
        return y


''' The SE connection of 1D case.
'''
# class SE_Connect(nn.Module):
#     def __init__(self, channels, s=4):
#         super().__init__()
#         assert channels % s == 0, "{} % {} != 0".format(channels, s)
#         self.linear1 = nn.Linear(channels, channels // s)
#         self.linear2 = nn.Linear(channels // s, channels)

#     def forward(self, x):
#         out = x.mean(dim=2)
#         out = F.relu(self.linear1(out))
#         out = torch.sigmoid(self.linear2(out))
#         out = x * out.unsqueeze(2)
#         return out

# Another implementation of SE_Connect


class SE_Connect(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SE_Connect, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return input * x


''' SE-Res2Block.
'''


class SE_Res2Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, scale=8, bn_params={}):
        super(SE_Res2Block, self).__init__()
        width = int(math.floor(in_channels / scale))

        self.conv_relu_bn1 = ReluBatchNormTdnnLayer(
            in_channels, width*scale, **bn_params)

        self.res2net_block = Res2NetBlock(
            width*scale, width*scale, scale=scale, kernel_size=kernel_size, dilation=dilation,bn_params=bn_params)
        self.conv_relu_bn2 = ReluBatchNormTdnnLayer(
            in_channels, width*scale, **bn_params)
        self.se = SE_Connect(out_channels)
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        x = self.conv_relu_bn1(x)

        x = self.res2net_block(x)

        x = self.conv_relu_bn2(x)
        x = self.se(x)
        return x+residual


''' Attentive weighted mean and standard deviation pooling.
'''


class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, bottleneck_dim=128, time_attention=False, bn={}):
        super().__init__()
        self.time_attention = time_attention
        accept_dim = in_dim
        # Use Conv1d with stride == 1 rather than Linear, then we don't need to transpose inputs.
        if time_attention:
            accept_dim = in_dim*3
        self.attention = nn.Sequential(
            nn.Conv1d(accept_dim, bottleneck_dim, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(bottleneck_dim, **bn),
            nn.Tanh(),
            nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x):

        if self.time_attention:
            global_mean = torch.mean(x, dim=2, keepdim=True).expand_as(x)
            global_std = torch.sqrt(
                torch.var(x, dim=-1, keepdim=True) + 1e-5).expand_as(x)
            x_in = torch.cat((x, global_mean, global_std), dim=1)

        else:
            x_in = x

        alpha = self.attention(x_in)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * (x ** 2), dim=2) - mean ** 2
        std = torch.sqrt(residuals.clamp(min=1e-5))
        return torch.cat([mean, std], dim=1)


''' Implementation of
    "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification".

    Note that we DON'T concatenate the last frame-wise layer with non-weighted mean and standard deviation, 
    because it brings little improvment but significantly increases model parameters. 
    As a result, this implementation basically equals the A.2 of Table 2 in the paper.
'''


class ECAPA_TDNN_XiVec_Uncer_Prior_noBN(TopVirtualNnet):
    def init(self, inputs_dim, num_targets, aug_dropout=0., tail_dropout=0., training=True,
             extracted_embedding="near", mixup=False, mixup_alpha=1.0, pooling="ecpa-attentive", pooling_params={},
             ecapa_params={}, fc1=False, fc1_params={}, fc2_params={},
             margin_loss=True, margin_loss_params={}, use_step=False, step_params={}, transfer_from="softmax_loss"):

        default_ecapa_params = {
            "channels": 1024,
            "embd_dim": 512,
            "mfa_conv": 1536,
            "bn_params": {"momentum": 0.5, "affine": True, "track_running_stats": True}
        }

        prior_ecapa_params1 = {
            "nonlinearity": 'relu', "nonlinearity_params": {"inplace": True},
            "bn-relu": True,
            "bn": False,
            "bn_params": {"momentum": 0.5, "affine": False, "track_running_stats": True}
        }

        prior_ecapa_params2 = {
            "nonlinearity": '', "nonlinearity_params": {"inplace": True},
            "bn-relu": False,
            "bn": False,
            "bn_params": {"momentum": 0.5, "affine": False, "track_running_stats": True}
        }

        default_pooling_params = {
            "hidden_size": 128,
            "time_attention": True,
            "stddev": True,
        }

        default_fc_params = {
            "nonlinearity": 'relu', "nonlinearity_params": {"inplace": True},
            "bn-relu": False,
            "bn": True,
            "bn_params": {"momentum": 0.5, "affine": True, "track_running_stats": True}
        }

        default_margin_loss_params = {
            "method": "am", "m": 0.2,
            }

        default_step_params = {
            "margin_warm":False,
            "margin_warm_conf":{"start_epoch":5.,"end_epoch":10.,"offset_margin":-0.2,"init_lambda":0.0},
            "T": None,
            "m": False, "lambda_0": 0, "lambda_b": 1000, "alpha": 5, "gamma": 1e-4,
            "s": False, "s_tuple": (30, 12), "s_list": None,
            "t": False, "t_tuple": (0.5, 1.2),
            "p": False, "p_tuple": (0.5, 0.1)
        }

        self.use_step = use_step
        self.step_params = step_params
        self.extracted_embedding = extracted_embedding

        ecapa_params = utils.assign_params_dict(
            default_ecapa_params, ecapa_params)

        pooling_params = utils.assign_params_dict(
            default_pooling_params, pooling_params,support_unknow=True)
        fc1_params = utils.assign_params_dict(default_fc_params, fc1_params)
        fc2_params = utils.assign_params_dict(default_fc_params, fc2_params)
        margin_loss_params = utils.assign_params_dict(
            default_margin_loss_params, margin_loss_params,support_unknow=True)
        step_params = utils.assign_params_dict(
            default_step_params, step_params)
        embd_dim = ecapa_params["embd_dim"]
        self.embd_dim = embd_dim

        self.speaker_prior_mean = torch.nn.Parameter(torch.randn(1, 128, 1))
        self.speaker_prior_cov = torch.nn.Parameter(torch.randn(1, 128, 1))
        self.priormean1_relu_bn = ReluBatchNormTdnnLayer(128, embd_dim, [0], **prior_ecapa_params1)
        self.priormean1_1_relu_bn = ReluBatchNormTdnnLayer(embd_dim, embd_dim, [0], **prior_ecapa_params1)
        self.priormean2_relu_bn = ReluBatchNormTdnnLayer(embd_dim, embd_dim, [0], **prior_ecapa_params2)
        self.priorcov1_relu_bn = ReluBatchNormTdnnLayer(128, embd_dim, [0], **prior_ecapa_params1)
        self.priorcov1_1_relu_bn = ReluBatchNormTdnnLayer(embd_dim, embd_dim, [0], **prior_ecapa_params1)
        self.priorcov2_relu_bn = ReluBatchNormTdnnLayer(embd_dim, embd_dim, [0], **prior_ecapa_params1)
        self.priorcov3 = ReluBatchNormTdnnLayer(embd_dim, embd_dim, [0], **prior_ecapa_params2)
        self.softplus_priorcov = torch.nn.Softplus(beta=1, threshold=20)

        channels = ecapa_params["channels"]
        self.mixup = Mixup(alpha=mixup_alpha) if mixup else None
        self.layer1 = ReluBatchNormTdnnLayer(
            inputs_dim, channels, [-2, -1, 0, 1, 2], **ecapa_params)
        
        self.layer2 = SE_Res2Block(
            channels, channels, kernel_size=3, dilation=2, scale=8, bn_params=ecapa_params)
        self.layer3 = SE_Res2Block(
            channels, channels, kernel_size=3, dilation=3, scale=8, bn_params=ecapa_params)
        self.layer4 = SE_Res2Block(
            channels, channels, kernel_size=3, dilation=4, scale=8, bn_params=ecapa_params)
        cat_channels = channels * 3
        mfa_conv = ecapa_params["mfa_conv"]
        self.mfa = ReluBatchNormTdnnLayer(
            cat_channels, mfa_conv, **ecapa_params)

        self.manual_BN = False

        # Pooling
        stddev = pooling_params.pop("stddev")
        if pooling == "attentive":
            self.stats = AttentiveStatisticsPooling(
                mfa_conv, hidden_size=pooling_params["hidden_size"], context=pooling_params["context"], stddev=stddev)
            self.bn_stats = nn.BatchNorm1d(
                mfa_conv * 2, **ecapa_params["bn_params"])
            self.fc1 = ReluBatchNormTdnnLayer(
                mfa_conv * 2, embd_dim, **fc1_params) if fc1 else None
        elif pooling == "ecpa-attentive":
            self.stats = AttentiveStatsPool(
                mfa_conv, pooling_params["hidden_size"], pooling_params["time_attention"])
            self.bn_stats = nn.BatchNorm1d(
                mfa_conv * 2, **ecapa_params["bn_params"])
            self.fc1 = ReluBatchNormTdnnLayer(
                mfa_conv * 2, embd_dim, **fc1_params) if fc1 else None
        elif pooling == "mqmha":
            self.stats = MQMHASP(
                mfa_conv, **pooling_params)
            self.bn_stats = nn.BatchNorm1d(
                self.stats.get_output_dim(), **ecapa_params["bn_params"])
            self.fc1 = ReluBatchNormTdnnLayer(
                self.stats.get_output_dim(), embd_dim, **fc1_params) if fc1 else None
        elif pooling == "multi-head":
            self.stats = MultiHeadAttentionPooling(
                mfa_conv, stddev=stddev, **pooling_params)
            self.bn_stats = nn.BatchNorm1d(
                mfa_conv * 2, **ecapa_params["bn_params"])
            self.fc1 = ReluBatchNormTdnnLayer(
                mfa_conv * 2, embd_dim, **fc1_params) if fc1 else None
        elif pooling == "global-multi":
            self.stats = GlobalMultiHeadAttentionPooling(
                mfa_conv, stddev=stddev, **pooling_params)
            self.bn_stats = nn.BatchNorm1d(
                mfa_conv * 2 * pooling_params["num_head"], **ecapa_params["bn_params"])
            self.fc1 = ReluBatchNormTdnnLayer(
                mfa_conv * 2 * pooling_params["num_head"], embd_dim, **fc1_params) if fc1 else None
        elif pooling == "multi-resolution":
            self.stats = MultiResolutionMultiHeadAttentionPooling(
                mfa_conv, **pooling_params)
            self.bn_stats = nn.BatchNorm1d(
                mfa_conv * 2 * pooling_params["num_head"], **ecapa_params["bn_params"])
            self.fc1 = ReluBatchNormTdnnLayer(
                mfa_conv * 2 * pooling_params["num_head"], embd_dim, **fc1_params) if fc1 else None
        elif pooling == "xi-postmean-softplus2":
            self.stats = xivec_stdinit_softplus2_prec_pooling(pooling_params["num_head"], hidden_size=pooling_params["hidden_size"], stddev=False)
            stats_dim = self.stats.get_output_dim()
            self.fc1 = ReluBatchNormTdnnLayer(stats_dim, embd_dim, **fc1_params) if fc1 else None
        elif pooling == "xi-postdist-softplus2":
            self.stats = xivec_stdinit_softplus2_prec_pooling(pooling_params["num_nodes"], hidden_size=pooling_params["hidden_size"], stddev=True)
            stats_dim = self.stats.get_output_dim()
            self.fc1 = ReluBatchNormTdnnLayer(
                stats_dim, embd_dim, **fc1_params) if fc1 else None
        elif pooling == "xi-postmean-softplus2-uncer":
            self.stats = xivec_stdinit_softplus2_prec_uncertainty_pooling(pooling_params["num_nodes"], hidden_size=pooling_params["hidden_size"], stddev=False)
            stats_dim = self.stats.get_output_dim()
            # self.bn_stats = nn.BatchNorm1d(stats_dim, **ecapa_params["bn_params"])
            # self.manual_BN = True
            self.fc1 = ReluBatchNormTdnnLayer(
                stats_dim, embd_dim, **fc1_params) if fc1 else None
        elif pooling == "xi-postdist-softplus2-uncer":
            self.stats = xivec_stdinit_softplus2_prec_uncertainty_pooling(pooling_params["num_nodes"], hidden_size=pooling_params["hidden_size"], stddev=True)
            stats_dim = self.stats.get_output_dim()
            # self.bn_stats = nn.BatchNorm1d(stats_dim, **ecapa_params["bn_params"])
            # self.manual_BN = True
            self.fc1 = ReluBatchNormTdnnLayer(
                stats_dim, embd_dim, **fc1_params) if fc1 else None
        else:
            self.stats = StatisticsPooling(mfa_conv, stddev=stddev)
            self.bn_stats = nn.BatchNorm1d(mfa_conv * 2)
            self.fc1 = ReluBatchNormTdnnLayer(
                mfa_conv * 2, embd_dim, **fc1_params) if fc1 else None

        self.tail_dropout = torch.nn.Dropout2d(
            p=tail_dropout) if tail_dropout > 0 else None

        if fc1:
            fc2_in_dim = embd_dim
        else:
            fc2_in_dim = mfa_conv * 2
            if pooling == "mqmha":
                fc2_in_dim = self.stats.get_output_dim()
        self.fc2 = ReluBatchNormTdnnLayer(fc2_in_dim, embd_dim, **fc2_params)
        self.tail_dropout = torch.nn.Dropout2d(
            p=tail_dropout) if tail_dropout > 0 else None
        # print("num_targets---------------",num_targets)
        # Loss
        # Do not need when extracting embedding.
        if training:
            if margin_loss:
                self.loss = MarginSoftmaxLoss_v1(
                    embd_dim, num_targets, **margin_loss_params)
                if self.use_step and self.step_params["margin_warm"]:
                    self.margin_warm = MarginWarm(**step_params["margin_warm_conf"])
            else:
                self.loss = SoftmaxLoss(embd_dim, num_targets)
                # self.loss = AngleLoss(embd_dim,num_targets)
            self.wrapper_loss = MixupLoss(
                self.loss, self.mixup) if mixup else None
            # An example to using transform-learning without initializing loss.affine parameters
            self.transform_keys = ["layer1", "layer2", "layer3",
                                   "layer4", "stats", "mfa", "bn_stats", "fc1", "fc2", "loss"]

            if margin_loss and transfer_from == "softmax_loss":
                # For softmax_loss to am_softmax_loss
                self.rename_transform_keys = {
                    "loss.affine.weight": "loss.weight"}
                
            self.prior_kld_loss = nn.KLDivLoss(reduction="batchmean")

    @torch.jit.unused
    @utils.for_device_free
    def forward(self, x, x_len: torch.Tensor=torch.empty(0)):
        x = self.layer1(x)
        x1 = self.layer2(x)
        x2 = self.layer3(x+x1)
        x3 = self.layer4(x+x1+x2)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.mfa(x)
        # x = self.bn_stats(self.stats(x))
        y = self.stats(x)
        cov = None
        if len(y) == 2:
            x = y[0]
            cov = y[1]
            if len(cov.shape) == 3:
                cov = cov.squeeze(dim=2)
        else:
            x = y

        if len(x.shape) != 3:
            x = x.unsqueeze(dim=2)

        x = self.auto(self.fc1, x)
        post_mean = x.squeeze(dim=2)
        
        if cov is not None:
            cov = torch.diag_embed(cov, dim1 = 1)
            fc1_weight = self.fc1.affine.weight.squeeze(dim = 2)
            cov = torch.matmul(torch.matmul(fc1_weight, cov), fc1_weight.t())
            cov = torch.diagonal(cov, dim1=1, dim2=2)

        x = self.fc2(x)
        x = self.auto(self.tail_dropout, x)
        return [x, post_mean, cov]

    @utils.for_device_free
    def get_prior_kld_loss(self, postmean, postcov):

        prior_mean = self.priormean1_relu_bn(self.speaker_prior_mean)
        prior_mean = self.priormean1_1_relu_bn(prior_mean)
        prior_mean = self.priormean2_relu_bn(prior_mean)
        prior_cov = self.priorcov1_relu_bn(self.speaker_prior_cov)
        prior_cov = self.priorcov1_1_relu_bn(prior_mean)
        prior_cov = self.priorcov2_relu_bn(prior_cov)
        prior_cov = self.priorcov3(prior_cov)
        prior_cov = self.softplus_priorcov(prior_cov)

        prior_mean = prior_mean.squeeze(dim=2)
        prior_cov = prior_cov.squeeze(dim=2)

        print("mean")
        print(torch.mean(postmean, dim=0))
        print("prior_mean")
        print(prior_mean)
        print("postcov")
        print(postcov)
        print("prior_cov")
        print(prior_cov)

        # print(prior_mean)
        # print(prior_cov)

        # print(postmean[0])
        # print(postcov[0])

        kld_loss = 0.5*(torch.sum(1/postcov*prior_cov,dim=1) - postcov.shape[1] + torch.sum((postmean - prior_mean) * 1/postcov * (postmean - prior_mean),dim=1) + torch.sum(torch.log(postcov), dim=1) - torch.sum(torch.log(prior_cov), dim=1))
                        
        kld_loss = torch.mean(kld_loss)

        return kld_loss

    @utils.for_device_free
    def get_loss(self, inputs, targets):
        """Should call get_loss() after forward() with using Xvector model function.
        e.g.:
            m=Xvector(20,10)
            loss=m.get_loss(m(inputs),targets)

        model.get_loss [custom] -> loss.forward [custom]
          |
          v
        model.get_accuracy [custom] -> loss.get_accuracy [custom] -> loss.compute_accuracy [static] -> loss.predict [static]
        """
        if self.wrapper_loss is not None:
            return self.wrapper_loss(inputs, targets)
        else:
            return self.loss(inputs, targets)

    @utils.for_device_free
    def get_accuracy(self, targets):
        """Should call get_accuracy() after get_loss().
        @return: return accuracy
        """
        if self.wrapper_loss is not None:
            return self.wrapper_loss.get_accuracy(targets)
        else:
            return self.loss.get_accuracy(targets)

    @for_extract_embedding(maxChunk=10000, isMatrix=True)
    def extract_embedding(self, x):
        x = self.layer1(x)
        x1 = self.layer2(x)
        x2 = self.layer3(x+x1)
        x3 = self.layer4(x+x1+x2)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.mfa(x)
        # x = self.bn_stats(self.stats(x))
        x = self.stats(x)
        if len(x.shape) != 3:
            x = x.unsqueeze(dim=2)
        if self.extracted_embedding == "far":
            assert self.fc1 is not None
            xvector = self.fc1.affine(x)
        elif self.extracted_embedding == "near_affine":
            x = self.auto(self.fc1, x)
            xvector = self.fc2.affine(x)
        elif self.extracted_embedding == "near":
            x = self.auto(self.fc1, x)
            xvector = self.fc2(x)
        else:
            raise TypeError("Expected far or near position, but got {}".format(
                self.extracted_embedding))
        return xvector

    def extract_embedding_jit(self, x: torch.Tensor, position: str = 'near') -> torch.Tensor:

        x = self.layer1(x)
        x1 = self.layer2(x)
        x2 = self.layer3(x+x1)
        x3 = self.layer4(x+x1+x2)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.mfa(x)
        # x = self.bn_stats(self.stats(x))
        y = self.stats(x)
        if len(y) == 1:
            x = y[0]
        else:
            x = y[0]
        if len(x.shape) != 3:
            x = x.unsqueeze(dim=2)
        
        if self.manual_BN:
            bn_mean = torch.mean(x, dim=0)
            bn_cov = torch.var(x.squeeze(dim=2), dim=0)
            x = (x-bn_mean) / torch.sqrt(bn_cov)

        if position == "far" and self.fc1 is not None:
            xvector = self.fc1.affine(x)
        elif position == "near_affine":
            if self.fc1 is not None:
                x = self.fc1(x)
            xvector = self.fc2.affine(x)
        elif position == "near":
            if self.fc1 is not None:
                x = self.fc1(x)
            xvector = self.fc2(x)
        else:
            raise TypeError("Expected far or near position, but got {}".format(
                self.extracted_embedding))
        return xvector

    @torch.jit.export
    def extract_embedding_whole(self, input: torch.Tensor, position: str = 'near', maxChunk: int = 4000, isMatrix: bool = True):
        with torch.no_grad():
            if isMatrix:
                input = torch.unsqueeze(input, dim=0)
                input = input.transpose(1, 2)
            num_frames = input.shape[2]
            num_split = (num_frames + maxChunk - 1) // maxChunk
            split_size = num_frames // num_split
            offset = 0
            embedding_stats = torch.zeros(1, self.embd_dim, 1).to(input.device)
            for _ in range(0, num_split-1):
                this_embedding = self.extract_embedding_jit(
                    input[:, :, offset:offset+split_size], position)
                offset += split_size
                embedding_stats += split_size*this_embedding

            last_embedding = self.extract_embedding_jit(
                input[:, :, offset:], position)

            embedding = (embedding_stats + (num_frames-offset)
                        * last_embedding) / num_frames
            return torch.squeeze(embedding.transpose(1, 2)).cpu()

    @torch.jit.export
    def embedding_dim(self) -> int:
        """ Export interface for c++ call, return embedding dim of the model
        """
        return self.embd_dim

    def get_warmR_T(self, T_0, T_mult, epoch):
        n = int(math.log(max(0.05, (epoch / T_0 * (T_mult - 1) + 1)), T_mult))
        T_cur = epoch - T_0 * (T_mult ** n - 1) / (T_mult - 1)
        T_i = T_0 * T_mult ** (n)
        return T_cur, T_i

    def compute_decay_value(self, start, end, T_cur, T_i):
        # Linear decay in every cycle time.
        return start - (start - end)/(T_i-1) * (T_cur % T_i)

    def step(self, epoch, this_iter, epoch_batchs):
        # Heated up for t and s.
        # Decay for margin and dropout p.
        if self.use_step:
            if self.step_params["m"]:
                current_postion = epoch*epoch_batchs + this_iter
                lambda_factor = max(self.step_params["lambda_0"],
                                    self.step_params["lambda_b"]*(1+self.step_params["gamma"]*current_postion)**(-self.step_params["alpha"]))
                lambda_m = 1/(1 + lambda_factor)
                self.loss.step(lambda_m)

            if self.step_params["T"] is not None and (self.step_params["t"] or self.step_params["p"]):
                T_cur, T_i = self.get_warmR_T(*self.step_params["T"], epoch)
                T_cur = T_cur*epoch_batchs + this_iter
                T_i = T_i * epoch_batchs

            if self.step_params["t"]:
                self.loss.t = self.compute_decay_value(
                    *self.step_params["t_tuple"], T_cur, T_i)

            if self.step_params["p"]:
                self.aug_dropout.p = self.compute_decay_value(
                    *self.step_params["p_tuple"], T_cur, T_i)

            if self.step_params["s"]:
                self.loss.s = self.step_params["s_tuple"][self.step_params["s_list"][epoch]]

    def step_iter(self, epoch, cur_step):
        # For iterabledataset
        if self.use_step:
            if self.step_params["margin_warm"]:
                offset_margin, lambda_m = self.margin_warm.step(cur_step)
                lambda_m =  max(1e-3,lambda_m)
                self.loss.step(lambda_m,offset_margin)
            if self.step_params["m"]:
                lambda_factor = max(self.step_params["lambda_0"],
                                    self.step_params["lambda_b"]*(1+self.step_params["gamma"]*cur_step)**(-self.step_params["alpha"]))
                lambda_m = 1/(1 + lambda_factor)
                self.loss.step(lambda_m)

            if self.step_params["T"] is not None and (self.step_params["t"] or self.step_params["p"]):
                T_cur, T_i = self.get_warmR_T(*self.step_params["T"], cur_step)

            if self.step_params["t"]:
                self.loss.t = self.compute_decay_value(
                    *self.step_params["t_tuple"], T_cur, T_i)

            if self.step_params["p"]:
                self.aug_dropout.p = self.compute_decay_value(
                    *self.step_params["p_tuple"], T_cur, T_i)

            if self.step_params["s"]:
                self.loss.s = self.step_params["s_tuple"][self.step_params["s_list"][epoch]]


if __name__ == '__main__':
    # Input size: batch_size * seq_len * feat_dim
    timer = utils.Timer()
    x = torch.zeros(2,80, 1000)

    model = ECAPA_TDNN(inputs_dim=26, num_targets=1211,training=False)
    out = model(x)
    total = sum(p.numel() for p in model.parameters())
    print(model)
    print(total)
    print(out.shape)    # should be [2, 192]
