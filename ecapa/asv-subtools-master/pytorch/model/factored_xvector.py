# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2020-02-05)
#                              Hao Lu  2020-09-16)
import math
from warnings import resetwarnings
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, "subtools/pytorch")
from libs.nnet import *
import libs.support.utils as utils

class Xvector(TopVirtualNnet):
    """ A factored x-vector framework """

    def init(self, inputs_dim, num_targets, nonlinearity="relu", semi_orth=True,embd_dim=512,
             aug_dropout=0.2, training=False, extracted_embedding="far", jit_compile=False):

        # Var
        self.semi_orth = semi_orth
        self.extracted_embedding = extracted_embedding

        if self.semi_orth:
            self.use_step = True

        # Nnet
        self.aug_dropout = torch.nn.Dropout2d(
            p=aug_dropout) if aug_dropout > 0 else None

        self.layer01 = ReluBatchNormTdnnLayer(
            inputs_dim, 512, [-2, -1, 0, 1, 2], nonlinearity=nonlinearity, jit_compile=jit_compile)
        # FTdnnBlock(input_dim,output_dim,bottleneck_dim,context_size,bypass_scale)
        self.layer02 = FTdnnBlock(512, 1024, 256, 2, 0)
        self.layer03 = FTdnnBlock(1024, 1024, 256, 0, 0.66)
        self.layer04 = FTdnnBlock(1024, 1024, 256, 3, 0.66)
        self.layer05 = FTdnnBlock(1024, 1024, 256, 0, 0.66)
        self.layer06 = FTdnnBlock(1024, 1024, 256, 3, 0.66)
        self.layer07 = FTdnnBlock(2048, 1024, 256, 3, 0)
        self.layer08 = FTdnnBlock(1024, 1024, 256, 3, 0.66)
        self.layer09 = FTdnnBlock(3072, 1024, 256, 0, 0)

        self.layer10 = ReluBatchNormTdnnLayer(
            1024, 2048, nonlinearity=nonlinearity, jit_compile=jit_compile)

        self.stats = StatisticsPooling(2048, stddev=True)

        self.embedding1 = ReluBatchNormTdnnLayer(self.stats.get_output_dim(
        ), embd_dim, nonlinearity=nonlinearity, jit_compile=jit_compile)
        self.embedding2 = ReluBatchNormTdnnLayer(
            embd_dim, embd_dim, nonlinearity=nonlinearity, jit_compile=jit_compile)
        self.embd_dim=embd_dim
        # Do not need when extracting embedding.
        if training:
            self.loss = SoftmaxLoss(512, num_targets)

            # An example to using transform-learning without initializing loss.affine parameters
            self.transform_keys = ["layer01", "layer02", "layer03", "layer04", "layer05",
                                   "layer06", "layer07", "layer08", "layer09", "layer10",
                                   "stats", "embedding1", "embedding2"]

    @torch.jit.unused
    @utils.for_device_free
    def forward(self, inputs, x_len: torch.Tensor=torch.empty(0)):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index,  frames-dim-index, frames-index]
        """
        x = inputs
        x = self.auto(self.aug_dropout, x)

        x_1 = self.layer01(x)
        x_2 = self.layer02(x_1)
        x_3 = self.layer03(x_2)
        x_4 = self.layer04(x_3)
        x_5 = self.layer05(x_3)
        x_6 = self.layer06(x_5)
        x_7 = self.layer07(torch.cat((x_2, x_4), 1))
        x_8 = self.layer08(x_7)
        x_9 = self.layer09(torch.cat((x_4, x_6, x_8), 1))
        x = self.layer10(x_9)
        x = self.stats(x)
        x = self.embedding1(x)
        outputs = self.embedding2(x)

        return outputs

    @utils.for_device_free
    def get_loss(self, inputs, targets):
        """Should call get_loss() after forward() with using Xvector model function.
        e.g.:
            m=Xvector(20,10)
            loss=m.get_loss(m(inputs),targets)
        """
        return self.loss(inputs, targets)

    def get_posterior(self):
        """Should call get_posterior after get_loss. This function is to get outputs from loss component.
        @return: return posterior
        """
        return self.loss.get_posterior()

    @for_extract_embedding(maxChunk=10000, isMatrix=True)
    def extract_embedding(self, inputs):
        """
        inputs: a 3-dimensional tensor with batch-dim = 1 or normal features matrix
        return: an 1-dimensional vector after processed by decorator
        """

        x = inputs
        x_1 = self.layer01(x)
        x_2 = self.layer02(x_1)
        x_3 = self.layer03(x_2)
        x_4 = self.layer04(x_3)
        x_5 = self.layer05(x_3)
        x_6 = self.layer06(x_5)
        x_7 = self.layer07(torch.cat((x_2, x_4), 1))
        x_8 = self.layer08(x_7)
        x_9 = self.layer09(torch.cat((x_4, x_6, x_8), 1))
        x = self.layer10(x_9)
        x = self.stats(x)

        if self.extracted_embedding == "far":
            xvector = self.embedding1.affine(x)
        elif self.extracted_embedding == "near":
            x = self.embedding1(x)
            xvector = self.embedding2.affine(x)

        return xvector

   

  
    
    def extract_embedding_jit(self, inputs: torch.Tensor, position: str = 'far') -> torch.Tensor:
        """
        inputs: a 3-dimensional tensor with batch-dim = 1 or  2-dim matrix tensor
        return: an 1-dimensional vector after processed by decorator
        """
        x = inputs
        x_1 = self.layer01(x)
        x_2 = self.layer02(x_1)
        x_3 = self.layer03(x_2)
        x_4 = self.layer04(x_3)
        x_5 = self.layer05(x_3)
        x_6 = self.layer06(x_5)
        x_7 = self.layer07(torch.cat((x_2, x_4), 1))
        x_8 = self.layer08(x_7)
        x_9 = self.layer09(torch.cat((x_4, x_6, x_8), 1))
        x = self.layer10(x_9)
        x = self.stats(x)

        if position == "far":
            xvector = self.embedding1.affine(x)
        else:
            x = self.embedding1(x)
            xvector = self.embedding2.affine(x)

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


    def step(self, epoch, this_iter, epoch_batchs):
        if self.semi_orth:
            self.step_semi_orth(this_iter)

    def step_semi_orth(self, this_iter):
        if int(this_iter) % 4 == 0:
            self.layer02.step_semi_orth()
            self.layer03.step_semi_orth()
            self.layer04.step_semi_orth()
            self.layer05.step_semi_orth()
            self.layer06.step_semi_orth()
            self.layer07.step_semi_orth()
            self.layer08.step_semi_orth()
            self.layer09.step_semi_orth()



if __name__ == "__main__":
    model = Xvector(41,11942,semi_orth=True,nonlinearity='relu',aug_dropout=0.2,training=False,extracted_embedding='far',jit_compile=True)
    print(model)
   