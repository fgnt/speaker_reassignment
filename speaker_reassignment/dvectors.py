"""
Resnet taken from torchvision.models
See https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
"""

import numpy as np
import torch.nn as nn
from einops import einops

import paderbox as pb
import padertorch as pt
from padertorch.contrib.je.modules.reduce import Mean
from padertorch.contrib.je.modules.conv import CNN2d, Conv2d


class ResNet34(pt.Module):
    """
    The basic Speaker ID neural network based on ResNet.
    """
    def __init__(
            self,
            in_channels=1,
            channels=(64, 128, 256, 256),
            dvec_dim=256,
            activation_fn='relu',
            norm='batch',
            pre_activation=True,
    ):
        super().__init__()
        # ResNet18
        out_channels = 3*2*[channels[0]] + 4*2*[channels[1]] + 6*2*[channels[2]] + 3*2*[channels[3]]
        assert len(out_channels) == 32, len(out_channels)

        # ResNet34
        #        out_channels = [channels[0]] + 3*3*[channels[1]] + 4*3*[channels[2]] + 2*3*[channels[3]]

        kernel_size = 32*[3]
        stride = 3*2*[(1,1)] + [(2,2)] + (4*2 - 1)*[(1,1)] + 6*2*[(1,1)] + [(2,1)] + (3*2 -1)*[(1,1)]
        pool_size = 32 * [1]
        pool_stride = 32 * [1]
        pool_type = 32 * [None]
        residual_connections = 32 * [None]
        for i in range(0, 32, 2):
            residual_connections[i] = i+2
        norm = norm
        self.embedding_dim = dvec_dim
        self.input_convolution = Conv2d(in_channels, channels[0], kernel_size=3, stride=2, bias=False, norm=norm)
        self.resnet = CNN2d(
            input_layer=False,
            output_layer=False,
            in_channels=channels[0],
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            pool_size=pool_size,
            pool_stride=pool_stride,
            pool_type=pool_type,
            residual_connections=residual_connections,
            activation_fn=activation_fn,
            pre_activation=pre_activation,
            norm=norm,
            normalize_skip_convs=True,
        )
        self.output_convolution = Conv2d(channels[-1], dvec_dim, kernel_size=3, stride=(2, 1), bias=False,
                                         activation_fn='relu', norm=norm, pre_activation=True)
        self.bn = nn.BatchNorm1d(self.embedding_dim, affine=False)

    def forward(self, x, seq_len):
        """

        Args:
            x: Shape (B T F)
            seq_len: Shape (B)
        Returns:

        """
        # Add a singleton dimension for the convolutions
        # Shape (b t f) -> (b 1 t f)
        x = einops.rearrange(x, 'b t f -> b 1 f t')

        x, seq_len = self.input_convolution(x, seq_len)
        x, seq_len = self.resnet(x, seq_len)
        x, seq_len = self.output_convolution(x, seq_len)

        # Calculate Mean over reduced frequency dim (same len for each example)
        embeddings = Mean(axis=-2)(x)
        # Calculate Mean over reduced time dim (different len for each example)
        x = Mean(axis=-1)(embeddings, seq_len)

        x = x.view(-1, self.embedding_dim)
        x = self.bn(x)

        return x, embeddings


class SpeakerNetModel(pt.Model):
    """
    The Model provides the "glue" between the actual network (`speaker_net`),
    the data (`prepare_example`), the configuration (inherits from
    `Configurable` and `finalize_dogmatic_config`) and the loss and logging
    (`loss`, `review` and `modify_summary`).
    """
    def __init__(
            self,
            speaker_net: ResNet34,
            sampling_rate=16000,
    ):
        super().__init__()
        self.speaker_net = speaker_net
        self.sampling_rate = sampling_rate

    def prepare_example(self, example):
        """
        Takes an example dictionary and prepares it for use in the model.
        The forward method receives a batched/collated version of outputs of
        this method.
        """
        # Extract fbank features from observation signal
        observation = example['audio_data']['observation']

        # Normalize signal: VoxCeleb has large variations in its variance
        observation = (observation - np.mean(observation)) / (np.std(observation) + 1e-7)
        if self.sampling_rate == 16000:
        # Extract 80 log-fbank features
            fbank_features = pb.transform.logfbank(
                observation, sample_rate=self.sampling_rate,
                number_of_filters=80,
            )
        else:
            fbank_features = pb.transform.logfbank(
                observation, sample_rate=self.sampling_rate,
                number_of_filters=80,
                window_length=200, stft_shift=80, stft_size=256
            )
        return {
            'observation': observation.astype(np.float32),
            'features': fbank_features.astype(np.float32),
            'num_frames': fbank_features.shape[0],
            'speaker_id': example['speaker_id'],
            'example_id': example['example_id'],
        }

    def forward(self, example):
        sequence_lengths = example['num_frames']
        sequence = pt.pad_sequence(example['features'], batch_first=True)
        return self.speaker_net(sequence, sequence_lengths)

    def review(self, *args, **kwargs):
        raise NotImplementedError()
