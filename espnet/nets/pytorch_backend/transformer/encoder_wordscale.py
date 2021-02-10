#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""

import logging
import torch

from espnet.nets.pytorch_backend.nets_utils import rename_state_dict
from espnet.nets.pytorch_backend.transducer.vgg2l import VGG2L
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention_wordscale
from espnet.nets.pytorch_backend.transformer.dynamic_conv import DynamicConvolution
from espnet.nets.pytorch_backend.transformer.dynamic_conv2d import DynamicConvolution2D
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder_layer_wordscale import EncoderLayer
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.lightconv import LightweightConvolution
from espnet.nets.pytorch_backend.transformer.lightconv2d import LightweightConvolution2D
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import Conv1dLinear
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import MultiLayeredConv1d
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling6
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling8


def _pre_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    # https://github.com/espnet/espnet/commit/21d70286c354c66c0350e65dc098d2ee236faccc#diff-bffb1396f038b317b2b64dd96e6d3563
    rename_state_dict(prefix + "input_layer.", prefix + "embed.", state_dict)
    # https://github.com/espnet/espnet/commit/3d422f6de8d4f03673b89e1caef698745ec749ea#diff-bffb1396f038b317b2b64dd96e6d3563
    rename_state_dict(prefix + "norm.", prefix + "after_norm.", state_dict)


class Encoder(torch.nn.Module):
    """Transformer encoder module.

    Args:
        idim (int): Input dimension.
        attention_dim (int): Dimention of attention.
        attention_heads (int): The number of heads of multi head attention.
        linear_units (int): The number of units of position-wise feed forward.
        num_blocks (int): The number of decoder blocks.
        dropout_rate (float): Dropout rate.
        attention_dropout_rate (float): Dropout rate in attention.
        positional_dropout_rate (float): Dropout rate after adding positional encoding.
        input_layer (Union[str, torch.nn.Module]): Input layer type.
        pos_enc_class (torch.nn.Module): Positional encoding module class.
            `PositionalEncoding `or `ScaledPositionalEncoding`
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        positionwise_layer_type (str): "linear", "conv1d", or "conv1d-linear".
        positionwise_conv_kernel_size (int): Kernel size of positionwise conv1d layer.
        padding_idx (int): Padding idx for input_layer=embed.

    """

    def __init__(
        self,
        idim,
        selfattention_layer_type="selfattn",
        attention_dim=256,
        attention_heads=4,
        conv_wshare=4,
        conv_kernel_length=11,
        conv_usebias=False,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.0,
        input_layer="conv2d",
        pos_enc_class=PositionalEncoding,
        normalize_before=True,
        concat_after=False,
        positionwise_layer_type="linear",
        positionwise_conv_kernel_size=1,
        padding_idx=-1,
    ):
        """Construct an Encoder object."""
        super(Encoder, self).__init__()
        self._register_load_state_dict_pre_hook(_pre_hook)

        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(idim, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(idim, attention_dim, dropout_rate)
        elif input_layer == "conv2d-scaled-pos-enc":
            self.embed = Conv2dSubsampling(
                idim,
                attention_dim,
                dropout_rate,
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling6(idim, attention_dim, dropout_rate)
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling8(idim, attention_dim, dropout_rate)
        elif input_layer == "vgg2l":
            self.embed = VGG2L(idim, attention_dim)
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(idim, attention_dim, padding_idx=padding_idx),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer is None:
            self.embed = torch.nn.Sequential(
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before
        positionwise_layer, positionwise_layer_args = self.get_positionwise_layer(
            positionwise_layer_type,
            attention_dim,
            linear_units,
            dropout_rate,
            positionwise_conv_kernel_size,
        )
        if selfattention_layer_type == "selfattn":
            logging.info("encoder self-attention layer type = self-attention")
            self.encoders = repeat(
                num_blocks,
                lambda lnum: EncoderLayer(
                    attention_dim,
                    MultiHeadedAttention_wordscale(
                        attention_heads, attention_dim, attention_dropout_rate
                    ),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                ),
            )
        elif selfattention_layer_type == "lightconv":
            logging.info("encoder self-attention layer type = lightweight convolution")
            self.encoders = repeat(
                num_blocks,
                lambda lnum: EncoderLayer(
                    attention_dim,
                    LightweightConvolution(
                        conv_wshare,
                        attention_dim,
                        attention_dropout_rate,
                        conv_kernel_length,
                        lnum,
                        use_bias=conv_usebias,
                    ),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                ),
            )
        elif selfattention_layer_type == "lightconv2d":
            logging.info(
                "encoder self-attention layer "
                "type = lightweight convolution 2-dimentional"
            )
            self.encoders = repeat(
                num_blocks,
                lambda lnum: EncoderLayer(
                    attention_dim,
                    LightweightConvolution2D(
                        conv_wshare,
                        attention_dim,
                        attention_dropout_rate,
                        conv_kernel_length,
                        lnum,
                        use_bias=conv_usebias,
                    ),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                ),
            )
        elif selfattention_layer_type == "dynamicconv":
            logging.info("encoder self-attention layer type = dynamic convolution")
            self.encoders = repeat(
                num_blocks,
                lambda lnum: EncoderLayer(
                    attention_dim,
                    DynamicConvolution(
                        conv_wshare,
                        attention_dim,
                        attention_dropout_rate,
                        conv_kernel_length,
                        lnum,
                        use_bias=conv_usebias,
                    ),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                ),
            )
        elif selfattention_layer_type == "dynamicconv2d":
            logging.info(
                "encoder self-attention layer type = dynamic convolution 2-dimentional"
            )
            self.encoders = repeat(
                num_blocks,
                lambda lnum: EncoderLayer(
                    attention_dim,
                    DynamicConvolution2D(
                        conv_wshare,
                        attention_dim,
                        attention_dropout_rate,
                        conv_kernel_length,
                        lnum,
                        use_bias=conv_usebias,
                    ),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                ),
            )
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)

    def get_positionwise_layer(
        self,
        positionwise_layer_type="linear",
        attention_dim=256,
        linear_units=2048,
        dropout_rate=0.1,
        positionwise_conv_kernel_size=1,
    ):
        """Define positionwise layer."""
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (attention_dim, linear_units, dropout_rate)
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")
        return positionwise_layer, positionwise_layer_args

    def forward(self, xs, masks, aver_mask):
        """Encode input sequence.

        Args:
            xs (torch.Tensor): Input tensor (#batch, time, idim).
            masks (torch.Tensor): Mask tensor (#batch, time).

        Returns:
            torch.Tensor: Output tensor (#batch, time, attention_dim).
            torch.Tensor: Mask tensor (#batch, time).

        """
        if isinstance(
            self.embed,
            (Conv2dSubsampling, Conv2dSubsampling6, Conv2dSubsampling8, VGG2L),
        ):
            xs, masks = self.embed(xs, masks)
        else:
            xs = self.embed(xs)
        xs, masks, aver_mask, word_mask = self.encoders(xs, masks, aver_mask)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks

    def forward_one_step(self, xs, masks, cache=None):
        """Encode input frame.

        Args:
            xs (torch.Tensor): Input tensor.
            masks (torch.Tensor): Mask tensor.
            cache (List[torch.Tensor]): List of cache tensors.

        Returns:
            torch.Tensor: Output tensor.
            torch.Tensor: Mask tensor.
            List[torch.Tensor]: List of new cache tensors.

        """
        if isinstance(self.embed, Conv2dSubsampling):
            xs, masks = self.embed(xs, masks)
        else:
            xs = self.embed(xs)
        if cache is None:
            cache = [None for _ in range(len(self.encoders))]
        new_cache = []
        for c, e in zip(cache, self.encoders):
            xs, masks, aver_mask = e(xs, masks, aver_mask, cache=c)
            new_cache.append(xs)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks, new_cache
