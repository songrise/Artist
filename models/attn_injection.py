# -*- coding : utf-8 -*-
# @FileName  : attn_injection.py
# @Author    : Ruixiang JIANG (Songrise)
# @Time      : Mar 20, 2024
# @Github    : https://github.com/songrise
# @Description: implement attention dump and attention injection for CPSD

from __future__ import annotations

from dataclasses import dataclass
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
import torch
import torch.nn as nn
from torch.nn import functional as nnf
from diffusers.models import attention_processor
import einops
from diffusers.models import unet_2d_condition, attention, transformer_2d, resnet
from diffusers.models.unets import unet_2d_blocks

# from diffusers.models.unet_2d import CrossAttnUpBlock2D
from typing import Optional, List

T = torch.Tensor
import os


@dataclass(frozen=True)
class StyleAlignedArgs:
    share_group_norm: bool = True
    share_layer_norm: bool = (True,)
    share_attention: bool = True
    adain_queries: bool = True
    adain_keys: bool = True
    adain_values: bool = False
    full_attention_share: bool = False
    shared_score_scale: float = 1.0
    shared_score_shift: float = 0.0
    only_self_level: float = 0.0


def expand_first(
    feat: T,
    scale=1.0,
) -> T:
    b = feat.shape[0]
    feat_style = torch.stack((feat[0], feat[b // 2])).unsqueeze(1)
    if scale == 1:
        feat_style = feat_style.expand(2, b // 2, *feat.shape[1:])
    else:
        feat_style = feat_style.repeat(1, b // 2, 1, 1, 1)
        feat_style = torch.cat([feat_style[:, :1], scale * feat_style[:, 1:]], dim=1)
    return feat_style.reshape(*feat.shape)


def concat_first(feat: T, dim=2, scale=1.0) -> T:
    feat_style = expand_first(feat, scale=scale)
    return torch.cat((feat, feat_style), dim=dim)


def calc_mean_std(feat, eps: float = 1e-5) -> tuple[T, T]:
    feat_std = (feat.var(dim=-2, keepdims=True) + eps).sqrt()
    feat_mean = feat.mean(dim=-2, keepdims=True)
    return feat_mean, feat_std


def adain(feat: T) -> T:
    feat_mean, feat_std = calc_mean_std(feat)
    feat_style_mean = expand_first(feat_mean)
    feat_style_std = expand_first(feat_std)
    feat = (feat - feat_mean) / feat_std
    feat = feat * feat_style_std + feat_style_mean
    return feat


def my_adain(feat: T) -> T:
    batch_size = feat.shape[0] // 2
    feat_mean, feat_std = calc_mean_std(feat)
    feat_uncond_content, feat_cond_content = feat[0], feat[batch_size]

    feat_style_mean = torch.stack((feat_mean[1], feat_mean[batch_size + 1])).unsqueeze(
        1
    )
    feat_style_mean = feat_style_mean.expand(2, batch_size, *feat_mean.shape[1:])
    feat_style_mean = feat_style_mean.reshape(*feat_mean.shape)  # (6, D)

    feat_style_std = torch.stack((feat_std[1], feat_std[batch_size + 1])).unsqueeze(1)
    feat_style_std = feat_style_std.expand(2, batch_size, *feat_std.shape[1:])
    feat_style_std = feat_style_std.reshape(*feat_std.shape)

    feat = (feat - feat_mean) / feat_std
    feat = feat * feat_style_std + feat_style_mean
    feat[0] = feat_uncond_content
    feat[batch_size] = feat_cond_content
    return feat


class DefaultAttentionProcessor(nn.Module):

    def __init__(self):
        super().__init__()
        # self.processor = attention_processor.AttnProcessor2_0()
        self.processor = attention_processor.AttnProcessor()  # for torch 1.11.0

    def __call__(
        self,
        attn: attention_processor.Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **kwargs,
    ):
        return self.processor(
            attn, hidden_states, encoder_hidden_states, attention_mask
        )


class ArtistAttentionProcessor(DefaultAttentionProcessor):
    def __init__(
        self,
        inject_query: bool = True,
        inject_key: bool = True,
        inject_value: bool = True,
        use_adain: bool = False,
        name: str = None,
        use_content_to_style_injection=False,
    ):
        super().__init__()

        self.inject_query = inject_query
        self.inject_key = inject_key
        self.inject_value = inject_value
        self.share_enabled = True
        self.use_adain = use_adain

        self.__custom_name = name
        self.content_to_style_injection = use_content_to_style_injection

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        #######Code from original attention impl
        residual = hidden_states

        # args = () if USE_PEFT_BACKEND else (scale,)
        args = ()

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)
        ######## inject begins here, here we assume the style image is always the 2nd instance in batch
        batch_size = query.shape[0] // 2  # divide 2 since CFG is used
        if self.share_enabled and batch_size > 1:  # when == 1, no need to inject,
            ref_q_uncond, ref_q_cond = query[1, ...].unsqueeze(0), query[
                batch_size + 1, ...
            ].unsqueeze(0)
            ref_k_uncond, ref_k_cond = key[1, ...].unsqueeze(0), key[
                batch_size + 1, ...
            ].unsqueeze(0)

            ref_v_uncond, ref_v_cond = value[1, ...].unsqueeze(0), value[
                batch_size + 1, ...
            ].unsqueeze(0)
            if self.inject_query:
                if self.use_adain:
                    query = my_adain(query)

                    if self.content_to_style_injection:
                        content_v_uncond = value[0, ...].unsqueeze(0)
                        content_v_cond = value[batch_size, ...].unsqueeze(0)
                        query[1] = content_v_uncond
                        query[batch_size + 1] = content_v_cond
                else:
                    query[2] = ref_q_uncond
                    query[batch_size + 2] = ref_q_cond
            if self.inject_key:
                if self.use_adain:
                    key = my_adain(key)
                else:
                    key[2] = ref_k_uncond
                    key[batch_size + 2] = ref_k_cond

            if self.inject_value:
                if self.use_adain:
                    value = my_adain(value)
                else:
                    value[2] = ref_v_uncond
                    value[batch_size + 2] = ref_v_cond

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # inject here, swap the attention map
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class ArtistResBlockWrapper(nn.Module):

    def __init__(
        self, block: resnet.ResnetBlock2D, injection_method: str, name: str = None
    ):
        super().__init__()
        self.block = block
        self.output_scale_factor = self.block.output_scale_factor
        self.injection_method = injection_method
        self.name = name

    def forward(
        self,
        input_tensor: torch.FloatTensor,
        temb: torch.FloatTensor,
        scale: float = 1.0,
    ):
        if self.injection_method == "hidden":
            feat = self.block(
                input_tensor, temb, scale
            )  # when disentangle, feat should be [recon, uncontrolled style, controlled style]
            batch_size = feat.shape[0] // 2
            if batch_size == 1:
                return feat

            # the features of the reconstruction
            recon_feat_uncond, recon_feat_cond = feat[0, ...].unsqueeze(0), feat[
                batch_size, ...
            ].unsqueeze(0)
            # residual
            input_tensor = self.block.conv_shortcut(input_tensor)
            input_content_uncond, input_content_cond = input_tensor[0, ...].unsqueeze(
                0
            ), input_tensor[batch_size, ...].unsqueeze(0)
            # since feat = (input + h) / scale
            recon_feat_uncond, recon_feat_cond = (
                recon_feat_uncond * self.output_scale_factor,
                recon_feat_cond * self.output_scale_factor,
            )
            h_content_uncond, h_content_cond = (
                recon_feat_uncond - input_content_uncond,
                recon_feat_cond - input_content_cond,
            )
            # only share the h, the residual is not shared
            h_shared = torch.cat(
                ([h_content_uncond] * batch_size) + ([h_content_cond] * batch_size),
                dim=0,
            )

            output_feat_shared = (input_tensor + h_shared) / self.output_scale_factor
            # do not inject the feat for the 2nd instance, which is uncontrolled style
            output_feat_shared[1] = feat[1]
            output_feat_shared[batch_size + 1] = feat[batch_size + 1]
            # uncomment to not inject content to controlled style
            # output_feat_shared[2] = feat[2]
            # output_feat_shared[batch_size + 2] = feat[batch_size + 2]
            return output_feat_shared
        else:
            raise NotImplementedError(f"Unknown injection method {self.injection_method}")


class SharedResBlockWrapper(nn.Module):
    def __init__(self, block: resnet.ResnetBlock2D):
        super().__init__()
        self.block = block
        self.output_scale_factor = self.block.output_scale_factor
        self.share_enabled = True

    def forward(
        self,
        input_tensor: torch.FloatTensor,
        temb: torch.FloatTensor,
        scale: float = 1.0,
    ):
        if self.share_enabled:
            feat = self.block(input_tensor, temb, scale)
            batch_size = feat.shape[0] // 2
            if batch_size == 1:
                return feat

            # the features of the reconstruction
            feat_uncond, feat_cond = feat[0, ...].unsqueeze(0), feat[
                batch_size, ...
            ].unsqueeze(0)
            # residual
            input_tensor = self.block.conv_shortcut(input_tensor)
            input_content_uncond, input_content_cond = input_tensor[0, ...].unsqueeze(
                0
            ), input_tensor[batch_size, ...].unsqueeze(0)
            # since feat = (input + h) / scale
            feat_uncond, feat_cond = (
                feat_uncond * self.output_scale_factor,
                feat_cond * self.output_scale_factor,
            )
            h_content_uncond, h_content_cond = (
                feat_uncond - input_content_uncond,
                feat_cond - input_content_cond,
            )
            # only share the h, the residual is not shared
            h_shared = torch.cat(
                ([h_content_uncond] * batch_size) + ([h_content_cond] * batch_size),
                dim=0,
            )
            output_shared = (input_tensor + h_shared) / self.output_scale_factor
            return output_shared
        else:
            return self.block(input_tensor, temb, scale)




def register_attention_processors(
    pipe,
    base_dir: str = None,
    disentangle: bool = False,
    attn_mode: str = "artist",
    resnet_mode: str = "hidden",
    share_resblock: bool = True,
    share_attn: bool = True,
    share_cross_attn: bool = False,
    share_attn_layers: Optional[int] = None,
    share_resnet_layers: Optional[int] = None,
    c2s_layers: Optional[int] = [0, 1],
    share_query: bool = True,
    share_key: bool = True,
    share_value: bool = True,
    use_adain: bool = False,
):
    unet: unet_2d_condition.UNet2DConditionModel = pipe.unet
    if isinstance(pipe, StableDiffusionPipeline):
        up_blocks: List[unet_2d_blocks.CrossAttnUpBlock2D] = unet.up_blocks[
            1:
        ]  # skip the first block, which is UpBlock2D
    elif isinstance(pipe, StableDiffusionXLPipeline):
        up_blocks: List[unet_2d_blocks.CrossAttnUpBlock2D] = unet.up_blocks[:-1]
    layer_idx_attn = 0
    layer_idx_resnet = 0
    for block in up_blocks:
        # each block should have 3 transformer layer
        #  transformer_layer : transformer_2d.Transformer2DModel
        if share_resblock:
            if share_resnet_layers is not None:
                resnet_wrappers = []
                resnets = block.resnets
                for resnet_block in resnets:
                    if layer_idx_resnet not in share_resnet_layers:
                        resnet_wrappers.append(
                            resnet_block
                        )  # use original implementation
                    else:
                        if disentangle:
                            resnet_wrappers.append(
                                ArtistResBlockWrapper(
                                    resnet_block,
                                    injection_method=resnet_mode,
                                    name=f"layer_{layer_idx_resnet}",
                                )
                            )
                            print(
                                f"Disentangle resnet {resnet_mode} set for layer {layer_idx_resnet}"
                            )
                        else:
                            resnet_wrappers.append(SharedResBlockWrapper(resnet_block))
                            print(
                                f"Share resnet feature set for layer {layer_idx_resnet}"
                            )

                    layer_idx_resnet += 1
                block.resnets = nn.ModuleList(
                    resnet_wrappers
                )  # actually apply the change
        if share_attn:
            for transformer_layer in block.attentions:
                transformer_block: attention.BasicTransformerBlock = (
                    transformer_layer.transformer_blocks[0]
                )
                self_attn: attention_processor.Attention = transformer_block.attn1
                # cross attn does not inject
                cross_attn: attention_processor.Attention = transformer_block.attn2

                if attn_mode == "artist":
                    if (
                        share_attn_layers is not None
                        and layer_idx_attn in share_attn_layers
                    ):
                        if layer_idx_attn in c2s_layers:
                            content_to_style = True
                        else:
                            content_to_style = False
                        pnp_inject_processor = ArtistAttentionProcessor(
                            inject_query=share_query,
                            inject_key=share_key,
                            inject_value=share_value,  
                            use_adain=use_adain,
                            name=f"layer_{layer_idx_attn}_self",
                            use_content_to_style_injection=content_to_style,
                        )
                        self_attn.set_processor(pnp_inject_processor)
                        print(
                            f"Disentangled Pnp inject processor set for self-attention in layer {layer_idx_attn} with c2s={content_to_style}"
                        )
                        if share_cross_attn:
                            cross_attn_processor = ArtistAttentionProcessor(
                                inject_query=False,
                                inject_key=True,
                                inject_value=True,
                                use_adain=False,
                                name=f"layer_{layer_idx_attn}_cross",
                            )
                            cross_attn.set_processor(cross_attn_processor)
                            print(
                                f"Disentangled Pnp inject processor set for cross-attention in layer {layer_idx_attn}"
                            )
                layer_idx_attn += 1


def unset_attention_processors(
    pipe,
    unset_share_attn: bool = False,
    unset_share_resblock: bool = False,
):
    unet: unet_2d_condition.UNet2DConditionMode = pipe.unet
    if isinstance(pipe, StableDiffusionPipeline):
        up_blocks: List[unet_2d_blocks.CrossAttnUpBlock2D] = unet.up_blocks[
            1:
        ]  # skip the first block, which is UpBlock2D
    elif isinstance(pipe, StableDiffusionXLPipeline):
        up_blocks: List[unet_2d_blocks.CrossAttnUpBlock2D] = unet.up_blocks[:-1]
    block_idx = 1
    layer_idx = 0
    for block in up_blocks:
        if unset_share_resblock:
            resnet_origs = []
            resnets = block.resnets
            for resnet_block in resnets:
                if isinstance(resnet_block, SharedResBlockWrapper) or isinstance(
                    resnet_block, ArtistResBlockWrapper
                ):
                    resnet_origs.append(resnet_block.block)
                else:
                    resnet_origs.append(resnet_block)
            block.resnets = nn.ModuleList(resnet_origs)
        if unset_share_attn:
            for transformer_layer in block.attentions:
                layer_idx += 1
                transformer_block: attention.BasicTransformerBlock = (
                    transformer_layer.transformer_blocks[0]
                )
                self_attn: attention_processor.Attention = transformer_block.attn1
                cross_attn: attention_processor.Attention = transformer_block.attn2
                self_attn.set_processor(DefaultAttentionProcessor())
                cross_attn.set_processor(DefaultAttentionProcessor())
        block_idx += 1
        layer_idx = 0
