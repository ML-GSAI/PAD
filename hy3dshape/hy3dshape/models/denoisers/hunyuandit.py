# Open Source Model Licensed under the Apache License Version 2.0
# and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited
# ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.

# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software and/or models in this distribution may have been
# modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import os
import yaml
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .moe_layers import MoEBlock
from ...utils import logger, synchronize_timer, smart_load_model


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    return np.concatenate([emb_sin, emb_cos], axis=1)


class Timesteps(nn.Module):
    def __init__(self,
                 num_channels: int,
                 downscale_freq_shift: float = 0.0,
                 scale: int = 1,
                 max_period: int = 10000
                 ):
        super().__init__()
        self.num_channels = num_channels
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale
        self.max_period = max_period

    def forward(self, timesteps):
        assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
        embedding_dim = self.num_channels
        half_dim = embedding_dim // 2
        exponent = -math.log(self.max_period) * torch.arange(
            start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - self.downscale_freq_shift)
        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = self.scale * emb
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, cond_proj_dim=None, out_size=None):
        super().__init__()
        if out_size is None:
            out_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, frequency_embedding_size, bias=True),
            nn.GELU(),
            nn.Linear(frequency_embedding_size, out_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, frequency_embedding_size, bias=False)

        self.time_embed = Timesteps(hidden_size)

    def forward(self, t, condition):

        t_freq = self.time_embed(t).type(self.mlp[0].weight.dtype)

        # t_freq = timestep_embedding(t, self.frequency_embedding_size).type(self.mlp[0].weight.dtype)
        if condition is not None:
            t_freq = t_freq + self.cond_proj(condition)

        t = self.mlp(t_freq)
        t = t.unsqueeze(dim=1)
        return t


class MLP(nn.Module):
    def __init__(self, *, width: int):
        super().__init__()
        self.width = width
        self.fc1 = nn.Linear(width, width * 4)
        self.fc2 = nn.Linear(width * 4, width)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


class CrossAttention(nn.Module):
    def __init__(
        self,
        qdim,
        kdim,
        num_heads,
        qkv_bias=True,
        qk_norm=False,
        norm_layer=nn.LayerNorm,
        with_decoupled_ca=False,
        decoupled_ca_dim=16,
        decoupled_ca_weight=1.0,
        **kwargs,
    ):
        super().__init__()
        self.qdim = qdim
        self.kdim = kdim
        self.num_heads = num_heads
        assert self.qdim % num_heads == 0, "self.qdim must be divisible by num_heads"
        self.head_dim = self.qdim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(qdim, qdim, bias=qkv_bias)
        self.to_k = nn.Linear(kdim, qdim, bias=qkv_bias)
        self.to_v = nn.Linear(kdim, qdim, bias=qkv_bias)

        # TODO: eps should be 1 / 65530 if using fp16
        self.q_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.out_proj = nn.Linear(qdim, qdim, bias=True)

        self.with_dca = with_decoupled_ca
        if self.with_dca:
            self.kv_proj_dca = nn.Linear(kdim, 2 * qdim, bias=qkv_bias)
            self.k_norm_dca = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
            self.dca_dim = decoupled_ca_dim
            self.dca_weight = decoupled_ca_weight

    def forward(self, x, y):
        """
        Parameters
        ----------
        x: torch.Tensor
            (batch, seqlen1, hidden_dim) (where hidden_dim = num heads * head dim)
        y: torch.Tensor
            (batch, seqlen2, hidden_dim2)
        freqs_cis_img: torch.Tensor
            (batch, hidden_dim // 2), RoPE for image
        """
        b, s1, c = x.shape  # [b, s1, D]

        if self.with_dca:
            token_len = y.shape[1]
            context_dca = y[:, -self.dca_dim:, :]
            kv_dca = self.kv_proj_dca(context_dca).view(b, self.dca_dim, 2, self.num_heads, self.head_dim)
            k_dca, v_dca = kv_dca.unbind(dim=2)  # [b, s, h, d]
            k_dca = self.k_norm_dca(k_dca)
            y = y[:, :(token_len - self.dca_dim), :]

        _, s2, c = y.shape  # [b, s2, 1024]
        q = self.to_q(x)
        k = self.to_k(y)
        v = self.to_v(y)

        kv = torch.cat((k, v), dim=-1)
        split_size = kv.shape[-1] // self.num_heads // 2
        kv = kv.view(1, -1, self.num_heads, split_size * 2)
        k, v = torch.split(kv, split_size, dim=-1)

        q = q.view(b, s1, self.num_heads, self.head_dim)  # [b, s1, h, d]
        k = k.view(b, s2, self.num_heads, self.head_dim)  # [b, s2, h, d]
        v = v.view(b, s2, self.num_heads, self.head_dim)  # [b, s2, h, d]

        q = self.q_norm(q)
        k = self.k_norm(k)

        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=False,
            enable_mem_efficient=True
        ):
            q, k, v = map(lambda t: rearrange(t, 'b n h d -> b h n d', h=self.num_heads), (q, k, v))
            context = F.scaled_dot_product_attention(
                q, k, v
            ).transpose(1, 2).reshape(b, s1, -1)

        if self.with_dca:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=False,
                enable_mem_efficient=True
            ):
                k_dca, v_dca = map(lambda t: rearrange(t, 'b n h d -> b h n d', h=self.num_heads),
                                   (k_dca, v_dca))
                context_dca = F.scaled_dot_product_attention(
                    q, k_dca, v_dca).transpose(1, 2).reshape(b, s1, -1)

            context = context + self.dca_weight * context_dca

        out = self.out_proj(context)  # context.reshape - B, L1, -1

        return out


class Attention(nn.Module):
    """
    We rename some layer names to align with flash attention
    """

    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=True,
        qk_norm=False,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.head_dim = self.dim // num_heads
        # This assertion is aligned with flash attention
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        # TODO: eps should be 1 / 65530 if using fp16
        self.q_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        qkv = torch.cat((q, k, v), dim=-1)
        split_size = qkv.shape[-1] // self.num_heads // 3
        qkv = qkv.view(1, -1, self.num_heads, split_size * 3)
        q, k, v = torch.split(qkv, split_size, dim=-1)

        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [b, h, s, d]
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [b, h, s, d]
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)  # [b, h, s, d]
        k = self.k_norm(k)  # [b, h, s, d]

        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=False,
            enable_mem_efficient=True
        ):
            x = F.scaled_dot_product_attention(q, k, v)
            x = x.transpose(1, 2).reshape(B, N, -1)

        x = self.out_proj(x)
        return x


class HunYuanDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        c_emb_size,
        num_heads,
        text_states_dim=1024,
        use_flash_attn=False,
        qk_norm=False,
        norm_layer=nn.LayerNorm,
        qk_norm_layer=nn.RMSNorm,
        with_decoupled_ca=False,
        decoupled_ca_dim=16,
        decoupled_ca_weight=1.0,
        init_scale=1.0,
        qkv_bias=True,
        skip_connection=True,
        timested_modulate=False,
        use_moe: bool = False,
        num_experts: int = 8,
        moe_top_k: int = 2,
        depth_cond: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.depth_cond = depth_cond
        self.use_flash_attn = use_flash_attn
        use_ele_affine = True

        # ========================= Self-Attention =========================
        self.norm1 = norm_layer(hidden_size, elementwise_affine=use_ele_affine, eps=1e-6)
        self.attn1 = Attention(hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm,
                               norm_layer=qk_norm_layer)

        # ========================= FFN =========================
        self.norm2 = norm_layer(hidden_size, elementwise_affine=use_ele_affine, eps=1e-6)
        if depth_cond:
            self.norm_dpt = norm_layer(hidden_size, elementwise_affine=use_ele_affine, eps=1e-6)

        # ========================= Add =========================
        # Simply use add like SDXL.
        self.timested_modulate = timested_modulate
        if self.timested_modulate:
            self.default_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(c_emb_size, hidden_size, bias=True)
            )

        # ========================= Cross-Attention =========================
        self.attn2 = CrossAttention(hidden_size, text_states_dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                    qk_norm=qk_norm, norm_layer=qk_norm_layer,
                                    with_decoupled_ca=with_decoupled_ca, decoupled_ca_dim=decoupled_ca_dim,
                                    decoupled_ca_weight=decoupled_ca_weight, init_scale=init_scale,
                                    )
        
        if depth_cond:
            self.attn_dpt = CrossAttention(hidden_size, text_states_dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                    qk_norm=qk_norm, norm_layer=qk_norm_layer,
                                    with_decoupled_ca=with_decoupled_ca, decoupled_ca_dim=decoupled_ca_dim,
                                    decoupled_ca_weight=decoupled_ca_weight, init_scale=init_scale,
                                    )
            # Zero-initialize all parameters of attn3 so its initial output is effectively 0.
            # This keeps the module present (same interface) but disables its effect at init.
            for p in self.attn_dpt.out_proj.parameters():
                if p is not None:
                    nn.init.zeros_(p)
        
        self.norm3 = norm_layer(hidden_size, elementwise_affine=True, eps=1e-6)

        if skip_connection:
            self.skip_norm = norm_layer(hidden_size, elementwise_affine=True, eps=1e-6)
            self.skip_linear = nn.Linear(2 * hidden_size, hidden_size)
        else:
            self.skip_linear = None

        self.use_moe = use_moe
        if self.use_moe:
            print("using moe")
            self.moe = MoEBlock(
                hidden_size,
                num_experts=num_experts,
                moe_top_k=moe_top_k,
                dropout=0.0,
                activation_fn="gelu",
                final_dropout=False,
                ff_inner_dim=int(hidden_size * 4.0),
                ff_bias=True,
            )
        else:
            self.mlp = MLP(width=hidden_size)

    def forward(self, x, c=None, text_states=None, skip_value=None):

        if self.skip_linear is not None:
            cat = torch.cat([skip_value, x], dim=-1)
            x = self.skip_linear(cat)
            x = self.skip_norm(x)

        # Self-Attention
        if self.timested_modulate: # False
            print("use timested_modulate")
            shift_msa = self.default_modulation(c).unsqueeze(dim=1)
            x = x + shift_msa

        attn_out = self.attn1(self.norm1(x))

        x = x + attn_out

        # Cross-Attention
        if self.depth_cond:
            assert type(text_states) is list, "depth_cond is True, text_states should be a list of two tensors"
            main_text_states = text_states[0]
            depth_text_states = text_states[1]
            x = x + self.attn2(self.norm2(x), main_text_states)
            x_backup = x
            x = x + self.attn_dpt(self.norm_dpt(x), depth_text_states)
            # assert (x_backup - x).abs().sum() == 0 , "attn_dpt should has no effect, please check the implementation"
        else:
            x = x + self.attn2(self.norm2(x), text_states)

        # FFN Layer
        mlp_inputs = self.norm3(x)

        if self.use_moe:
            x = x + self.moe(mlp_inputs)
        else:
            x = x + self.mlp(mlp_inputs)

        return x


class AttentionPool(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x, attention_mask=None):
        x = x.permute(1, 0, 2)  # NLC -> LNC
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1).permute(1, 0, 2)
            global_emb = (x * attention_mask).sum(dim=0) / attention_mask.sum(dim=0)
            x = torch.cat([global_emb[None,], x], dim=0)

        else:
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (L+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (L+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class FinalLayer(nn.Module):
    """
    The final layer of HunYuanDiT.
    """

    def __init__(self, final_hidden_size, out_channels):
        super().__init__()
        self.final_hidden_size = final_hidden_size
        self.norm_final = nn.LayerNorm(final_hidden_size, elementwise_affine=True, eps=1e-6)
        self.linear = nn.Linear(final_hidden_size, out_channels, bias=True)

    def forward(self, x):
        x = self.norm_final(x)
        x = x[:, 1:]
        x = self.linear(x)
        return x


class HunYuanDiTPlain(nn.Module):

    @classmethod
    @synchronize_timer('HunYuanDiTPlain Model Loading')
    def from_single_file(
        cls,
        ckpt_path,
        config_path,
        device='cuda',
        dtype=torch.float16,
        use_safetensors=None,
        **kwargs,
    ):
        # load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # load ckpt
        if use_safetensors:
            ckpt_path = ckpt_path.replace('.ckpt', '.safetensors')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model file {ckpt_path} not found")

        logger.info(f"Loading model from {ckpt_path}")
        if use_safetensors:
            import safetensors.torch
            ckpt = safetensors.torch.load_file(ckpt_path, device='cpu')
        else:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)

        if 'model' in ckpt:
            ckpt = ckpt['model']
        if 'model' in config:
            config = config['model']

        model_kwargs = config['params']
        model_kwargs.update(kwargs)

        model = cls(**model_kwargs)

        missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
        print(f"Missing keys: {missing_keys}", f"Unexpected keys: {unexpected_keys}")

        # current_device_idx = torch.cuda.current_device()
        # print(f"Process {os.getpid()}:Model loaded on device {current_device_idx} with dtype {model.parameters().__next__().dtype}")
        model.to(device=device, dtype=dtype)
        return model

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        device='cuda',
        dtype=torch.float16,
        use_safetensors=False,
        variant='fp16',
        subfolder='hunyuan3d-dit-v2-1',
        initialize_depth_cond=False,
        **kwargs,
    ):
        if initialize_depth_cond:
            return cls.from_pretrained_init(
                model_path,
                device=device,
                dtype=dtype,
                use_safetensors=use_safetensors,
                variant=variant,
                subfolder=subfolder,
                **kwargs
            )

        config_path, ckpt_path = smart_load_model(
            model_path,
            subfolder=subfolder,
            use_safetensors=use_safetensors,
            variant=variant
        )

        return cls.from_single_file(
            ckpt_path,
            config_path,
            device=device,
            dtype=dtype,
            use_safetensors=use_safetensors,
            **kwargs
        )

    @classmethod
    def from_pretrained_init(
        cls,
        model_path,
        device='cuda',
        dtype=torch.float16,
        use_safetensors=False,
        variant='fp16',
        subfolder='hunyuan3d-dit-v2-1',
        **kwargs,
    ):
        '''
        depth conditiondepth condition
        ,depth condition
        depth conditiondepth condition
        from_pretrained
        '''
        # Load model config path and checkpoint path
        config_path, ckpt_path = smart_load_model(
            model_path,
            subfolder=subfolder,
            use_safetensors=use_safetensors,
            variant=variant
        )

        # Read config to obtain model constructor kwargs
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if 'model' in config:
            config = config['model']

        model_kwargs = config.get('params', {})
        # Ensure we construct a model with depth condition enabled
        model_kwargs.update(kwargs)
        model_kwargs['depth_cond'] = True

        # Instantiate model (keeps default initialization for depth-related params)
        model = cls(**model_kwargs)

        # Load checkpoint (on CPU) and filter matching keys by name and shape
        logger.info(f"Loading checkpoint from {ckpt_path} for partial init (depth_cond enabled)")
        if use_safetensors:
            import safetensors.torch
            ckpt = safetensors.torch.load_file(ckpt_path, device='cpu')
        else:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)

        # Some checkpoints nest under 'model'
        if isinstance(ckpt, dict) and 'model' in ckpt:
            ckpt = ckpt['model']

        # Prepare a filtered state dict that only contains keys that exist in the
        # new model and have matching tensor shapes. This preserves default
        # initialization for any newly added depth-condition parameters.
        model = cls.load_partial_ckpt(model, ckpt)

        # Move model to target device and dtype
        model.to(device=device, dtype=dtype)
        return model


    @classmethod
    def load_partial_ckpt(cls, model, ckpt_dict):
        model_state = model.state_dict()
        filtered_ckpt = {}
        for k, v in ckpt_dict.items():
            if k in model_state:
                try:
                    if tuple(model_state[k].shape) == tuple(v.shape):
                        filtered_ckpt[k] = v
                    else:
                        logger.info(f"Skip param due to shape mismatch: {k} checkpoint {tuple(v.shape)} vs model {tuple(model_state[k].shape)}")
                except Exception:
                    # If shapes cannot be compared for some reason, skip
                    logger.info(f"Skip param (exception comparing shape): {k}")
            else:
                # key not present in model; skip
                logger.debug(f"Checkpoint key not found in model and will be skipped: {k}")

        # Load filtered params into model (non-strict to allow missing keys)
        res = model.load_state_dict(filtered_ckpt, strict=False)
        # res may be a NamedTuple with attributes or a tuple (missing, unexpected)
        missing = getattr(res, 'missing_keys', None)
        unexpected = getattr(res, 'unexpected_keys', None)
        if missing is None or unexpected is None:
            try:
                missing, unexpected = res
            except Exception:
                missing = unexpected = []
        missing = missing or []
        unexpected = unexpected or []
        logger.info(f"Loaded {len(filtered_ckpt)} params into model; missing: {len(missing)}, unexpected: {len(unexpected)}")
        logger.info(f"Missing keys: {missing}")
        return model




    def __init__(
        self,
        input_size=1024,
        in_channels=4,
        hidden_size=1024,
        context_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        norm_type='layer',
        qk_norm_type='rms',
        qk_norm=False,
        text_len=257,
        with_decoupled_ca=False,
        additional_cond_hidden_state=768,
        decoupled_ca_dim=16,
        decoupled_ca_weight=1.0,
        use_pos_emb=False,
        use_attention_pooling=True,
        guidance_cond_proj_dim=None,
        qkv_bias=True,
        num_moe_layers: int = 6,
        num_experts: int = 8,
        moe_top_k: int = 2,
        depth_cond: bool = False,
        **kwargs
    ):
        super().__init__()
        self.input_size = input_size
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads

        self.hidden_size = hidden_size
        self.norm = nn.LayerNorm if norm_type == 'layer' else nn.RMSNorm
        self.qk_norm = nn.RMSNorm if qk_norm_type == 'rms' else nn.LayerNorm
        self.context_dim = context_dim

        self.with_decoupled_ca = with_decoupled_ca
        self.decoupled_ca_dim = decoupled_ca_dim
        self.decoupled_ca_weight = decoupled_ca_weight
        self.use_pos_emb = use_pos_emb
        self.use_attention_pooling = use_attention_pooling
        self.guidance_cond_proj_dim = guidance_cond_proj_dim

        self.text_len = text_len

        self.x_embedder = nn.Linear(in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size, hidden_size * 4, cond_proj_dim=guidance_cond_proj_dim)

        # Will use fixed sin-cos embedding:
        if self.use_pos_emb:
            assert 0
            self.register_buffer("pos_embed", torch.zeros(1, input_size, hidden_size))
            pos = np.arange(self.input_size, dtype=np.float32)
            pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], pos)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.use_attention_pooling = use_attention_pooling
        if use_attention_pooling:
            assert 0
            self.pooler = AttentionPool(self.text_len, context_dim, num_heads=8, output_dim=1024)
            self.extra_embedder = nn.Sequential(
                nn.Linear(1024, hidden_size * 4),
                nn.SiLU(),
                nn.Linear(hidden_size * 4, hidden_size, bias=True),
            )

        if with_decoupled_ca:
            assert 0
            self.additional_cond_hidden_state = additional_cond_hidden_state
            self.additional_cond_proj = nn.Sequential(
                nn.Linear(additional_cond_hidden_state, hidden_size * 4),
                nn.SiLU(),
                nn.Linear(hidden_size * 4, 1024, bias=True),
            )

        # HUnYuanDiT Blocks
        self.blocks = nn.ModuleList([
            HunYuanDiTBlock(hidden_size=hidden_size,
                            c_emb_size=hidden_size,
                            num_heads=num_heads,
                            mlp_ratio=mlp_ratio,
                            text_states_dim=context_dim,
                            qk_norm=qk_norm,
                            norm_layer=self.norm,
                            qk_norm_layer=self.qk_norm,
                            skip_connection=layer > depth // 2,
                            with_decoupled_ca=with_decoupled_ca,
                            decoupled_ca_dim=decoupled_ca_dim,
                            decoupled_ca_weight=decoupled_ca_weight,
                            qkv_bias=qkv_bias,
                            use_moe=True if depth - layer <= num_moe_layers else False,
                            num_experts=num_experts,
                            moe_top_k=moe_top_k,
                            depth_cond=depth_cond
                            )
            for layer in range(depth)
        ])
        self.depth = depth

        self.final_layer = FinalLayer(hidden_size, self.out_channels)

    def forward(self, x, t, contexts, **kwargs):
        cond = contexts['main']
        if 'depth' in contexts:
            cond = [cond, contexts['depth']]
        assert kwargs.get("guidance_cond") is None, "guidance_cond should be provided in kwargs"
        
        t = self.t_embedder(t, condition=kwargs.get('guidance_cond'))
        x = self.x_embedder(x)

        if self.use_pos_emb:
            pos_embed = self.pos_embed.to(x.dtype)
            x = x + pos_embed

        if self.use_attention_pooling: # False
            print("use attention pooling")
            extra_vec = self.pooler(cond, None)
            c = t + self.extra_embedder(extra_vec)  # [B, D]
        else:
            c = t

        if self.with_decoupled_ca: # False
            print("use additional cond")
            additional_cond = self.additional_cond_proj(contexts['additional'])
            cond = torch.cat([cond, additional_cond], dim=1)

        x = torch.cat([c, x], dim=1)

        skip_value_list = []
        for layer, block in enumerate(self.blocks):
            skip_value = None if layer <= self.depth // 2 else skip_value_list.pop()
            x = block(x, c, cond, skip_value=skip_value)
            if layer < self.depth // 2:
                skip_value_list.append(x)

        x = self.final_layer(x)
        return x


class HunYuanDiTPlainWithRefiner(nn.Module):
    """
    AModel wrapper that includes a pre-trained HunYuanDiTPlain model and additional refinement layers.
    """
    def __init__(
        self, 
        ckpt_path=None,
        refine_start_channel=None,
        refiner_depth=2,
        refiner_hidden_size=None,
        refiner_num_heads=None,
        refiner_mlp_ratio=4.0,
        device='cuda',
        dtype=torch.float16,
        **kwargs
    ):
        super().__init__()
        
        logger.info(f"Initializing Base Model from {ckpt_path}")
        self.base_model = HunYuanDiTPlain.from_pretrained(ckpt_path)
        
        # inherit attributes from base model for convenience
        self.context_dim = self.base_model.context_dim
        self.out_channels = self.base_model.out_channels if refine_start_channel is None else refine_start_channel
        self.hidden_size = self.base_model.hidden_size
        self.with_decoupled_ca = self.base_model.with_decoupled_ca
        self.decoupled_ca_dim = self.base_model.decoupled_ca_dim
        self.decoupled_ca_weight = self.base_model.decoupled_ca_weight
        self.norm = self.base_model.norm
        self.qk_norm = self.base_model.qk_norm

        # Refiner Configuration
        self.refiner_depth = refiner_depth
        self.refiner_hidden_size = refiner_hidden_size if refiner_hidden_size is not None else self.hidden_size
        self.refiner_num_heads = refiner_num_heads if refiner_num_heads is not None else self.base_model.num_heads
        
        # Project output to refiner hidden size
        self.refiner_start = nn.Linear(self.out_channels, self.refiner_hidden_size)

        # Create Refiner Blocks
        self.refiner_blocks = nn.ModuleList([
            HunYuanDiTBlock(
                hidden_size=self.refiner_hidden_size,
                c_emb_size=self.refiner_hidden_size, # Conditioning dimension adapted
                num_heads=self.refiner_num_heads,
                mlp_ratio=refiner_mlp_ratio,
                text_states_dim=self.context_dim,
                qk_norm=self.qk_norm,
                norm_layer=self.norm,
                qk_norm_layer=self.qk_norm,
                skip_connection=layer > refiner_depth // 2, # Enable skip connection for refiner
                with_decoupled_ca=self.with_decoupled_ca,
                decoupled_ca_dim=self.decoupled_ca_dim,
                decoupled_ca_weight=self.decoupled_ca_weight,
                qkv_bias=False,
                use_moe=False,
                num_experts=8,
                moe_top_k=2,
                depth_cond=False,
            )
            for layer in range(refiner_depth)
        ])
        
        # Final projection layer for refiner
        self.refiner_final_layer = FinalLayer(self.refiner_hidden_size, 4)
        
        # Initialize final layer to zero for stable fine-tuning start
        # nn.init.constant_(self.refiner_final_layer.linear.weight, 0)
        # nn.init.constant_(self.refiner_final_layer.linear.bias, 0)

        # Freeze base model parameters
        self._freeze_base_model()
        # Move refiner to device/dtype
        self.refiner_blocks.to(device=device, dtype=dtype)
        self.refiner_start.to(device=device, dtype=dtype)
        self.refiner_final_layer.to(device=device, dtype=dtype)
        self.learnable_embed = nn.Parameter(torch.zeros(1, 1, self.refiner_hidden_size, device=device, dtype=dtype))

    def _freeze_base_model(self):
        """
        Freezes parameters of the base model.
        """
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Ensure refiner parts are trainable (default is True, but just to be explicit logic)
        for name, param in self.named_parameters():
             if "base_model" not in name:
                 param.requires_grad = True


    def forward(self, x, t, contexts, **kwargs):
        # 1. Run the base model to get initial estimation
        # x_base = self.base_model(x, t, contexts, **kwargs)
        latents = contexts['latents']
        x_base = latents
        
        # 2. Re-prepare conditioning (c and cond) for refiner blocks
        # We need to re-compute embeddings because they are local variables in base_model.forward
        # Alternatively, we could modify base_model to return them, but that touches original code.
        # So we duplicate the lightweight embedding logic here.
        
        cond = contexts['main']

        t_emb = self.base_model.t_embedder(t, condition=kwargs.get('guidance_cond'))
        
        c = t_emb

        # 3. Refiner Logic
        x_refine = self.refiner_start(x_base)
        x_refine = torch.cat([c, self.learnable_embed.expand(x_refine.size(0), -1, -1), x_refine], dim=1)
        
        skip_value_list = []
        for layer, block in enumerate(self.refiner_blocks):
            skip_value = None if layer <= self.refiner_depth // 2 else skip_value_list.pop()
            x_refine = block(x_refine, c, cond, skip_value=skip_value)
            if layer < self.refiner_depth // 2:
                skip_value_list.append(x_refine)
            
        x_refine = self.refiner_final_layer(x_refine)
        
        # Residual connection: base output + refinement
        return x_refine[:, :1, :]
    

class HunYuanDiTPlainWithExtraOutput(HunYuanDiTPlain):

    def __init__(
        self,
        input_size=1024,
        in_channels=4,
        hidden_size=1024,
        context_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        norm_type='layer',
        qk_norm_type='rms',
        qk_norm=False,
        text_len=257,
        with_decoupled_ca=False,
        additional_cond_hidden_state=768,
        decoupled_ca_dim=16,
        decoupled_ca_weight=1.0,
        use_pos_emb=False,
        use_attention_pooling=True,
        guidance_cond_proj_dim=None,
        qkv_bias=True,
        num_moe_layers: int = 6,
        num_experts: int = 8,
        moe_top_k: int = 2,
        depth_cond: bool = False,
        **kwargs
    ):
        nn.Module.__init__(self)
        self.input_size = input_size
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads

        self.hidden_size = hidden_size
        self.norm = nn.LayerNorm if norm_type == 'layer' else nn.RMSNorm
        self.qk_norm = nn.RMSNorm if qk_norm_type == 'rms' else nn.LayerNorm
        self.context_dim = context_dim

        self.with_decoupled_ca = with_decoupled_ca
        self.decoupled_ca_dim = decoupled_ca_dim
        self.decoupled_ca_weight = decoupled_ca_weight
        self.use_pos_emb = use_pos_emb
        self.use_attention_pooling = use_attention_pooling
        self.guidance_cond_proj_dim = guidance_cond_proj_dim

        self.text_len = text_len

        self.x_embedder = nn.Linear(in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size, hidden_size * 4, cond_proj_dim=guidance_cond_proj_dim)

        self.use_attention_pooling = use_attention_pooling
        # HUnYuanDiT Blocks
        self.blocks = nn.ModuleList([
            HunYuanDiTBlock(hidden_size=hidden_size,
                            c_emb_size=hidden_size,
                            num_heads=num_heads,
                            mlp_ratio=mlp_ratio,
                            text_states_dim=context_dim,
                            qk_norm=qk_norm,
                            norm_layer=self.norm,
                            qk_norm_layer=self.qk_norm,
                            skip_connection=layer > depth // 2,
                            with_decoupled_ca=with_decoupled_ca,
                            decoupled_ca_dim=decoupled_ca_dim,
                            decoupled_ca_weight=decoupled_ca_weight,
                            qkv_bias=qkv_bias,
                            use_moe=True if depth - layer <= num_moe_layers else False,
                            num_experts=num_experts,
                            moe_top_k=moe_top_k,
                            depth_cond=depth_cond
                            )
            for layer in range(depth)
        ])
        self.depth = depth


        self.learnable_embed = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.1)
        self.final_layer = FinalLayer(hidden_size, self.out_channels)
        self.final_layer_extra = FinalLayer(hidden_size, 4)  # Extra output channel

    def forward(self, x, t, contexts, **kwargs):
        cond = contexts['main']
        if 'depth' in contexts:
            cond = [cond, contexts['depth']]
        assert kwargs.get("guidance_cond") is None, "guidance_cond should be provided in kwargs"
        
        t = self.t_embedder(t, condition=kwargs.get('guidance_cond'))
        x = self.x_embedder(x)

        c = t

        x = torch.cat([c, self.learnable_embed.expand(x.size(0), -1, -1), x], dim=1)

        skip_value_list = []
        for layer, block in enumerate(self.blocks):
            skip_value = None if layer <= self.depth // 2 else skip_value_list.pop()
            x = block(x, c, cond, skip_value=skip_value)
            if layer < self.depth // 2:
                skip_value_list.append(x)

        output = self.final_layer(x[:, 1:, :])
        x_extra = self.final_layer_extra(x[:, :2, :])
        self.extra = x_extra
        return output
    
    def get_extra_output(self):
        return self.extra

class HunYuanDiTPlainWithPtsOutput(HunYuanDiTPlain):

    def __init__(
        self,
        input_size=1024,
        in_channels=4,
        hidden_size=1024,
        context_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        norm_type='layer',
        qk_norm_type='rms',
        qk_norm=False,
        text_len=257,
        with_decoupled_ca=False,
        additional_cond_hidden_state=768,
        decoupled_ca_dim=16,
        decoupled_ca_weight=1.0,
        use_pos_emb=False,
        use_attention_pooling=True,
        guidance_cond_proj_dim=None,
        qkv_bias=True,
        num_moe_layers: int = 6,
        num_experts: int = 8,
        moe_top_k: int = 2,
        depth_cond: bool = False,
        **kwargs
    ):
        nn.Module.__init__(self)
        self.input_size = input_size
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads

        self.hidden_size = hidden_size
        self.norm = nn.LayerNorm if norm_type == 'layer' else nn.RMSNorm
        self.qk_norm = nn.RMSNorm if qk_norm_type == 'rms' else nn.LayerNorm
        self.context_dim = context_dim

        self.with_decoupled_ca = with_decoupled_ca
        self.decoupled_ca_dim = decoupled_ca_dim
        self.decoupled_ca_weight = decoupled_ca_weight
        self.use_pos_emb = use_pos_emb
        self.use_attention_pooling = use_attention_pooling
        self.guidance_cond_proj_dim = guidance_cond_proj_dim

        self.text_len = text_len

        self.x_embedder = nn.Linear(in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size, hidden_size * 4, cond_proj_dim=guidance_cond_proj_dim)

        self.use_attention_pooling = use_attention_pooling
        # HUnYuanDiT Blocks
        self.blocks = nn.ModuleList([
            HunYuanDiTBlock(hidden_size=hidden_size,
                            c_emb_size=hidden_size,
                            num_heads=num_heads,
                            mlp_ratio=mlp_ratio,
                            text_states_dim=context_dim,
                            qk_norm=qk_norm,
                            norm_layer=self.norm,
                            qk_norm_layer=self.qk_norm,
                            skip_connection=layer > depth // 2,
                            with_decoupled_ca=with_decoupled_ca,
                            decoupled_ca_dim=decoupled_ca_dim,
                            decoupled_ca_weight=decoupled_ca_weight,
                            qkv_bias=qkv_bias,
                            use_moe=True if depth - layer <= num_moe_layers else False,
                            num_experts=num_experts,
                            moe_top_k=moe_top_k,
                            depth_cond=depth_cond
                            )
            for layer in range(depth)
        ])

        self.depth = depth
        # self.learnable_pe = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.1)

        self.final_layer = FinalLayer(hidden_size, self.out_channels)


    def forward(self, x, t, contexts, **kwargs):
        cond = contexts['main']
        pts = contexts["pts"]
        assert kwargs.get("guidance_cond") is None, "guidance_cond should be provided in kwargs"
        
        t = self.t_embedder(t, condition=kwargs.get('guidance_cond'))
        x = self.x_embedder(x)
        pts = self.x_embedder(pts)


        c = t

        # pts = pts + self.learnable_pe.expand(pts.size(0), -1, -1)

        x = torch.cat([c, x, pts], dim=1)

        skip_value_list = []
        for layer, block in enumerate(self.blocks):
            skip_value = None if layer <= self.depth // 2 else skip_value_list.pop()
            x = block(x, c, cond, skip_value=skip_value)
            if layer < self.depth // 2:
                skip_value_list.append(x)

        output = self.final_layer(x[:, :-pts.shape[1], :])

        return output
    


class HunYuanDiTPlainWithExtraDepth(HunYuanDiTPlain):

    def __init__(
        self,
        input_size=1024,
        in_channels=4,
        hidden_size=1024,
        context_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        norm_type='layer',
        qk_norm_type='rms',
        qk_norm=False,
        text_len=257,
        with_decoupled_ca=False,
        additional_cond_hidden_state=768,
        decoupled_ca_dim=16,
        decoupled_ca_weight=1.0,
        use_pos_emb=False,
        use_attention_pooling=True,
        guidance_cond_proj_dim=None,
        qkv_bias=True,
        num_moe_layers: int = 6,
        num_experts: int = 8,
        moe_top_k: int = 2,
        depth_cond: bool = False,
        **kwargs
    ):
        nn.Module.__init__(self)
        self.input_size = input_size
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads

        self.hidden_size = hidden_size
        self.norm = nn.LayerNorm if norm_type == 'layer' else nn.RMSNorm
        self.qk_norm = nn.RMSNorm if qk_norm_type == 'rms' else nn.LayerNorm
        self.context_dim = context_dim

        self.with_decoupled_ca = with_decoupled_ca
        self.decoupled_ca_dim = decoupled_ca_dim
        self.decoupled_ca_weight = decoupled_ca_weight
        self.use_pos_emb = use_pos_emb
        self.use_attention_pooling = use_attention_pooling
        self.guidance_cond_proj_dim = guidance_cond_proj_dim

        self.text_len = text_len

        self.x_embedder = nn.Linear(in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size, hidden_size * 4, cond_proj_dim=guidance_cond_proj_dim)

        self.use_attention_pooling = use_attention_pooling
        # HUnYuanDiT Blocks
        self.blocks = nn.ModuleList([
            HunYuanDiTBlock(hidden_size=hidden_size,
                            c_emb_size=hidden_size,
                            num_heads=num_heads,
                            mlp_ratio=mlp_ratio,
                            text_states_dim=context_dim,
                            qk_norm=qk_norm,
                            norm_layer=self.norm,
                            qk_norm_layer=self.qk_norm,
                            skip_connection=layer > depth // 2,
                            with_decoupled_ca=with_decoupled_ca,
                            decoupled_ca_dim=decoupled_ca_dim,
                            decoupled_ca_weight=decoupled_ca_weight,
                            qkv_bias=qkv_bias,
                            use_moe=True if depth - layer <= num_moe_layers else False,
                            num_experts=num_experts,
                            moe_top_k=moe_top_k,
                            depth_cond=depth_cond
                            )
            for layer in range(depth)
        ])

        self.extra_layer = nn.ModuleList([
            HunYuanDiTBlock(hidden_size=hidden_size,
                            c_emb_size=hidden_size,
                            num_heads=num_heads,
                            mlp_ratio=mlp_ratio,
                            text_states_dim=context_dim,
                            qk_norm=qk_norm,
                            norm_layer=self.norm,
                            qk_norm_layer=self.qk_norm,
                            skip_connection=layer > depth // 2,
                            with_decoupled_ca=with_decoupled_ca,
                            decoupled_ca_dim=decoupled_ca_dim,
                            decoupled_ca_weight=decoupled_ca_weight,
                            qkv_bias=qkv_bias,
                            use_moe=True if depth - layer <= num_moe_layers else False,
                            num_experts=num_experts,
                            moe_top_k=moe_top_k,
                            depth_cond=depth_cond
                            )
            for layer in range(5)
        ]) 

        self.depth = depth

        self.final_layer = FinalLayer(hidden_size, self.out_channels)


    def forward(self, x, t, contexts, **kwargs):
        cond = contexts['main']
        # pts = contexts["pts"]
        assert kwargs.get("guidance_cond") is None, "guidance_cond should be provided in kwargs"
        
        t = self.t_embedder(t, condition=kwargs.get('guidance_cond'))
        x = self.x_embedder(x)


        c = t

        x = torch.cat([c, x], dim=1)

        skip_value_list = []
        for layer, block in enumerate(self.blocks):
            skip_value = None if layer <= self.depth // 2 else skip_value_list.pop()
            x = block(x, c, cond, skip_value=skip_value)
            if layer < self.depth // 2:
                skip_value_list.append(x)

        skip_value_list = []
        for layer, block in enumerate(self.extra_layer):
            skip_value = None if layer <= 2 else skip_value_list.pop()
            x = block(x, c, cond, skip_value=skip_value)
            if layer < 2:
                skip_value_list.append(x)

        output = self.final_layer(x)

        return output