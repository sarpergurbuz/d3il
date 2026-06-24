# I took this code from Affordance-based Robot Manipulation with Flow Matching

from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import math
import torch
import torch.nn as nn
import torchvision
import collections
import einops


from tqdm.auto import tqdm
import sys





class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 cond_dim,
                 kernel_size=3,
                 n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn, n_groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(n_groups, dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))

    


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x
    


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b (h c) d -> b h c d', h=self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim = -1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = einops.rearrange(out, 'b h c d -> b (h c) d')
        return self.to_out(out)

class ObstacleEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        gn_groups = 8
        self.cnn_stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(gn_groups, 32),
            nn.Mish(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(gn_groups, 32),
            nn.Mish(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(gn_groups, 32),
            nn.Mish(),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((8, 8))

        self.mlp = nn.Sequential(
            nn.Linear(2048, 512),
            nn.Mish(),
            nn.Linear(512, out_dim)
        )

    def forward(self, x, return_map=False):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        map_feat = self.cnn_stem(x)
        pooled = self.global_pool(map_feat)
        global_emb = self.mlp(torch.flatten(pooled, start_dim=1))

        if return_map:
            return {
                "global_feat": global_emb,
                "map_feat": map_feat,
            }
        return global_emb

class DictConditioningAdapter(nn.Module):
    def __init__(self, cond_embed_dim, max_K=None):
        super().__init__()
        self.cond_embed_dim = cond_embed_dim
        se_dim = cond_embed_dim // 2
        obstacle_dim = cond_embed_dim - se_dim

        # Start/end encoder: input is [start_x, start_y, end_x, end_y]
        self.se_encoder = nn.Sequential(
            nn.Linear(4, se_dim),
            nn.Mish(),
            nn.Linear(se_dim, se_dim),
        )

        # Obstacle encoder: maps (B, K, 2) to (B, obstacle_dim)
        self.obstacle_env_encoder = ObstacleEncoder(out_dim=obstacle_dim)

    def forward(self, cond_dict):
        # start/end
        se = torch.cat([cond_dict["start"], cond_dict["end"]], dim=-1)  # (B, 4)
        se_emb = self.se_encoder(se)

        # obstacle locations (kept as obstacle_mask key for compatibility)
        obstacle_env = cond_dict["obstacle_mask"].float()
        obstacle_encoded = self.obstacle_env_encoder(obstacle_env, return_map=True)
        obstacle_env_emb = obstacle_encoded["global_feat"]
        map_feat = obstacle_encoded["map_feat"]

        if "presence_flag" in cond_dict:
            B = obstacle_env_emb.shape[0]
            flag = cond_dict["presence_flag"].view(B, 1).float()
            obstacle_env_emb = obstacle_env_emb * flag
            map_feat = map_feat * flag.view(B, 1, 1, 1)

        cond_feat = torch.cat([se_emb, obstacle_env_emb], dim=-1)

        return {
            "global_feat": cond_feat,
            "map_feat": map_feat,
        }


class ConditionalUnet1D(nn.Module):
    def __init__(self,
                 input_dim,
                 global_cond_dim=None, # We keep this for backward compatibility or metadata
                 max_K=None,           # Pass this explicitly now
                 diffusion_step_embed_dim=256,
                 cond_embed_dim=128,
                 down_dims=[256, 512, 1024],
                 kernel_size=5,
                 n_groups=8,
                 attention=False,
                 local_map_dim=64,
                 coord_bounds=(-0.5, 0.5, -0.95, 0.05)):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        self.cond_embed_dim = cond_embed_dim
        self.local_map_dim = local_map_dim
        self.coord_bounds = coord_bounds

        # Rectangle-footprint local feature sampling
        self.rect_length = 0.118
        self.rect_width = 0.014
        self.rect_center_offset = 0.03  # Offset between control point and rectangle center
        self.footprint_n_len = 9
        self.footprint_n_width = 3

        # map_feat has 32 channels from ObstacleEncoder.cnn_stem
        self.map_feat_dim = 32

        self.footprint_projector = nn.Sequential(
            nn.Linear(self.map_feat_dim * 3, self.local_map_dim),
            nn.Mish(),
            nn.Linear(self.local_map_dim, self.local_map_dim),
        )

        all_dims = [input_dim + self.local_map_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim

        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

       

        # NEW: The Adapter replaces the manual slicing logic
        self.cond_adapter = DictConditioningAdapter(cond_embed_dim)

        cond_dim = dsed + cond_embed_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        # NEW: optional attention at bottleneck
        self.mid_attn = Residual(
            PreNorm(mid_dim, LinearAttention(mid_dim), n_groups=n_groups)
        ) if attention else nn.Identity()


        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out * 2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        # print("number of parameters: {:e}".format(
        #     sum(p.numel() for p in self.parameters()))
        # )

    def encode_global_condition(self, global_cond):
        if global_cond is None:
            return None
        if isinstance(global_cond, dict):
            if "global_feat" in global_cond and "map_feat" in global_cond:
                return global_cond
            return self.cond_adapter(global_cond)
        if torch.is_tensor(global_cond):
            return {
                "global_feat": global_cond,
                "map_feat": None,
            }
        raise ValueError("global_cond must be a dict, tensor, or None.")

    def _coords_to_grid(self, xy):
        """
        xy: (B, T, 2) in physical coordinates.
        Converts to grid_sample coordinates in [-1, 1] using self.coord_bounds.
        """
        x_min, x_max, y_min, y_max = self.coord_bounds

        x = xy[..., 0]
        y = xy[..., 1]

        x = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
        y = 2.0 * (y - y_min) / (y_max - y_min) - 1.0

        return torch.stack((x, y), dim=-1).clamp(-1.0, 1.0)
    
    def _make_rectangle_edge_points(self,  device="cuda", dtype=torch.float32):
        xs = torch.linspace(
            -self.rect_length / 2,
            self.rect_length / 2,
            self.footprint_n_len,
            device=device,
            dtype=dtype,
        )
        ys = torch.linspace(
            -self.rect_width / 2,
            self.rect_width / 2,
            self.footprint_n_width,
            device=device,
            dtype=dtype,
        )

        top = torch.stack([xs, torch.full_like(xs, self.rect_width / 2)], dim=-1)
        bottom = torch.stack([xs, torch.full_like(xs, -self.rect_width / 2)], dim=-1)
        left = torch.stack([torch.full_like(ys, -self.rect_length / 2), ys], dim=-1)
        right = torch.stack([torch.full_like(ys, self.rect_length / 2), ys], dim=-1)

        edge_pts = torch.cat([top, bottom, left, right], dim=0)
        edge_pts = torch.unique(edge_pts, dim=0)

        return edge_pts

    def _sample_local_map_features(self, map_feat, sample):
        """
        map_feat: (B, C, H, W)
        sample:   (B, T, 4) -> x, y, sin(theta), cos(theta)

        returns:  (B, T, local_map_dim)
        """
        B, C, H, W = map_feat.shape
        B2, T, D = sample.shape

        if B != B2:
            raise ValueError(f"Batch mismatch: map_feat batch {B} vs sample batch {B2}")

        xy = sample[..., :2]  # (B, T, 2)

        # -----------------------------
        # 1) Center feature
        # -----------------------------
        center_grid = self._coords_to_grid(xy).unsqueeze(2)  # (B, T, 1, 2)

        center_feat = torch.nn.functional.grid_sample(
            map_feat,
            center_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )  # (B, C, T, 1)

        center_feat = center_feat.squeeze(-1).moveaxis(1, 2)  # (B, T, C)

        # -----------------------------
        # 2) Rectangle footprint points
        # -----------------------------
        if D < 4:
            return center_feat

        sincos = sample[..., 2:4]
        sincos = sincos / (torch.norm(sincos, dim=-1, keepdim=True) + 1e-8)

        s = sincos[..., 0]  # (B, T)
        c = sincos[..., 1]  # (B, T)

        # Apply offset to get rectangle center from reference point for footprint sampling
        offset_direction = torch.stack([c, s], dim=-1)  # (B, T, 2)
        xy_center = xy + self.rect_center_offset * offset_direction

        local_pts = self._make_rectangle_edge_points(
            device=sample.device,
            dtype=sample.dtype,
        )  # (P, 2)

        lx = local_pts[:, 0][None, None, :]  # (1, 1, P)
        ly = local_pts[:, 1][None, None, :]  # (1, 1, P)

        wx = xy_center[..., 0:1] + c[..., None] * lx - s[..., None] * ly
        wy = xy_center[..., 1:2] + s[..., None] * lx + c[..., None] * ly

        footprint_pts = torch.stack([wx, wy], dim=-1)  # (B, T, P, 2)

        P = footprint_pts.shape[2]
        footprint_pts_flat = footprint_pts.reshape(B, T * P, 2)

        footprint_grid = self._coords_to_grid(footprint_pts_flat).unsqueeze(2)  # (B, T*P, 1, 2)

        footprint_feat = torch.nn.functional.grid_sample(
            map_feat,
            footprint_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )  # (B, C, T*P, 1)

        footprint_feat = footprint_feat.squeeze(-1).moveaxis(1, 2)  # (B, T*P, C)
        footprint_feat = footprint_feat.reshape(B, T, P, C)         # (B, T, P, C)

        # -----------------------------
        # 3) Summarize footprint features
        # -----------------------------
        footprint_mean = footprint_feat.mean(dim=2)       # (B, T, C)
        footprint_max = footprint_feat.max(dim=2).values  # (B, T, C)

        combined = torch.cat(
            [center_feat, footprint_mean, footprint_max],
            dim=-1,
        )  # (B, T, 3C)

        local_feats = self.footprint_projector(combined)  # (B, T, local_map_dim)

        return local_feats

    def forward(self,
                sample: torch.Tensor,
                timestep: Union[torch.Tensor, float, int],
                global_cond=None):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        sample_xy = sample[..., :2]

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.float32, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        cond_feat = self.encode_global_condition(global_cond)

        if cond_feat is None:
            cond_vec = torch.zeros((sample.shape[0], self.cond_embed_dim), device=sample.device, dtype=global_feature.dtype)
            local_feats = torch.zeros(
                (sample.shape[0], sample.shape[1], self.local_map_dim),
                device=sample.device,
                dtype=sample.dtype,
            )
        else:
            cond_vec = cond_feat["global_feat"].to(device=sample.device, dtype=global_feature.dtype)
            map_feat = cond_feat["map_feat"]

            if cond_vec.ndim != 2:
                raise ValueError(f"Encoded conditioning must be rank-2 (B,C), got shape {cond_vec.shape}")
            if cond_vec.shape[0] == 1 and sample.shape[0] > 1:
                cond_vec = cond_vec.expand(sample.shape[0], -1)
            elif cond_vec.shape[0] != sample.shape[0]:
                raise ValueError(
                    f"Conditioning batch size mismatch: cond {cond_vec.shape[0]} vs sample {sample.shape[0]}"
                )

            if map_feat is None:
                local_feats = torch.zeros(
                    (sample.shape[0], sample.shape[1], self.local_map_dim),
                    device=sample.device,
                    dtype=sample.dtype,
                )
            else:
                map_feat = map_feat.to(device=sample.device, dtype=sample.dtype)
                local_feats = self._sample_local_map_features(map_feat, sample)

        global_feature = torch.cat([
            global_feature, cond_vec
        ], axis=-1)

        sample_plus_local = torch.cat([sample, local_feats], dim=-1)
        x = sample_plus_local.moveaxis(-1, -2)
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        #for mid_module in self.mid_modules:      # BEFORE ATTENTION WAS ADDED
        #    x = mid_module(x, global_feature)

        # mid / bottleneck
        x = self.mid_modules[0](x, global_feature)
        x = self.mid_attn(x)                  # <--- global attention here    TODO: do it not hard coded
        x = self.mid_modules[1](x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1, -2)
        # (B,T,C)
        return x