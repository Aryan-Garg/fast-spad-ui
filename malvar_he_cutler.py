from __future__ import annotations

from typing import Literal

import torch
from jaxtyping import Float
from torch import nn, Tensor
from torch.nn import functional as F

BayerPatternLiteral = Literal["gray", "grey", "rggb", "bggr", "grbg", "gbrg", "rgbg"]


class MalvarHeCutlerDemosaic(nn.Module):
    def __init__(
        self,
        bayer_pattern: BayerPatternLiteral = "rggb",
        wr: float = 0.2126,
        wg: float = 0.7152,
        wb: float = 0.0722,
    ):
        super().__init__()
        self.bayer_pattern = bayer_pattern.lower()
        self.epsilon = 1e-8
        self.padding = 2

        self.is_bayer = self.bayer_pattern not in ["gray", "grey"]

        if self.is_bayer:
            self.wr, self.wg, self.wb = wr, wg, wb

            k_g_at_rb = torch.tensor(
                [
                    [0, 0, -2, 0, 0],
                    [0, 0, 4, 0, 0],
                    [-2, 4, 8, 4, -2],
                    [0, 0, 4, 0, 0],
                    [0, 0, -2, 0, 0],
                ],
                dtype=torch.float,
            )
            k_rb_at_gr = torch.tensor(
                [
                    [0, 0, 1, 0, 0],
                    [0, -2, 0, -2, 0],
                    [-2, 8, 10, 8, -2],
                    [0, -2, 0, -2, 0],
                    [0, 0, 1, 0, 0],
                ],
                dtype=torch.float,
            )
            k_rb_at_gb = k_rb_at_gr.T
            k_rb_at_rb = torch.tensor(
                [
                    [0, 0, -3, 0, 0],
                    [0, 4, 0, 4, 0],
                    [-3, 0, 12, 0, -3],
                    [0, 4, 0, 4, 0],
                    [0, 0, -3, 0, 0],
                ],
                dtype=torch.float,
            )
            k_delta = torch.zeros((5, 5), dtype=torch.float)
            k_delta[2, 2] = 16.0

            kernels_rgb = (
                torch.stack([k_g_at_rb, k_rb_at_gr, k_rb_at_gb, k_rb_at_rb]) / 16.0
            )
            self.register_buffer("kernels_rgb", kernels_rgb.view(4, 1, 5, 5))

            k_y_r = self.wr * k_delta + self.wg * k_g_at_rb + self.wb * k_rb_at_rb
            k_y_gr = self.wr * k_rb_at_gr + self.wg * k_delta + self.wb * k_rb_at_gb
            k_y_gb = self.wr * k_rb_at_gb + self.wg * k_delta + self.wb * k_rb_at_gr
            k_y_b = self.wr * k_rb_at_rb + self.wg * k_g_at_rb + self.wb * k_delta

            kernels_y = torch.stack([k_y_r, k_y_gr, k_y_gb, k_y_b]) / 16.0
            self.register_buffer("kernels_y", kernels_y.view(4, 1, 5, 5))

            rggb_rgb_index = torch.tensor(
                [
                    [[4, 1], [2, 3]],
                    [[0, 4], [4, 0]],
                    [[3, 2], [1, 4]],
                ],
                dtype=torch.long,
            ).view(1, 3, 2, 2)

            rggb_y_index = torch.tensor([[[0, 1], [2, 3]]], dtype=torch.long).view(
                1, 1, 2, 2
            )

            norm_y_r = torch.sqrt(torch.sum(k_y_r**2))
            norm_y_gr = torch.sqrt(torch.sum(k_y_gr**2))
            norm_y_gb = torch.sqrt(torch.sum(k_y_gb**2))
            norm_y_b = torch.sqrt(torch.sum(k_y_b**2))

            rggb_norm_map = (
                torch.tensor(
                    [[[norm_y_r, norm_y_gr], [norm_y_gb, norm_y_b]]],
                    dtype=torch.float,
                )
                / 16.0
            ).view(1, 1, 2, 2)

            if self.bayer_pattern == "rggb":
                self.register_buffer("index_rgb", rggb_rgb_index)
                self.register_buffer("index_y", rggb_y_index)
                self.register_buffer("norm_y_map", rggb_norm_map)
            elif self.bayer_pattern == "bggr":
                self.register_buffer(
                    "index_rgb", torch.roll(rggb_rgb_index, (1, 1), (-1, -2))
                )
                self.register_buffer(
                    "index_y", torch.roll(rggb_y_index, (1, 1), (-1, -2))
                )
                self.register_buffer(
                    "norm_y_map", torch.roll(rggb_norm_map, (1, 1), (-1, -2))
                )
            elif self.bayer_pattern == "grbg":
                self.register_buffer("index_rgb", torch.roll(rggb_rgb_index, 1, -1))
                self.register_buffer("index_y", torch.roll(rggb_y_index, 1, -1))
                self.register_buffer("norm_y_map", torch.roll(rggb_norm_map, 1, -1))

            elif self.bayer_pattern == "gbrg":
                self.register_buffer("index_rgb", torch.roll(rggb_rgb_index, 1, -2))
                self.register_buffer("index_y", torch.roll(rggb_y_index, 1, -2))
                self.register_buffer("norm_y_map", torch.roll(rggb_norm_map, 1, -2))


        self.register_buffer("cached_index_rgb", None, persistent=False)
        self.register_buffer("cached_index_y", None, persistent=False)
        self.register_buffer("cached_norm_map", None, persistent=False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.cached_index_rgb = None
        self.cached_index_y = None
        self.cached_norm_map = None
        return self

    def _get_tiled_map(
        self,
        ref_tensor: Tensor,
        map_buffer: Tensor,
        cache_name: str,
    ) -> Tensor:
        b, _, h, w = ref_tensor.shape
        cached_map = getattr(self, cache_name)

        if (
            cached_map is not None
            and cached_map.shape == (b, map_buffer.shape[1], h, w)
            and cached_map.device == ref_tensor.device
        ):
            return cached_map

        h_rep = (h + 1) // 2
        w_rep = (w + 1) // 2

        tiled_map = map_buffer.repeat(1, 1, h_rep, w_rep)
        tiled_map = tiled_map[..., :h, :w]
        tiled_map = tiled_map.expand(b, -1, -1, -1).to(device=ref_tensor.device)

        setattr(self, cache_name, tiled_map)
        return tiled_map

    @torch.compile()
    @torch.amp.autocast(device_type="cuda", enabled=False)
    def demosaic_linear(
        self, img: Float[Tensor, "batch 1 height width"]
    ) -> Float[Tensor, "batch 3 height width"]:
        if not self.is_bayer:
            return img.repeat(1, 3, 1, 1)

        img_f32 = img.float()
        xpad = F.pad(img_f32, (self.padding,) * 4, mode="replicate")
        planes = F.conv2d(xpad, self.kernels_rgb, padding=0)
        planes = torch.cat([planes, img_f32], dim=1)

        index_map = self._get_tiled_map(img_f32, self.index_rgb, "cached_index_rgb")
        rgb = torch.gather(planes, 1, index_map)

        return rgb.to(img.dtype)

    @torch.compile()
    @torch.amp.autocast(device_type="cuda", enabled=False)
    def forward(
        self, img: Float[Tensor, "batch 1 height width"]
    ) -> Float[Tensor, "batch 1 height width"]:
        if not self.is_bayer:
            return img

        img_f32 = img.float()
        xpad = F.pad(img_f32, (self.padding,) * 4, mode="replicate")
        planes_y = F.conv2d(xpad, self.kernels_y, padding=0)

        index_map = self._get_tiled_map(img_f32, self.index_y, "cached_index_y")
        y_unnorm = torch.gather(planes_y, 1, index_map)

        norm_map = self._get_tiled_map(img_f32, self.norm_y_map, "cached_norm_map")
        return (y_unnorm / (norm_map + self.epsilon)).to(img.dtype)


_module_cache: dict[tuple[str, torch.device], MalvarHeCutlerDemosaic] = {}

# pixel_unshuffle(x, 2) maps (B,1,H,W) → (B,4,H/2,W/2) where channel k is
# the 2x2-tile position (k//2, k%2): 0=TL, 1=TR, 2=BL, 3=BR.
# For each Bayer pattern: (R_channel, (Gr_channel, Gb_channel), B_channel)
_BAYER_CHANNEL_MAP: dict[str, tuple[int, tuple[int, int], int]] = {
    "rggb": (0, (1, 2), 3),
    "bggr": (3, (2, 1), 0),
    "grbg": (1, (0, 3), 2),
    "gbrg": (2, (3, 0), 1),
}


def demosaic_malvar(
    img: Float[Tensor, "h w"],
    bayer_pattern: str,
) -> Float[Tensor, "h w 3"]:
    """Demosaic a single ``(H, W)`` mosaicked image to ``(H, W, 3)`` RGB.

    Caches the compiled :class:`MalvarHeCutlerDemosaic` module per
    ``(bayer_pattern, device)`` so ``torch.compile`` only runs once.

    :param img: Mosaicked image in ``[0, 1]``.
    :param bayer_pattern: CFA pattern, e.g. ``"rggb"``.
    :returns: Demosaiced RGB image ``(H, W, 3)`` on the same device as *img*.
    """
    key = (bayer_pattern.lower(), img.device)
    if key not in _module_cache:
        module = MalvarHeCutlerDemosaic(bayer_pattern=bayer_pattern).to(img.device)
        module.eval()
        _module_cache[key] = module
    demosaicer = _module_cache[key]
    rgb = demosaicer.demosaic_linear(img.float().unsqueeze(0).unsqueeze(0))
    return rgb.squeeze(0).permute(1, 2, 0).to(dtype=img.dtype)


def demosaic_interp(
    img: Float[Tensor, "h w"],
    bayer_pattern: str,
    mode: str = "bilinear",
) -> Float[Tensor, "h w 3"]:
    """Demosaic via per-channel upsampling (bilinear or bicubic).

    Uses :func:`torch.nn.functional.pixel_unshuffle` to extract the four
    Bayer sub-channels at half resolution, upsamples each independently,
    then combines R, mean(Gr, Gb), B into an RGB image.

    :param img: Mosaicked image in ``[0, 1]``, shape ``(H, W)``.
    :param bayer_pattern: CFA pattern, e.g. ``"rggb"``.
    :param mode: Interpolation mode — ``"bilinear"`` or ``"bicubic"``.
    :returns: Demosaiced RGB image ``(H, W, 3)``.
    """
    h, w = img.shape
    pattern = bayer_pattern.lower()

    if pattern == "rgbg":
        # Based on [3, 1, 0, 2] -> [R, G, B, G]
        r_ch, gr_ch, gb_ch, b_ch = 3, 1, 2, 0
    else:
        # Fallback to the original mapping for standard Bayer patterns
        r_ch, (gr_ch, gb_ch), b_ch = _BAYER_CHANNEL_MAP[pattern]

    x = img.float().unsqueeze(0).unsqueeze(0)
    sub = F.pixel_unshuffle(x, 2)  # (1, 4, H/2, W/2)

    up_kwargs = dict(size=(h, w), mode=mode, align_corners=False)
    r = F.interpolate(sub[:, r_ch : r_ch + 1], **up_kwargs)
    g = F.interpolate((sub[:, gr_ch : gr_ch + 1] + sub[:, gb_ch : gb_ch + 1]) / 2, **up_kwargs)
    b = F.interpolate(sub[:, b_ch : b_ch + 1], **up_kwargs)

    rgb = torch.cat([r, g, b], dim=1).squeeze(0).permute(1, 2, 0)
    return rgb.to(dtype=img.dtype)



def demosaic_robust_hybrid(
    img: Float[Tensor, "h w"],
    bayer_pattern: str,
    low_thresh: float = 0.02,   # Threshold below which shadows speckle
    high_thresh: float = 0.95,  # Threshold above which highlights clip
    blend_ksize: int = 5,       # Smooths the transition zones
) -> Float[Tensor, "h w 3"]:
    """
    Combines the sharpness of Malvar-He-Cutler with the stability of Bilinear
    interpolation in extreme high-flux and low-flux regions.
    """
    # 1. Run both of your existing demosaicking pipelines
    mhc_rgb = demosaic_malvar(img, bayer_pattern)
    bilinear_rgb = demosaic_interp(img, bayer_pattern, mode="bilinear")
    
    # 2. Create an artifact mask from the raw input image
    # Detects where pixels are dangerously close to 0 or saturation limits
    low_mask = (img < low_thresh).float()
    high_mask = (img > high_thresh).float()
    artifact_mask = torch.clamp(low_mask + high_mask, 0.0, 1.0)
    
    # Add batch/channel dimensions for PyTorch filtering: (1, 1, H, W)
    mask_tensor = artifact_mask.unsqueeze(0).unsqueeze(0)
    
    # Dilate and smooth the mask so the transition between MHC and Bilinear is invisible
    padding = blend_ksize // 2
    dilated_mask = F.max_pool2d(mask_tensor, kernel_size=blend_ksize, stride=1, padding=padding)
    smooth_mask = F.avg_pool2d(dilated_mask, kernel_size=blend_ksize, stride=1, padding=padding)
    
    # Reshape mask back to align with RGB image: (H, W, 1)
    weight = smooth_mask.squeeze(0).permute(1, 2, 0)
    
    # 3. Linear blend: Use MHC for normal details, Bilinear for trouble zones
    robust_rgb = (1.0 - weight) * mhc_rgb + weight * bilinear_rgb
    
    return robust_rgb.to(dtype=img.dtype)