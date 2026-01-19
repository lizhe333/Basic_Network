"""
=========================================================
# 论文地址：https://arxiv.org/abs/2103.14030
# 代码地址：https://github.com/microsoft/Swin-Transformer
# Swin Transformer 的主干通常是 4 个 stage：

(1) PatchEmbed：把图像切成 patch（如 4×4），投影到 embed_dim
    输出 token map： [B, H/4 * W/4, C]，也可 reshape 为 [B, H/4, W/4, C]

(2) Stage i（i=1..4）：
    含 depth[i] 个 SwinTransformerBlock
    每个 block 里有：
        LN → (W-MSA 或 SW-MSA) → 残差
        LN → MLP → 残差
    除最后一个 stage 外，stage 末尾有 PatchMerging（分辨率 /2，通道 ×2）

严格的分辨率/通道变化（以 patch_size=4 为例）：
    输入 x: [B, 3, H, W]
    PatchEmbed 后：H0=H/4, W0=W/4, C0=embed_dim

    Stage1：   [B, H0,   W0,   C0]
    Merging →  [B, H0/2, W0/2, 2*C0]
    Stage2：   [B, H0/2, W0/2, 2*C0]
    Merging →  [B, H0/4, W0/4, 4*C0]
    Stage3：   [B, H0/4, W0/4, 4*C0]
    Merging →  [B, H0/8, W0/8, 8*C0]
    Stage4：   [B, H0/8, W0/8, 8*C0]

本实现包含：
W-MSA / SW-MSA（shifted window）严格结构
attention mask（SW-MSA 必需）
相对位置偏置（relative position bias）
任意输入尺寸：自动 padding 到 window_size 整除，再还原
分类头（可选 num_classes=0 作为特征提取）

参考：Swin Transformer 原论文 / 常见官方实现逻辑（如 torchvision / timm）
=========================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# 一些小工具：DropPath（Stochastic Depth）
# ----------------------------
class DropPath(nn.Module):
    """
    Stochastic Depth：训练时按概率丢弃残差分支（按 sample 级别），推理时无效。
    """
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or (not self.training):
            return x
        keep_prob = 1.0 - self.drop_prob
        # x: [B, ..., C] -> 按 batch 维生成 mask
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_mask = torch.floor(random_tensor)
        return x / keep_prob * binary_mask


# ----------------------------
# MLP（FFN）
# ----------------------------
class Mlp(nn.Module):
    """
    Transformer 的前馈网络：
    x: [B, N, C] -> Linear(C->hidden) -> GELU -> Drop -> Linear(hidden->C) -> Drop
    """
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


# ----------------------------
# Window partition / reverse
# ----------------------------
def window_partition(x, window_size):
    """
    将特征图按窗口切分
    x: [B, H, W, C]
    -> windows: [B*nW, window_size, window_size, C]
    """
    B, H, W, C = x.shape
    assert H % window_size == 0 and W % window_size == 0
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    将窗口还原成特征图
    windows: [B*nW, window_size, window_size, C]
    -> x: [B, H, W, C]
    """
    B_ = windows.shape[0]
    nW = (H // window_size) * (W // window_size)
    B = B_ // nW
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# ----------------------------
# 构造 SW-MSA 所需的 attention mask
# ----------------------------
def build_attn_mask(H, W, window_size, shift_size, device):
    """
    构造 shifted window attention 的 mask（严格对应 Swin 思路）
    返回 attn_mask: [nW, M*M, M*M]，取值 {0, -100.0}

    解释：
    - SW-MSA 先对特征图做 cyclic shift，再切 window
    - 一个 window 里会混入多个“原区域”的 token
    - mask 用来禁止不同区域 token 之间的 attention（加一个大负数）
    """
    if shift_size == 0:
        return None

    M = window_size
    img_mask = torch.zeros((1, H, W, 1), device=device)  # [1,H,W,1]

    # 按照 window_size 和 shift_size 把 H/W 划成 3 段（这是官方常用写法）
    h_slices = (slice(0, -M),
                slice(-M, -shift_size),
                slice(-shift_size, None))
    w_slices = (slice(0, -M),
                slice(-M, -shift_size),
                slice(-shift_size, None))

    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, M)              # [nW, M, M, 1]
    mask_windows = mask_windows.view(-1, M * M)               # [nW, M*M]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, M*M, M*M]
    attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
    return attn_mask


# ----------------------------
# Window Attention（带相对位置偏置 + mask）
# ----------------------------
class WindowAttention(nn.Module):
    """
    Window-based Multi-Head Self-Attention（窗口内注意力）
    输入：
        x: [B*nW, M*M, C]
        attn_mask（可选，仅 SW-MSA 用）：[nW, M*M, M*M]
    输出：
        [B*nW, M*M, C]

    关键：
    - 相对位置偏置表 relative_position_bias_table
    - 相对位置索引 relative_position_index
    - SW-MSA 时需要把 mask 加到 attention logits 上
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # M
        self.num_heads = num_heads
        head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim 必须能整除 num_heads"
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # -------- 相对位置偏置（Swin 核心之一）--------
        # 表大小：(2M-1)*(2M-1)，每个 head 一份偏置
        M = window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * M - 1) * (2 * M - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # 预计算相对位置索引：把窗口内任意两点的相对位移映射到 table 的 index
        coords_h = torch.arange(M)
        coords_w = torch.arange(M)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # [2, M, M]
        coords_flatten = torch.flatten(coords, 1)                                # [2, M*M]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, M*M, M*M]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()            # [M*M, M*M, 2]
        relative_coords[:, :, 0] += M - 1
        relative_coords[:, :, 1] += M - 1
        relative_coords[:, :, 0] *= 2 * M - 1
        relative_position_index = relative_coords.sum(-1)                          # [M*M, M*M]
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

    def forward(self, x, attn_mask=None):
        B_, N, C = x.shape  # N = M*M
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B_, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # [B_, heads, N, N]

        # 加上相对位置偏置
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(N, N, self.num_heads).permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)  # [B_, heads, N, N]

        # SW-MSA：加 mask（阻止不同区域 token 互相注意力）
        if attn_mask is not None:
            # attn_mask: [nW, N, N]
            nW = attn_mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + attn_mask.unsqueeze(0).unsqueeze(2)  # broadcast
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)  # [B_, N, C]
        x = self.proj_drop(self.proj(x))
        return x


# ----------------------------
# Swin Transformer Block（LN + (W/SW)-MSA + LN + MLP）
# ----------------------------
class SwinTransformerBlock(nn.Module):
    """
    一个 SwinTransformerBlock（严格结构对应）：

    输入：
        x: [B, H*W, C]
    内部 reshape：
        [B, H, W, C]
    流程：
        1) LN -> W-MSA 或 SW-MSA -> 残差
        2) LN -> MLP -> 残差
    输出：
        [B, H*W, C]

    其中：
    - shift_size = 0      -> W-MSA
    - shift_size = M//2   -> SW-MSA（需要 attn_mask）
    """
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4.0, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x, H, W, attn_mask=None):
        B, L, C = x.shape
        assert L == H * W, f"L={L} 必须等于 H*W={H*W}"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # 1) SW-MSA：先 cyclic shift（沿 H/W 维滚动）
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # 2) 切 window
        x_windows = window_partition(x, self.window_size)             # [B*nW, M, M, C]
        x_windows = x_windows.view(-1, self.window_size**2, C)        # [B*nW, M*M, C]

        # 3) window 内注意力（W-MSA / SW-MSA）
        attn_windows = self.attn(x_windows, attn_mask=attn_mask)      # [B*nW, M*M, C]

        # 4) 还原 window -> feature map
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W)      # [B, H, W, C]

        # 5) 还原 cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.view(B, H * W, C)

        # 残差 #1（注意力分支）
        x = shortcut + self.drop_path(x)

        # 残差 #2（MLP 分支）
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ----------------------------
# Patch Merging（分辨率 /2，通道 ×2）
# ----------------------------
class PatchMerging(nn.Module):
    """
    PatchMerging（严格对应论文）：

    输入：
        x: [B, H*W, C]  <-> reshape [B, H, W, C]
    合并：
        以 2x2 邻域合并： (H,W)->(H/2,W/2)，通道拼接：C->4C
    线性降维：
        4C -> 2C
    输出：
        [B, (H/2)*(W/2), 2C]
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W
        # 若 H/W 为奇数，这里需要先 pad 到偶数（严格工程实现）
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x_ = x.view(B, H, W, C)
            x_ = F.pad(x_, (0, 0, 0, W % 2, 0, H % 2))  # pad W/H 到偶数
            H, W = x_.shape[1], x_.shape[2]
            x = x_.view(B, H * W, C)

        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)          # [B, H/2, W/2, 4C]
        x = x.view(B, -1, 4 * C)                         # [B, (H/2)*(W/2), 4C]
        x = self.reduction(self.norm(x))                 # [B, (H/2)*(W/2), 2C]
        return x, H // 2, W // 2


# ----------------------------
# Patch Embedding（Conv patchify）
# ----------------------------
class PatchEmbed(nn.Module):
    """
    PatchEmbed：
    - 使用 Conv2d(kernel=patch_size, stride=patch_size) 完成切 patch + 线性投影
    输入：
        [B, 3, H, W]
    输出：
        x: [B, H/ps * W/ps, embed_dim]
        Hp, Wp：patch 后的高宽
    """
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=nn.LayerNorm):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        B, C, H, W = x.shape
        # 若输入 H/W 不是 patch_size 的倍数，先 pad（工程上更稳）
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # pad right/bottom

        x = self.proj(x)  # [B, embed_dim, Hp, Wp]
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # [B, Hp*Wp, embed_dim]
        if self.norm is not None:
            x = self.norm(x)
        return x, Hp, Wp


# ----------------------------
# 一个 Stage（多层 block + 可选 PatchMerging）
# ----------------------------
class BasicLayer(nn.Module):
    """
    Swin 的一个 Stage（layer）：
    - depth 个 SwinTransformerBlock
      交替使用 W-MSA（shift=0）和 SW-MSA（shift=window//2）
    - 除最后一层外，末尾接 PatchMerging（下采样）
    """
    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4.0, drop=0.0, attn_drop=0.0, drop_path=0.0,
                 downsample=True):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.shift_size = window_size // 2

        # 给每个 block 分配 drop_path（可用 list/线性递增）
        if isinstance(drop_path, (list, tuple)):
            assert len(drop_path) == depth
            dpr = drop_path
        else:
            dpr = [drop_path for _ in range(depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            shift = 0 if (i % 2 == 0) else self.shift_size
            self.blocks.append(
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=dpr[i],
                )
            )

        self.downsample = PatchMerging(dim) if downsample else None

    def forward(self, x, H, W):
        """
        输入：
            x: [B, H*W, C]
        说明：
            - 注意力需要 H/W 能被 window_size 整除
            - 若不整除：先 padding 到整除，再做 block，最后再裁剪还原
        """
        B, L, C = x.shape
        assert L == H * W

        # -------- 1) padding 到 window_size 整除（严格工程实现）--------
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        if pad_b > 0 or pad_r > 0:
            x_ = x.view(B, H, W, C)
            x_ = F.pad(x_, (0, 0, 0, pad_r, 0, pad_b))  # pad W(右), H(下)
            Hp, Wp = x_.shape[1], x_.shape[2]
            x = x_.view(B, Hp * Wp, C)
        else:
            Hp, Wp = H, W

        # -------- 2) 构造 SW-MSA mask（只依赖 Hp/Wp，整个 stage 复用）--------
        attn_mask = build_attn_mask(
            H=Hp, W=Wp,
            window_size=self.window_size,
            shift_size=self.shift_size,
            device=x.device
        )

        # -------- 3) 依次通过 blocks（W-MSA / SW-MSA 交替）--------
        for blk in self.blocks:
            # W-MSA 的 block 会忽略 mask（因为 shift_size=0）
            x = blk(x, Hp, Wp, attn_mask=attn_mask)

        # -------- 4) 去掉 padding，还原回原始 H/W --------
        if pad_b > 0 or pad_r > 0:
            x_ = x.view(B, Hp, Wp, C)
            x_ = x_[:, :H, :W, :].contiguous()
            x = x_.view(B, H * W, C)

        # -------- 5) PatchMerging 下采样（除最后 stage 外）--------
        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)

        return x, H, W


# ----------------------------
# Swin Transformer（4 stages）
# ----------------------------
class SwinTransformer(nn.Module):
    """
    Swin Transformer 主体（严格结构对应）：
        PatchEmbed -> Stage1 -> Merge -> Stage2 -> Merge -> Stage3 -> Merge -> Stage4 -> Norm -> Head

    可选：
        num_classes=0 时不做分类头，直接输出全局特征向量（[B, C_last]）
    """
    def __init__(self,
                 patch_size=4,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=96,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 window_size=7,
                 mlp_ratio=4.0,
                 drop=0.0,
                 attn_drop=0.0,
                 drop_path_rate=0.1):
        super().__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=nn.LayerNorm)

        # 通道维度随 stage 翻倍
        dims = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]

        # 给所有 blocks 线性递增 drop_path
        total_blocks = sum(depths)
        dpr = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        cur = 0

        self.stage1 = BasicLayer(
            dim=dims[0], depth=depths[0], num_heads=num_heads[0], window_size=window_size,
            mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop,
            drop_path=dpr[cur:cur+depths[0]], downsample=True
        )
        cur += depths[0]

        self.stage2 = BasicLayer(
            dim=dims[1], depth=depths[1], num_heads=num_heads[1], window_size=window_size,
            mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop,
            drop_path=dpr[cur:cur+depths[1]], downsample=True
        )
        cur += depths[1]

        self.stage3 = BasicLayer(
            dim=dims[2], depth=depths[2], num_heads=num_heads[2], window_size=window_size,
            mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop,
            drop_path=dpr[cur:cur+depths[2]], downsample=True
        )
        cur += depths[2]

        self.stage4 = BasicLayer(
            dim=dims[3], depth=depths[3], num_heads=num_heads[3], window_size=window_size,
            mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop,
            drop_path=dpr[cur:cur+depths[3]], downsample=False
        )

        self.norm = nn.LayerNorm(dims[3])

        # 分类头：对 token 做全局平均，再线性分类
        self.head = nn.Linear(dims[3], num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, verbose=False):
        """
        x: [B,3,H,W]
        """
        x, H, W = self.patch_embed(x)  # [B, H*W, C]
        if verbose: print(f"PatchEmbed: x={tuple(x.shape)}  H,W={H},{W}")

        x, H, W = self.stage1(x, H, W)
        if verbose: print(f"Stage1   : x={tuple(x.shape)}  H,W={H},{W}")

        x, H, W = self.stage2(x, H, W)
        if verbose: print(f"Stage2   : x={tuple(x.shape)}  H,W={H},{W}")

        x, H, W = self.stage3(x, H, W)
        if verbose: print(f"Stage3   : x={tuple(x.shape)}  H,W={H},{W}")

        x, H, W = self.stage4(x, H, W)
        if verbose: print(f"Stage4   : x={tuple(x.shape)}  H,W={H},{W}")

        # 最终 norm + 全局平均池化（对 token 维度求平均）
        x = self.norm(x)          # [B, H*W, C_last]
        x = x.mean(dim=1)         # [B, C_last]
        x = self.head(x)          # [B, num_classes] 或 [B, C_last]（num_classes=0）
        return x


# ----------------------------
# 常用配置：Swin-T / S / B / L
# ----------------------------
def swin_tiny(num_classes=1000, window_size=7):
    # Swin-T: embed_dim=96, depths=(2,2,6,2), heads=(3,6,12,24)
    return SwinTransformer(
        num_classes=num_classes,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=window_size
    )

def swin_small(num_classes=1000, window_size=7):
    # Swin-S: embed_dim=96, depths=(2,2,18,2), heads=(3,6,12,24)
    return SwinTransformer(
        num_classes=num_classes,
        embed_dim=96,
        depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24),
        window_size=window_size
    )

def swin_base(num_classes=1000, window_size=7):
    # Swin-B: embed_dim=128, depths=(2,2,18,2), heads=(4,8,16,32)
    return SwinTransformer(
        num_classes=num_classes,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        window_size=window_size
    )

def swin_large(num_classes=1000, window_size=7):
    # Swin-L: embed_dim=192, depths=(2,2,18,2), heads=(6,12,24,48)
    return SwinTransformer(
        num_classes=num_classes,
        embed_dim=192,
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
        window_size=window_size
    )

if __name__ == "__main__":
    # 示例：Swin-T，输入 224x224
    model = swin_tiny(num_classes=1000, window_size=7)
    x = torch.randn(1, 3, 224, 224)
    y = model(x, verbose=True)
    print("Output:", y.shape)

    # 示例：任意尺寸（不是 window 的整数倍也能跑：内部会 pad）
    x2 = torch.randn(2, 3, 231, 197)
    y2 = model(x2, verbose=False)
    print("Output(any size):", y2.shape)
