"""
=========================================================
MobileNet（严格结构对应 + 中文注释 + 可运行）
---------------------------------------------------------
我们实现两种经典版本：

(1) MobileNetV1（Depthwise Separable Convolution）
    核心：把标准卷积分解为：
      - Depthwise Conv：每个输入通道单独做 3x3 卷积（groups=in_channels）
      - Pointwise Conv：1x1 卷积做通道混合
    计算量显著降低。

(2) MobileNetV2（Inverted Residual + Linear Bottleneck）
    核心 block：InvertedResidual
      - 1x1 expand（通道扩张 t 倍）
      - 3x3 depthwise
      - 1x1 project（线性，不用 ReLU，避免信息损失）
      - 若 stride=1 且输入输出通道相同：残差连接

你可以像 ResNet/Swin 一样在 forward 中打印 shape（可开关 verbose）。
=========================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# 通用小组件：Conv-BN-ReLU
# =========================================================
class ConvBNReLU(nn.Module):
    """
    标准的 Conv + BN + ReLU6（MobileNet 常用 ReLU6）
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# =========================================================
# (A) MobileNetV1：Depthwise Separable Convolution
# =========================================================
class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Conv：
      depthwise 3x3（groups=in_ch） + pointwise 1x1
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        # depthwise：每个通道独立卷积
        self.dw = ConvBNReLU(in_ch, in_ch, kernel_size=3, stride=stride, groups=in_ch)
        # pointwise：1x1 混合通道
        self.pw = ConvBNReLU(in_ch, out_ch, kernel_size=1, stride=1, groups=1)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return x


class MobileNetV1(nn.Module):
    """
    MobileNetV1（严格结构对应）：
      - 第一层：普通 3x3 conv stride=2
      - 后续堆叠：DepthwiseSeparableConv，按论文配置进行下采样
      - 最后：全局平均池化 + fc

    说明：典型输入 224x224 时，最终特征为 1024 通道。
    """
    def __init__(self, num_classes=1000, width_mult=1.0):
        super().__init__()
        def c(ch):  # width multiplier 控制通道数
            return int(ch * width_mult)

        self.stem = ConvBNReLU(3, c(32), kernel_size=3, stride=2)

        # 经典 MobileNetV1 配置（通道 & stride）
        cfg = [
            (64, 1),
            (128, 2),
            (128, 1),
            (256, 2),
            (256, 1),
            (512, 2),
            (512, 1),
            (512, 1),
            (512, 1),
            (512, 1),
            (512, 1),
            (1024, 2),
            (1024, 1),
        ]

        layers = []
        in_ch = c(32)
        for out_ch, stride in cfg:
            layers.append(DepthwiseSeparableConv(in_ch, c(out_ch), stride=stride))
            in_ch = c(out_ch)
        self.features = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c(1024), num_classes)

    def forward(self, x, verbose=False):
        if verbose: print("Input:", x.shape)
        x = self.stem(x)
        if verbose: print("Stem:", x.shape)

        for i, layer in enumerate(self.features):
            x = layer(x)
            if verbose and (i in [0, 1, 3, 5, 11, 12]):  # 打印关键下采样点
                print(f"Feature[{i}] ->", x.shape)

        x = self.pool(x)
        if verbose: print("GAP:", x.shape)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        if verbose: print("FC:", x.shape)
        return x


# =========================================================
# (B) MobileNetV2：Inverted Residual + Linear Bottleneck
# =========================================================
class InvertedResidual(nn.Module):
    """
    MobileNetV2 的核心 Block（严格对应论文）：
      输入 x: [B, Cin, H, W]

      1) expand 1x1（Cin -> Cin*t）  （t=expand_ratio）
      2) depthwise 3x3（groups=hidden）
      3) project 1x1（hidden -> Cout）  注意：这里是线性的（不加 ReLU）
      4) 若 stride=1 且 Cin=Cout：残差连接

    输出：
      [B, Cout, H/stride, W/stride]
    """
    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super().__init__()
        assert stride in [1, 2]
        hidden = int(round(in_ch * expand_ratio))
        self.use_res = (stride == 1 and in_ch == out_ch)

        layers = []
        # (1) expand：如果 expand_ratio=1，则不需要 expand
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_ch, hidden, kernel_size=1, stride=1))

        # (2) depthwise 3x3
        layers.append(ConvBNReLU(hidden, hidden, kernel_size=3, stride=stride, groups=hidden))

        # (3) project 1x1（线性：不加 ReLU6）
        layers.append(nn.Conv2d(hidden, out_ch, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(out_ch))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    """
    MobileNetV2（严格结构对应）：
      - Stem：3x3 conv stride=2
      - 多个 stage：InvertedResidual（t,c,n,s）
      - Head：1x1 conv 到 1280（可随 width_mult 调整）
      - GAP + fc

    论文经典配置（t,c,n,s）：
      [1,  16, 1, 1],
      [6,  24, 2, 2],
      [6,  32, 3, 2],
      [6,  64, 4, 2],
      [6,  96, 3, 1],
      [6, 160, 3, 2],
      [6, 320, 1, 1],
    """
    def __init__(self, num_classes=1000, width_mult=1.0, round_nearest=8):
        super().__init__()

        def _make_divisible(v, divisor=8):
            # 常见做法：让通道数对齐到 8 的倍数，利于硬件加速
            return int((v + divisor / 2) // divisor * divisor)

        input_ch = _make_divisible(32 * width_mult, round_nearest)
        last_ch  = _make_divisible(1280 * max(1.0, width_mult), round_nearest)

        self.stem = ConvBNReLU(3, input_ch, kernel_size=3, stride=2)

        cfg = [
            # t,  c,   n, s
            (1,  16,  1, 1),
            (6,  24,  2, 2),
            (6,  32,  3, 2),
            (6,  64,  4, 2),
            (6,  96,  3, 1),
            (6, 160,  3, 2),
            (6, 320,  1, 1),
        ]

        layers = []
        in_ch = input_ch
        for t, c, n, s in cfg:
            out_ch = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedResidual(in_ch, out_ch, stride=stride, expand_ratio=t))
                in_ch = out_ch

        self.features = nn.Sequential(*layers)

        # head：1x1 conv
        self.head = ConvBNReLU(in_ch, last_ch, kernel_size=1, stride=1)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(last_ch, num_classes)

    def forward(self, x, verbose=False):
        if verbose: print("Input:", x.shape)
        x = self.stem(x)
        if verbose: print("Stem :", x.shape)

        # 打印关键下采样节点（仅示例）
        for i, layer in enumerate(self.features):
            x = layer(x)
            if verbose and i in [0, 2, 5, 9, 16]:  # 这些位置大致对应几个 stage 的末尾
                print(f"IR[{i}] :", x.shape)

        x = self.head(x)
        if verbose: print("Head :", x.shape)

        x = self.pool(x)
        if verbose: print("GAP  :", x.shape)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        if verbose: print("FC   :", x.shape)
        return x


# =========================================================
# Quick test
# =========================================================
if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224)

    print("==== MobileNetV1 ====")
    m1 = MobileNetV1(num_classes=10, width_mult=1.0)
    y1 = m1(x, verbose=True)
    print("Output:", y1.shape)

    print("\n==== MobileNetV2 ====")
    m2 = MobileNetV2(num_classes=10, width_mult=1.0)
    y2 = m2(x, verbose=True)
    print("Output:", y2.shape)
