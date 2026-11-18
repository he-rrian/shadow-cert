"""w5_model.py -- depthwise-separable U-Net for S5 residual phase restoration.

Input  (7 channels): I/255, sin(h-carrier), cos(h-carrier), roi, M_fix, sin(carrier), cos(carrier)
Output (7 channels): dpsi, log_sigma, A_logit, B_logit, 3-class logits (P/U/R)

The hard-boundary formula is applied OUTSIDE the network:
    psi = h + M_fix * dpsi
so the network only learns the residual inside the blend mask.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def gn(ch, groups=8):
    g = min(groups, ch)
    while ch % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, ch)


class DSConv(nn.Module):
    """depthwise 3x3 + pointwise 1x1 + GroupNorm + SiLU"""
    def __init__(self, cin, cout):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.norm = gn(cout)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.pw(self.dw(x))))


class Up(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DSConv(cin, cout)
        self.conv2 = DSConv(cout, cout)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv2(self.conv(torch.cat([x, skip], 1)))


class W5UNet(nn.Module):
    def __init__(self, cin=7, base=128, cout=7):
        super().__init__()
        self.e1 = DSConv(cin, base); self.e1b = DSConv(base, base)
        self.e2 = DSConv(base, base * 2); self.e2b = DSConv(base * 2, base * 2)
        self.e3 = DSConv(base * 2, base * 4); self.e3b = DSConv(base * 4, base * 4)
        self.e4 = DSConv(base * 4, base * 8); self.e4b = DSConv(base * 8, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.d3 = Up(base * 8 + base * 4, base * 4)
        self.d2 = Up(base * 4 + base * 2, base * 2)
        self.d1 = Up(base * 2 + base, base)
        self.head = nn.Conv2d(base, cout, 1)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        e1 = self.e1b(self.e1(x))
        e2 = self.e2b(self.e2(self.pool(e1)))
        e3 = self.e3b(self.e3(self.pool(e2)))
        e4 = self.e4b(self.e4(self.pool(e3)))
        d3 = self.d3(e4, e3)
        d2 = self.d2(d3, e2)
        d1 = self.d1(d2, e1)
        return self.head(d1)


def split_output(out):
    dpsi = out[:, 0:1]
    log_sigma = out[:, 1:2].clamp(-6.0, 3.0)
    a_logit = out[:, 2:3]
    b_logit = out[:, 3:4]
    logits = out[:, 4:7]
    return dpsi, log_sigma, a_logit, b_logit, logits


def make_input(batch):
    """batch: dict of torch tensors from w5_dataset; returns 7-channel input."""
    I = batch["I"].float().unsqueeze(1) / 255.0
    h = batch["h"].float().unsqueeze(1)
    M = batch["M"].float().unsqueeze(1)
    roi = batch["roi"].float().unsqueeze(1)
    car2 = batch["carrier"].float()                 # (B,2) fy,fx
    B, _, H, W = h.shape
    yy = torch.arange(H, device=h.device, dtype=h.dtype)[None, None, :, None]
    xx = torch.arange(W, device=h.device, dtype=h.dtype)[None, None, None, :]
    car = (2 * torch.pi) * (car2[:, 0, None, None, None] * yy + car2[:, 1, None, None, None] * xx)
    dev = h - car
    return torch.cat([I, torch.sin(dev), torch.cos(dev), roi, M,
                      torch.sin(car), torch.cos(car)], 1)
