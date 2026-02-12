import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from segmentation_models_pytorch.losses import DiceLoss
# RCB (Residual Convolution Block)

class RES_Dilated_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RES_Dilated_Conv,self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same", bias=False) #dilation=1
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same", dilation=2, bias=False) #dilation=2
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same", dilation=3, bias=False) #dilation=3
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same", bias=False) #dilation=1
        
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.bn4 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1) 
        x1 = F.gelu(x1) 
        x1 = F.dropout(x1, p=0.1) 

        x2 = self.conv2(x)
        x2 = self.bn2(x2)
        x2 = F.gelu(x2)
        x2 = F.dropout(x2, p=0.1)

        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = F.gelu(x3)
        x3 = F.dropout(x3, p=0.1)

        added = torch.add(x1, x2)
        added = torch.add(added, x3)

        x_out = self.conv4(added)
        x_out = self.bn4(x_out)
        x_out = F.gelu(x_out)
        x_out = F.dropout(x_out, p=0.1)

        residual = self.shortcut(x)  # Either identity or 1x1 conv
        x_out += residual  # Residual connection
        return x_out


class RS_Dblock(nn.Module):
    def __init__(self, channel):
        super(RS_Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1, bias=False)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2, bias=False)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4, bias=False)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8, bias=False)

        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1) 

    def forward(self, x):
        dilate1_out = self.act(self.dilate1(x))
        dilate1_out = self.dropout(dilate1_out)
        dilate1_out += x 

        dilate2_out = self.act(self.dilate2(dilate1_out))
        dilate2_out = self.dropout(dilate2_out)
        dilate2_out += dilate1_out  

        dilate3_out = self.act(self.dilate3(dilate2_out))
        dilate3_out = self.dropout(dilate3_out)
        dilate3_out += dilate2_out  

        dilate4_out = self.act(self.dilate4(dilate3_out))
        dilate4_out = self.dropout(dilate4_out)
        dilate4_out += dilate3_out  

        out = x + dilate4_out
        return out


class Upsample(nn.Module):

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=True)
        return x


class DecoderBlock(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels):
        super(DecoderBlock, self).__init__()

        self.identity = nn.Sequential(
            Upsample(2, mode="bilinear"),
            nn.Conv2d(input_channels, output_channels, kernel_size=1, padding=0)
        )

 
        self.decodeD = nn.Sequential(
            Upsample(2, mode="bilinear"),
            ResBlock_CBAM(input_channels, output_channels//4),
            nn.BatchNorm2d(output_channels),
        )

    def forward(self, x):
        residual = self.identity(x)
        out = self.decodeD(x)
        out = F.interpolate(out, size=residual.shape[2:], mode='bilinear', align_corners=False)
        out += residual

        return out


class GatedFusion(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.gate = nn.Sequential(
            # nn.BatchNorm2d(in_channels),
            nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, padding=0),
            # nn.BatchNorm2d(in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        out = torch.cat([x, y], dim=1)
        G = self.gate(out)

        PG = x * G
        FG = y * (1 - G)

        return FG + PG



class FCM(nn.Module):
    """Feature Complementary Module.
    Channel domain: mutual cross-modal channel recalibration.
    Spatial domain: shared spatial gating from both modalities.
    gamma=0 init ensures identity at training start.
    """
    def __init__(self, dim):
        super().__init__()
        r = max(dim // 4, 16)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ch_fc = nn.Sequential(
            nn.Linear(dim * 2, r),
            nn.ReLU(inplace=True),
            nn.Linear(r, dim * 2),
            nn.Sigmoid()
        )
        self.sp_conv = nn.Sequential(
            nn.Conv2d(dim * 2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.proj_opt = nn.Conv2d(dim, dim, 1, bias=False)
        self.proj_sar = nn.Conv2d(dim, dim, 1, bias=False)
        self.gamma_opt = nn.Parameter(torch.zeros(1))
        self.gamma_sar = nn.Parameter(torch.zeros(1))

    def forward(self, x_opt, x_sar):
        B, C, H, W = x_opt.shape
        cat = torch.cat([x_opt, x_sar], dim=1)
        ch_w = self.ch_fc(self.gap(cat).flatten(1)).view(B, 2 * C, 1, 1)
        opt_ch = x_opt * ch_w[:, :C]
        sar_ch = x_sar * ch_w[:, C:]
        sp_mask = self.sp_conv(cat)
        opt_sp = x_opt * sp_mask
        sar_sp = x_sar * sp_mask
        out_opt = x_opt + self.gamma_opt * self.proj_opt(opt_ch + opt_sp)
        out_sar = x_sar + self.gamma_sar * self.proj_sar(sar_ch + sar_sp)
        return out_opt, out_sar


class RFI(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            nn.Sigmoid()
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dyn_fc = nn.Sequential(
            nn.Linear(dim * 2, max(dim // 2, 16)),
            nn.ReLU(inplace=True),
            nn.Linear(max(dim // 2, 16), 2)
        )
        self.refine = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x_opt, x_sar):
        B, C, H, W = x_opt.shape
        cat = torch.cat([x_opt, x_sar], dim=1)
        g = self.gate(cat)
        base = x_opt * g + x_sar * (1 - g)
        w = torch.softmax(self.dyn_fc(self.gap(cat).flatten(1)), dim=1)
        dyn = w[:, 0:1, None, None] * x_opt + w[:, 1:2, None, None] * x_sar
        dyn = self.refine(dyn)
        return base + self.gamma * dyn

# Define a simple learnable scale parameter (replacement for mmcv's Scale)
class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self, x):
        return x * self.scale

# ConvModule replacement
class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, norm_cfg=True, act_cfg=True):
        super(ConvModule, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)]
        if norm_cfg:
            layers.append(nn.BatchNorm2d(out_channels))
        if act_cfg:
            layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class CEB(nn.Module):
    """Channel Enhanced Block (from SFMFusion, TIP 2025)"""
    def __init__(self, num_feat):
        super(CEB, self).__init__()
        self.num_feat = num_feat
        self.conv = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        half = num_feat // 2
        self.mlp_avg = nn.Sequential(
            nn.Conv2d(half, half, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(half, half, kernel_size=1)
        )
        self.mlp_max = nn.Sequential(
            nn.Conv2d(half, half, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(half, half, kernel_size=1)
        )
        self.sigmoid_avg = nn.Sigmoid()
        self.sigmoid_max = nn.Sigmoid()
        idx = torch.arange(num_feat).reshape(2, num_feat // 2).t().reshape(-1)
        self.register_buffer('shuffle_idx', idx)

    def forward(self, x):
        x = self.conv(x)
        skip = x
        C = self.num_feat
        x1, x2 = torch.split(x, C // 2, dim=1)
        y1 = self.sigmoid_avg(self.mlp_avg(self.avg_pool(x1)))
        y2 = self.sigmoid_max(self.mlp_max(self.max_pool(x2)))
        z = torch.cat((x1 * y1, x2 * y2), dim=1)
        z = z[:, self.shuffle_idx, :, :]
        z = z + skip
        return z


class LayerNorm2d(nn.Module):
    """Channel-wise Layer Normalization for 2D feature maps."""
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[None, :, None, None] * x + self.bias[None, :, None, None]


class FreMLP(nn.Module):
    def __init__(self, nc, expand=1):
        super(FreMLP, self).__init__()
        self.process_mag = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * nc, nc, 1, 1, 0)
        )
        self.process_pha = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * nc, nc, 1, 1, 0)
        )

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')

        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)

        mag = self.process_mag(mag)
        pha = self.process_pha(pha)

        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        
        return x_out


class SRB(nn.Module):

    def __init__(self, in_channels):
        super(SRB, self).__init__()

        self.process_ll = FreMLP(in_channels)

        self.process_high = FreMLP(in_channels * 3)
        
        self.refine = nn.Conv2d(in_channels, in_channels, 1)
 
        self.gamma = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def dwt_2d(self, x):
        # Haar DWT
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]
        
        x_ll = (x00 + x01 + x10 + x11) / 4.0
        x_lh = (x00 + x01 - x10 - x11) / 2.0
        x_hl = (x00 - x01 + x10 - x11) / 2.0
        x_hh = (x00 - x01 - x10 + x11) / 2.0
        
        return x_ll, x_lh, x_hl, x_hh

    def idwt_2d(self, ll, lh, hl, hh):
        # Haar IDWT
        x00 = ll + lh + hl + hh
        x01 = ll + lh - hl - hh
        x10 = ll - lh + hl - hh
        x11 = ll - lh - hl + hh
        
        B, C, H, W = ll.shape
        out = torch.zeros(B, C, H * 2, W * 2, device=ll.device, dtype=ll.dtype)
        out[:, :, 0::2, 0::2] = x00
        out[:, :, 0::2, 1::2] = x01
        out[:, :, 1::2, 0::2] = x10
        out[:, :, 1::2, 1::2] = x11
        return out

    def forward(self, x):
        ll, lh, hl, hh = self.dwt_2d(x)
        
        ll_enhanced = self.process_ll(ll) + ll

        highs = torch.cat([lh, hl, hh], dim=1)      # (B, 3C, H/2, W/2)
        highs_enhanced = self.process_high(highs) + highs
        
        lh_new, hl_new, hh_new = torch.chunk(highs_enhanced, 3, dim=1)
        
        out = self.idwt_2d(ll_enhanced, lh_new, hl_new, hh_new)
        
        if out.shape != x.shape:
             out = out[:, :, :x.shape[2], :x.shape[3]]
             
        return self.gamma * self.refine(out) + x

SFE = SRB


class ResBlock_CBAM(nn.Module):
    """Ours: CEB (channel enhancement) only, SFE applied only at final decoder output"""
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=4):
        super(ResBlock_CBAM, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places * self.expansion),
        )
        self.ceb = CEB(places * self.expansion)

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places * self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        out = self.ceb(out)
        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out



from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding


# AlignDecoderBlock
class AlignDecoderBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(AlignDecoderBlock, self).__init__()
        self.up = ConvModule(input_channels, output_channels, kernel_size=1)
        self.identity_conv = nn.Conv2d(input_channels, output_channels, kernel_size=1, padding=0)
        self.decode = nn.Sequential(
            # nn.BatchNorm2d(input_channels),
            nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(output_channels)
        )
        
        self.decodeD =ResBlock_CBAM(input_channels, output_channels//4)
        self.up1 = CAB(input_channels)
        
    def forward(self, low_feat, high_feat):
        f = self.up1(low_feat, high_feat)

        out = self.decodeD(f)

        return out



import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class CCDC(nn.Module):
    def __init__(self,
                 enc_opt_dims=[64, 256, 512, 1024, 2048],
                 enc_sar_dims=[64, 256, 512, 1024, 2048],
                 center_block="dblock",
                 side_dim=64,
                 att_dim_factor=2,
                 norm_cfg=dict(type='BN')
                 ):
        super(CCDC, self).__init__()

        self.backbone_opt = smp.encoders.get_encoder(
            name='resnet50',
            in_channels=4,
            depth=5,
            weights="imagenet"
        )

        self.backbone_sar = smp.encoders.get_encoder(
            name='resnet50',
            in_channels=1,
            depth=5,
            weights='imagenet'
        )


        # bridge module (center block)
        if center_block == 'dblock':
            self.center_opt = RS_Dblock(enc_opt_dims[-1])
            self.center_sar = RS_Dblock(enc_sar_dims[-1])
        else:
            self.center_opt = nn.Identity()
            self.center_sar = nn.Identity()

        self.side1_rgb = RES_Dilated_Conv(enc_opt_dims[0], side_dim)
        self.side2_rgb = RES_Dilated_Conv(enc_opt_dims[1], side_dim)
        self.side3_rgb = RES_Dilated_Conv(enc_opt_dims[2], side_dim)
        self.side4_rgb = RES_Dilated_Conv(enc_opt_dims[3], side_dim)
        self.side5_rgb = RES_Dilated_Conv(enc_opt_dims[4], side_dim)

        self.side1_sar = RES_Dilated_Conv(enc_sar_dims[0], side_dim)
        self.side2_sar = RES_Dilated_Conv(enc_sar_dims[1], side_dim)
        self.side3_sar = RES_Dilated_Conv(enc_sar_dims[2], side_dim)
        self.side4_sar = RES_Dilated_Conv(enc_sar_dims[3], side_dim)
        self.side5_sar = RES_Dilated_Conv(enc_sar_dims[4], side_dim)

        # FCM: Feature Complementary Module at each scale
        self.fcm1 = FCM(side_dim)
        self.fcm2 = FCM(side_dim)
        self.fcm3 = FCM(side_dim)
        self.fcm4 = FCM(side_dim)
        self.fcm5 = FCM(side_dim)

        # RFI: Refined Feature Integration → single fused stream for decoder
        self.rfi1 = RFI(side_dim)
        self.rfi2 = RFI(side_dim)
        self.rfi3 = RFI(side_dim)
        self.rfi4 = RFI(side_dim)
        self.rfi5 = RFI(side_dim)

        # Single decoder using DFF fused features (CEB inside ResBlock_CBAM, SFE at final output)
        self.decode1 = DecoderBlock(side_dim, side_dim)
        self.decode2 = AlignDecoderBlock(side_dim, side_dim)
        self.decode3 = AlignDecoderBlock(side_dim, side_dim)
        self.decode4 = AlignDecoderBlock(side_dim, side_dim)
        self.decode5 = AlignDecoderBlock(side_dim, side_dim)
        
        # SFE applied only at final decoder output
        self.sfe_final = SFE(side_dim)
        
        
        
    def forward(self, x):
        # Split RGB (4 channels) and SAR (1 channel)
        x_img, x_aux = torch.split(x, (4, 1), dim=1)

        # RGB encoding
        x0,x1, x2, x3, x4, x5 = self.backbone_opt(x_img)
        
        x5 = self.center_opt(x5)
        x1_side = self.side1_rgb(x1)
        x2_side = self.side2_rgb(x2)
        x3_side = self.side3_rgb(x3)
        x4_side = self.side4_rgb(x4)
        x5_side = self.side5_rgb(x5)

        # SAR encoding
        y0,y1, y2, y3, y4, y5 = self.backbone_sar(x_aux)
        y5 = self.center_sar(y5)
        y1_side = self.side1_sar(y1)
        y2_side = self.side2_sar(y2)
        y3_side = self.side3_sar(y3)
        y4_side = self.side4_sar(y4)
        y5_side = self.side5_sar(y5)

        # FCM + RFI at each scale (channel and spatial fusion per scale)
        x5_side, y5_side = self.fcm5(x5_side, y5_side)
        f5 = self.rfi5(x5_side, y5_side)

        x4_side, y4_side = self.fcm4(x4_side, y4_side)
        f4 = self.rfi4(x4_side, y4_side)

        x3_side, y3_side = self.fcm3(x3_side, y3_side)
        f3 = self.rfi3(x3_side, y3_side)

        x2_side, y2_side = self.fcm2(x2_side, y2_side)
        f2 = self.rfi2(x2_side, y2_side)

        x1_side, y1_side = self.fcm1(x1_side, y1_side)
        f1 = self.rfi1(x1_side, y1_side)

        # Single decoder (CEB active inside ResBlock_CBAM, SFE applied at final output)
        out = self.decode5(f4, f5)
        out = self.decode4(f3, out)
        out = self.decode3(f2, out)
        out = self.decode2(f1, out)
        out = self.decode1(out)
        
        # Apply SFE at final decoder output
        out = self.sfe_final(out)

        return out


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)
        


class CISRNet(nn.Module):
    def __init__(self,num_classes=21843):
        super(CISRNet, self).__init__()
        self.num_classes = num_classes
        self.network = CCDC()
        self.segmentation_head = SegmentationHead(
            in_channels=64,
            out_channels=self.num_classes,
            kernel_size=3,
        )

    def forward(self, images, masks=None):  
        
        images = self.network(images)  # (B, n_patch, hidden)
        logits = self.segmentation_head(images) 
        if masks != None:  
            loss1 = DiceLoss(mode='binary')(logits, masks)  
            loss2 = nn.BCEWithLogitsLoss()(logits, masks)  
            
            return logits, loss1, loss2   
        return logits


class CAB(nn.Module):
    def __init__(self, features, norm_cfg=dict(type='BN', requires_grad=True)):
        super(CAB, self).__init__()

        # Replacing build_norm_layer with nn.BatchNorm2d
        self.delta_gen1 = nn.Sequential(
            nn.Conv2d(features * 2, features, kernel_size=1, bias=False),
            nn.BatchNorm2d(features),  # Batch Normalization
            nn.Conv2d(features, 2, kernel_size=3, padding=1, bias=False)
        )

        self.delta_gen2 = nn.Sequential(
            nn.Conv2d(features * 2, features, kernel_size=1, bias=False),
            nn.BatchNorm2d(features),  # Batch Normalization
            nn.Conv2d(features, 2, kernel_size=3, padding=1, bias=False)
        )

        self.delta_gen1[2].weight.data.zero_()
        self.delta_gen2[2].weight.data.zero_()

    def bilinear_interpolate_torch_gridsample(self, input, size, delta=0):
        out_h, out_w = size
        n, c, h, w = input.shape
        s = 1.0
        norm = torch.tensor([[[[w / s, h / s]]]]).type_as(input).to(input.device)
        w_list = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h_list = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h_list.unsqueeze(2), w_list.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + delta.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output

    def forward(self, low_stage, high_stage):
        h, w = low_stage.size(2), low_stage.size(3)
        high_stage = F.interpolate(input=high_stage, size=(h, w), mode='bilinear', align_corners=True)

        concat = torch.cat((low_stage, high_stage), 1)
        delta1 = self.delta_gen1(concat)
        delta2 = self.delta_gen2(concat)
        high_stage = self.bilinear_interpolate_torch_gridsample(high_stage, (h, w), delta1)
        low_stage = self.bilinear_interpolate_torch_gridsample(low_stage, (h, w), delta2)

        high_stage += low_stage
        return high_stage



