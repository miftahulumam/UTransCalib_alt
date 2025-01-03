import math
import torch
import torch.nn as nn
# from timm.models.layers import trunc_normal_
from .encoders import conv_block
from models.realignment_layer import realignment_layer
import torch.nn.functional as F

# ============================== SUPER BASIC CNN BLOCKS ========================================== #
## SINGLE CNN LAYER (CNN + BN + ACT + DO)
class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, depthwise=False, activation='nn.SiLU(inplace=True)', drop_rate=0.1):
        super(SingleConv, self).__init__()

        if depthwise:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
                nn.BatchNorm2d(out_channels),
                eval(activation),
                nn.Dropout2d(drop_rate))
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=1, bias=False),
                nn.BatchNorm2d(out_channels),
                eval(activation),
                nn.Dropout2d(drop_rate))

    def forward(self, x):
        return self.conv(x)

## DOUBLE CNN LAYER (CNN + BN + ACT + DO)*2
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, depthwise=False, activation='nn.SiLU(inplace=True)', drop_rate=0.1):
        super(DoubleConv, self).__init__()

        if not mid_channels:
            mid_channels = out_channels

        if depthwise: # depthwise separable conv to reduce parameters
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0, groups=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                eval(activation),
                nn.Dropout2d(drop_rate),

                nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels, bias=False),
                nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0, groups=1, bias=False),
                nn.BatchNorm2d(out_channels),
                eval(activation),
                nn.Dropout2d(drop_rate)
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                eval(activation),
                nn.Dropout2d(drop_rate),

                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                eval(activation),
                nn.Dropout2d(drop_rate)
            )

    def forward(self, x):
        return self.double_conv(x)

## DOUBLE CNN LAYER (CNN + BN + ACT + DO)*2 WITH A RESIDUAL CONNECTION
class Residual_DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, depthwise=False, activation=nn.ReLU, drop_rate=0.1):
        super(Residual_DoubleConv, self).__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.double_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            activation(inplace=True),
            nn.Dropout2d(drop_rate),

            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation(inplace=True),
            nn.Dropout2d(drop_rate)
        )

        self.double_conv_2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            activation(inplace=True),
            nn.Dropout2d(drop_rate),

            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation(inplace=True),
            nn.Dropout2d(drop_rate)
        )

    def forward(self, x):
        skip_1 = x
        x = self.double_conv_1(x)
        x = skip_1 + x
        
        skip_2 = x
        x = self.double_conv_2(x)
        x = skip_2 + x
        
        return x

# ============================== END OF SUPER BASIC CNN BLOCKS ========================================== #


# =================================== MY OWN CNN BLOCKS ================================================= #
# PROPOSED FROM SCRATCH (CNN BLOCK WITH MULTIPLE DILATION RATES)
class conv_multidilation(nn.Module):
    def __init__(self, in_ch, out_ch = None, 
                 reduction = 8, act_layer='nn.SiLU(inplace=True)', drop_rate=0.1):
        super(conv_multidilation, self).__init__()

        mid_ch = in_ch // reduction
        if out_ch is None:
            out_ch = in_ch
        
        # reduction conv
        self.pointwise_in = nn.Sequential(nn.Conv2d(in_ch, mid_ch,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0,
                                                    groups=1,
                                                    bias=False),
                                          nn.BatchNorm2d(mid_ch),
                                          eval(act_layer))
        
        # attns
        self.conv_branch_1 = nn.Sequential(nn.Conv2d(mid_ch, mid_ch,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     dilation=1,
                                                     groups=mid_ch,
                                                     bias=False),
                                          nn.BatchNorm2d(mid_ch),
                                          eval(act_layer))
        
        self.conv_branch_2 = nn.Sequential(nn.Conv2d(mid_ch, mid_ch,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=2,
                                                     dilation=2,
                                                     groups=mid_ch,
                                                     bias=False),
                                          nn.BatchNorm2d(mid_ch),
                                          eval(act_layer))
        
        self.conv_branch_3 = nn.Sequential(nn.Conv2d(mid_ch, mid_ch,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=3,
                                                     dilation=3,
                                                     groups=mid_ch,
                                                     bias=False),
                                          nn.BatchNorm2d(mid_ch),
                                          eval(act_layer))
        
        self.max_branch_1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1),
            nn.Conv2d(mid_ch, mid_ch,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      groups=1,
                      bias=False),
            nn.BatchNorm2d(mid_ch),
            eval(act_layer)
            )
        
        self.max_branch_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1, dilation=2),
            nn.Conv2d(mid_ch, mid_ch,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      groups=1,
                      bias=False),
            nn.BatchNorm2d(mid_ch),
            eval(act_layer)
            )
    

        # expansion conv
        self.pointwise_out = nn.Sequential(nn.Conv2d(mid_ch*5, out_ch,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0,
                                                    groups=1,
                                                    bias=False),
                                           nn.BatchNorm2d(out_ch),
                                           eval(act_layer),
                                           nn.Dropout2d(drop_rate))

    def forward(self, x):
        x = self.pointwise_in(x)

        x1 = self.conv_branch_1(x)
        x2 = self.conv_branch_2(x)
        x3 = self.conv_branch_3(x)
        x4 = self.max_branch_1(x)
        x5 = self.max_branch_2(x)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.pointwise_out(x)
        
        return x
# =================================== END OF MY OWN CNN BLOCKS ================================================= #


# ======================================== UPSAMPLING BLOCKS =================================================== #
## UPSAMPLING + DOUBLE CONV
class upsampling(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, depthwise=False, activation='nn.SiLU(inplace=True)', drop_rate=0.1):
        super(upsampling, self).__init__()
        
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, in_ch // 2, depthwise, activation, drop_rate)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch, depthwise=depthwise, activation=activation, drop_rate=drop_rate)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)

## UPSAMPLING + CONV WITH MULTIPLE DILATION RATES
class multidilation_upsampling(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, activation='nn.SiLU(inplace=True)', drop_rate=0.1):
        super(multidilation_upsampling, self).__init__()
        
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = conv_multidilation(in_ch, out_ch, reduction=4, act_layer=activation, drop_rate=drop_rate)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.conv = conv_multidilation(in_ch, out_ch, reduction=4, act_layer=activation, drop_rate=drop_rate)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)

# NOT WORKING (NEEDS REPAIR)
class Residual_upsampling(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, depthwise=False, activation=nn.ReLU, drop_rate=0.1):
        super(Residual_upsampling, self).__init__()
        
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = Residual_DoubleConv(in_ch, out_ch, in_ch // 2, depthwise, activation, drop_rate)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.conv = Residual_DoubleConv(in_ch, out_ch, depthwise=depthwise, activation=activation, drop_rate=drop_rate)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x) 
# ======================================== END OF UPSAMPLING BLOCKS =================================================== #