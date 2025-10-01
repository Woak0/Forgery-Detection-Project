"""
Enhanced U-Net model with attention mechanisms for image forgery detection.
Fixed issues from code review.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel attention mechanism"""
    def __init__(self, in_planes, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention mechanism"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x_cat))


class CBAM(nn.Module):
    """Convolutional Block Attention Module - Fixed sequential application"""
    def __init__(self, in_planes, ratio=8, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
    
    def forward(self, x):
        # Apply channel attention first, then spatial attention to the result
        x_ca = self.ca(x) * x
        x_sa = self.sa(x_ca) * x_ca
        return x_sa


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        size = x.shape[2:]
        
        feat1 = F.relu(self.bn1(self.conv1(x)))
        feat2 = F.relu(self.bn2(self.conv2(x)))
        feat3 = F.relu(self.bn3(self.conv3(x)))
        feat4 = F.relu(self.bn4(self.conv4(x)))
        feat5 = F.relu(self.pool(x))
        feat5 = F.interpolate(feat5, size=size, mode='bilinear', align_corners=False)
        
        out = torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)
        return self.project(out)


class AttentionGate(nn.Module):
    """Attention gate for skip connections"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class EnhancedDoubleConv(nn.Module):
    """Enhanced double convolution with fixed residual connection"""
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout=0.2):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)
        self.attention = CBAM(out_channels)
        
        # Skip connection
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None
    
    def forward(self, x):
        residual = self.skip(x) if self.skip else x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        # Apply attention before adding residual
        out = self.attention(out)
        
        # Add residual before final activation
        if residual.shape == out.shape:
            out = out + residual
        
        return self.relu(out)


class EnhancedUNet(nn.Module):
    """U-Net with attention mechanisms - Fixed bottleneck and channel dimensions"""
    def __init__(self, n_channels=4, n_classes=1, features=[64, 128, 256, 512], dropout=0.2):
        super().__init__()
        
        # Initial processing
        self.initial = nn.Sequential(
            nn.Conv2d(n_channels, features[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
            ChannelAttention(features[0])
        )
        
        # Encoder
        self.encoder1 = EnhancedDoubleConv(features[0], features[0], dropout=dropout)
        self.pool1 = nn.MaxPool2d(2)
        
        self.encoder2 = EnhancedDoubleConv(features[0], features[1], dropout=dropout)
        self.pool2 = nn.MaxPool2d(2)
        
        self.encoder3 = EnhancedDoubleConv(features[1], features[2], dropout=dropout)
        self.pool3 = nn.MaxPool2d(2)
        
        self.encoder4 = EnhancedDoubleConv(features[2], features[3], dropout=dropout)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck - Fixed to not double channels unnecessarily
        self.bottleneck = nn.Sequential(
            EnhancedDoubleConv(features[3], features[3], dropout=dropout),
            ASPP(features[3], features[3])  # Apply ASPP directly without doubling
        )
        
        # Decoder with fixed channel dimensions
        self.upconv4 = nn.ConvTranspose2d(features[3], features[3], 2, stride=2)
        self.att4 = AttentionGate(features[3], features[3], features[3]//2)
        self.decoder4 = EnhancedDoubleConv(features[3]*2, features[3], dropout=dropout)
        
        self.upconv3 = nn.ConvTranspose2d(features[3], features[2], 2, stride=2)
        self.att3 = AttentionGate(features[2], features[2], features[2]//2)
        self.decoder3 = EnhancedDoubleConv(features[2]*2, features[2], dropout=dropout)
        
        self.upconv2 = nn.ConvTranspose2d(features[2], features[1], 2, stride=2)
        self.att2 = AttentionGate(features[1], features[1], features[1]//2)
        self.decoder2 = EnhancedDoubleConv(features[1]*2, features[1], dropout=dropout)
        
        self.upconv1 = nn.ConvTranspose2d(features[1], features[0], 2, stride=2)
        self.att1 = AttentionGate(features[0], features[0], features[0]//2)
        self.decoder1 = EnhancedDoubleConv(features[0]*2, features[0], dropout=dropout)
        
        # Output
        self.out = nn.Conv2d(features[0], n_classes, 1)
    
    def forward(self, x):
        # Initial processing
        x = self.initial(x)
        
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder with attention
        dec4 = self.upconv4(bottleneck)
        enc4 = self.att4(dec4, enc4)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        enc3 = self.att3(dec3, enc3)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        enc2 = self.att2(dec2, enc2)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        enc1 = self.att1(dec1, enc1)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)
        
        return self.out(dec1)