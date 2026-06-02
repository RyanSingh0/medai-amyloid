"""
Model architecture for Amyloid PET Centiloid Prediction.

Baseline 3D CNN with tracer conditioning.
"""

import os
import torch
import torch.nn as nn


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1      = conv3x3x3(inplanes, planes, stride)
        self.bn1        = nn.BatchNorm3d(planes)
        self.relu       = nn.ReLU(inplace=True)
        self.conv2      = conv3x3x3(planes, planes)
        self.bn2        = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class MedicalResNet34(nn.Module):
    """
    ResNet-34 matching MedicalNet architecture exactly.
    Layers: [3, 4, 6, 3] with BasicBlock.
    Output: 512-dim vector after global average pool.
    """
    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.conv1    = nn.Conv3d(1, 64, kernel_size=7,
                                  stride=(2,2,2), padding=(3,3,3),
                                  bias=False)
        self.bn1      = nn.BatchNorm3d(64)
        self.relu     = nn.ReLU(inplace=True)
        self.maxpool  = nn.MaxPool3d(kernel_size=(3,3,3),
                                     stride=2, padding=1)
        self.layer1   = self._make_layer(BasicBlock, 64,  3)
        self.layer2   = self._make_layer(BasicBlock, 128, 4, stride=2)
        self.layer3   = self._make_layer(BasicBlock, 256, 6, stride=2)
        self.layer4   = self._make_layer(BasicBlock, 512, 3, stride=2)
        self.avgpool  = nn.AdaptiveAvgPool3d(1)
        self._init_weights()

    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.avgpool(x).flatten(1)   # (B, 512)


class FiLMLayer(nn.Module):
    """
    Tracer-specific scale+shift: out = gamma(tracer)*x + beta(tracer)
    Initialized to identity → starts exactly like no conditioning.
    """
    def __init__(self, num_tracers, num_channels):
        super().__init__()
        self.gamma = nn.Embedding(num_tracers, num_channels)
        self.beta  = nn.Embedding(num_tracers, num_channels)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, x, tracer_ids):
        g = self.gamma(tracer_ids).view(-1, x.shape[1], 1, 1, 1)
        b = self.beta(tracer_ids).view(-1, x.shape[1], 1, 1, 1)
        return g * x + b


class BaselineCNN(nn.Module):
    """
    MedicalNet ResNet-34 + FiLM tracer conditioning.

    FiLM after each ResNet stage:
      layer1 → 64ch,  layer2 → 128ch,
      layer3 → 256ch, layer4 → 512ch

    Two LR groups via get_param_groups():
      backbone → lr/10  (fine-tune gently)
      FiLM + head → lr  (train from scratch)

    Input:  (B, 1, 128, 128, 128)
    Output: (B,) centiloid scores
    """

    def __init__(self, num_tracers, emb_dim=16,
                 mean_centiloid=0.0, pretrained_path=None):
        super().__init__()

        self.backbone = MedicalResNet34()

        # FiLM after each stage
        self.film1 = FiLMLayer(num_tracers, 64)
        self.film2 = FiLMLayer(num_tracers, 128)
        self.film3 = FiLMLayer(num_tracers, 256)
        self.film4 = FiLMLayer(num_tracers, 512)

        self.tracer_emb = nn.Embedding(num_tracers, emb_dim)

        # 512 (backbone) + 16 (tracer) = 528
        self.head = nn.Sequential(
            nn.Linear(512 + emb_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
        nn.init.constant_(self.head[-1].bias, mean_centiloid)

        if pretrained_path:
            self._load_pretrained(pretrained_path)

    def _load_pretrained(self, path):
        if not os.path.exists(path):
            print(f"Weights not found at {path} — training from scratch")
            return

        print(f"Loading MedicalNet ResNet-34 from {path}...")
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        sd   = ckpt.get('state_dict', ckpt)

        # Strip DataParallel 'module.' prefix
        clean = {k.replace('module.', ''): v for k, v in sd.items()}

        missing, unexpected = self.backbone.load_state_dict(
            clean, strict=False)
        loaded = len(clean) - len(missing)
        print(f"  Loaded {loaded}/{len(clean)} layers.")
        if missing:
            print(f"  Missing (OK if <5): {missing[:3]}")
        print("  Done.")

    def get_param_groups(self, base_lr):
        """Backbone at lr/10, FiLM+head at full lr."""
        backbone = list(self.backbone.parameters())
        new = (list(self.film1.parameters()) +
               list(self.film2.parameters()) +
               list(self.film3.parameters()) +
               list(self.film4.parameters()) +
               list(self.tracer_emb.parameters()) +
               list(self.head.parameters()))
        return [
            {'params': backbone, 'lr': base_lr / 10},
            {'params': new,      'lr': base_lr},
        ]

    def forward(self, x, tracer_idx):
        # Pass through backbone layer by layer with FiLM
        x = self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x)))
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.film1(x, tracer_idx)

        x = self.backbone.layer2(x)
        x = self.film2(x, tracer_idx)

        x = self.backbone.layer3(x)
        x = self.film3(x, tracer_idx)

        x = self.backbone.layer4(x)
        x = self.film4(x, tracer_idx)

        features = self.backbone.avgpool(x).flatten(1)  # (B, 512)
        tracer_f = self.tracer_emb(tracer_idx)           # (B, 16)
        combined = torch.cat([features, tracer_f], 1)
        return self.head(combined).squeeze(1)  