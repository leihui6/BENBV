import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.ReLU(), nn.Linear(dim, dim), nn.LayerNorm(dim))

    def forward(self, x):
        return x + self.net(x)


class PointCloudNet(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        # Point Cloud Feature Extraction
        self.pos_feat = nn.Sequential(
            self._make_conv_block(3, 64),
            self._make_conv_block(64, 128),
            self._make_conv_block(128, 256),
            self._make_conv_block(256, 512),
            self._make_conv_block(512, 1024),
        )
        self.normal_feat = nn.Sequential(
            self._make_conv_block(3, 64),
            self._make_conv_block(64, 128),
            self._make_conv_block(128, 256),
            self._make_conv_block(256, 512),
        )

        # Boundary Point Features
        self.feat_pos = nn.Sequential(
            nn.Linear(3, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU()
        )
        self.feat_normal = nn.Sequential(
            nn.Linear(3, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU()
        )

        # Context Features
        self.feat_density = nn.Sequential(
            nn.Linear(1, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        self.feat_view = nn.Sequential(
            nn.Linear(1, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        # DV Features Fusion
        self.dv_fusion = nn.Sequential(nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU())  # Changed from 256 to 128

        self.global_fusion = nn.Sequential(nn.Linear(1536, 1536), nn.LayerNorm(1536), nn.ReLU())

        # Main Branch with ResBlocks
        self.main_branch = nn.Sequential(
            self._make_fc_block(1792, 512),  # 1536(global) + 256(boundary)
            ResBlock(512),
            self._make_fc_block(512, 256),
            ResBlock(256),
            self._make_fc_block(256, 128),
            ResBlock(128),
        )

        # Attention and Final Prediction
        self.pre_attention = nn.Sequential(  # 新增预处理层
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU()  # 128(main) + 128(dv) -> 256
        )
        self.self_attention = nn.MultiheadAttention(256, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(256)

        self.score_head = nn.Sequential(
            nn.Linear(256, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(64, 1), nn.Sigmoid()
        )

    def _make_conv_block(self, in_channels, out_channels):
        return nn.Sequential(nn.Conv1d(in_channels, out_channels, 1), nn.BatchNorm1d(out_channels), nn.ReLU())

    def _make_fc_block(self, in_features, out_features):
        return nn.Sequential(nn.Linear(in_features, out_features), nn.BatchNorm1d(out_features), nn.ReLU())

    def forward(self, P, S, C):
        B = P.size(0)

        # Point Cloud Features
        positions = P[:, :, :3].transpose(2, 1)  # [B, 3, N]
        normals = P[:, :, 3:].transpose(2, 1)  # [B, 3, N]

        pos_features = self.pos_feat(positions)  # [B, 1024, N]
        normal_features = self.normal_feat(normals)  # [B, 512, N]

        pos_global = pos_features.max(dim=2)[0]  # [B, 1024]
        normal_global = normal_features.max(dim=2)[0]  # [B, 512]

        # Context Features
        density = C[:, :20, 0:1]  # [B, 20, 1]
        view_order = C[:, 20:, 0:1]  # [B, 1, 1]

        density_feat = self.feat_density(density.reshape(-1, 1))  # [B*20, 128]
        view_feat = self.feat_view(view_order.repeat(1, 20, 1).reshape(-1, 1))  # [B*20, 128]

        density_feat = density_feat.view(B, 20, -1)  # [B, 20, 128]
        view_feat = view_feat.view(B, 20, -1)  # [B, 20, 128]

        # Global Features
        global_feat = torch.cat([pos_global, normal_global], dim=1)  # [B, 1536]
        global_feat = self.global_fusion(global_feat)  # [B, 1536]
        global_feat = global_feat.unsqueeze(1).expand(-1, 20, -1)  # [B, 20, 1536]

        # Boundary Features
        pos, normal = S[:, :, :3], S[:, :, 3:]  # [B, 20, 3]
        pos = pos.reshape(-1, 3)  # [B*20, 3]
        normal = normal.reshape(-1, 3)  # [B*20, 3]
        pos_feat = self.feat_pos(pos)  # [B*20, 128]
        normal_feat = self.feat_normal(normal)  # [B*20, 128]
        S_feat = torch.cat([pos_feat, normal_feat], dim=1)  # [B*20, 256]
        S_feat = S_feat.view(B, 20, -1)  # [B, 20, 256]

        # Main Branch Processing
        combined = torch.cat([global_feat, S_feat], dim=2)  # [B, 20, 1792]
        combined = combined.view(-1, combined.size(2))  # [B*20, 1792]
        features = self.main_branch(combined)  # [B*20, 128]
        features = features.view(B, 20, -1)  # [B, 20, 128]

        # DV Features Fusion
        dv_combined = torch.cat([density_feat, view_feat], dim=2)  # [B, 20, 256]
        dv_feat = self.dv_fusion(dv_combined)  # [B, 20, 128]

        # Attention and Prediction
        pre_attention = torch.cat([features, dv_feat], dim=2)  # [B, 20, 256]
        pre_attention = self.pre_attention(pre_attention)  # [B, 20, 256]
        attended_features, _ = self.self_attention(pre_attention, pre_attention, pre_attention)
        features = self.norm1(attended_features)  # [B, 20, 256]

        features = features.reshape(-1, 256)  # [B*20, 256]
        score = self.score_head(features)  # [B*20, 1]
        score = score.view(B, 20, 1)  # [B, 20, 1]

        return score
