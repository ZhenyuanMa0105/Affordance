import torch
import torch.nn as nn

class GeoAwarePooling(nn.Module):
    """Pool point features to super points.
    """
    def __init__(self, channel_proj):
        super().__init__()
        self.pts_proj1 = nn.Sequential(
            nn.Linear(3, channel_proj),
            nn.LayerNorm(channel_proj),
            nn.ReLU(),
            nn.Linear(channel_proj, channel_proj),
            nn.LayerNorm(channel_proj)
        )
        self.pts_proj2 = nn.Sequential(
            nn.Linear(2 * channel_proj, channel_proj),
            nn.LayerNorm(channel_proj),
            nn.ReLU(),
            nn.Linear(channel_proj, 1, bias=False),
            nn.Sigmoid()
        )

    def norm_positions(self, xyz, point_masks):
        """
        Normalize the positions of the points for each view.

        Args:
            xyz: [B, N, 3] - Coordinates of the points in the batch.
            point_masks: [B, 4, N] - Binary masks for the points (4 views).

        Returns:
            normalized_xyz: [B, 4, N, 3] - Normalized positions of the points.
        """
        
        xyz_t = xyz.transpose(1, 2)  # [B, N, 3]
        masked_xyz = xyz_t.unsqueeze(1) * point_masks.unsqueeze(-1)  # [B, 4, N, 3]

        # Compute the center of the masked points in each view (mean of valid points)
        valid_points_per_view = point_masks.sum(dim=-1, keepdim=True)  # [B, 4, 1]
        valid_points_per_view = torch.clamp(valid_points_per_view, min=1)

        # Compute the center of valid points for each view
        center = masked_xyz.sum(dim=-2) / valid_points_per_view  # [B, 4, 3]

        # Compute relative positions for valid points (points - center)
        relative_positions = xyz_t.unsqueeze(1) - center.unsqueeze(-2)  # [B, 4, N, 3]

        # Compute the diameter (max distance between any two points in each view)
        max_segment = masked_xyz.max(dim=-2).values  # [B, 4, 3]
        min_segment = masked_xyz.min(dim=-2).values  # [B, 4, 3]
        diameter = (max_segment - min_segment).max(dim=-1).values  # [B, 4]
        diameter = torch.where(diameter == 0, torch.ones_like(diameter), diameter)
        # Normalize the relative positions by the diameter
        normalized_xyz = relative_positions / diameter.unsqueeze(-1).unsqueeze(-1)  # [B, 4, N, 3]

        return normalized_xyz

    def forward(self, xyz, point_features, point_masks, shape_id=None):
        """
        xyz: [B, 3, N] - Coordinates of the points in the batch.
        point_features: [B, emb_dim, N] - Features of the points.
        point_masks: [B, 4, N] - Binary masks for the points (4 views).
        
        Returns:
        pooled_feature: [B, 4, emb_dim] - The pooled query features for each view.
        """
        
        # xyz_t = xyz.transpose(1, 2)  # [B, N, 3]
        point_features_t = point_features.transpose(1, 2)  # [B, N, emb_dim]
        # masked_xyz = xyz_t.unsqueeze(1) * point_masks.unsqueeze(-1)  # [B, 4, N, 3]
        masked_features = point_features_t.unsqueeze(1) * point_masks.unsqueeze(-1)  # [B, 4, N, emb_dim]
        B, V, N, emb_dim = masked_features.size()

        norm_xyz = self.norm_positions(xyz, point_masks)
        
        local_xyz = self.pts_proj1(norm_xyz)  # [32, 4, 2048, 512]
        # Compute global feature per view by max pooling valid points, ignoring masked points
        global_xyz, _ = local_xyz.max(dim=-2)  # [B, V, channel_proj]
        global_xyz_N = global_xyz.unsqueeze(-2).repeat(1, 1, N, 1)  # [B, V, N, channel_proj]
        cat_xyz = torch.cat([local_xyz, global_xyz_N], dim=-1)  # [B, V, N, 2 * channel_proj]
        weights = self.pts_proj2(cat_xyz).squeeze(-1) * 2  # [B, V, N]
        pooled_feature = (masked_features * weights.unsqueeze(-1)).sum(dim=-2)  # [B, V, emb_dim]
        
        pooled_feature = pooled_feature / torch.clamp((point_masks * weights).sum(dim=-1, keepdim=True), min=1e-8) + global_xyz  # [B, V, emb_dim]
        return pooled_feature
