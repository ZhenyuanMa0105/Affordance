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
            xyz: [B, 3, N] - Coordinates of the points in the batch.
            point_masks: [B, 4, N] - Binary masks for the points (4 views).

        Returns:
            normalized_xyz: [B, 4, N, 3] - Normalized positions of the points.
        """
        # Transpose xyz to shape [B, N, 3] for easier operations
        xyz_t = xyz.transpose(1, 2)  # [B, N, 3]
        
        # Mask the xyz coordinates based on the point_masks
        masked_xyz = xyz_t.unsqueeze(1) * point_masks.unsqueeze(-1)  # [B, 4, N, 3]

        # Compute the center of the masked points in each view (mean of valid points)
        valid_points_per_view = point_masks.sum(dim=-1, keepdim=True)  # [B, 4, 1]
        valid_points_per_view = valid_points_per_view + (valid_points_per_view == 0).float()  # Avoid div-by-zero

        # Compute the center of valid points for each view
        center = (masked_xyz.sum(dim=-2) / valid_points_per_view)  # [B, 4, 3]

        # Compute relative positions for valid points (points - center)
        relative_positions = masked_xyz - center.unsqueeze(-2)  # [B, 4, N, 3]

        # Compute the diameter (max distance between any two points in each view)
        max_segment = masked_xyz.max(dim=-2).values  # [B, 4, 3]
        min_segment = masked_xyz.min(dim=-2).values  # [B, 4, 3]
        diameter = (max_segment - min_segment).max(dim=-1).values  # [B, 4]

        # Normalize the relative positions by the diameter
        normalized_xyz = relative_positions / (diameter.unsqueeze(-1).unsqueeze(-1) + 1e-8)  # [B, 4, N, 3]

        return normalized_xyz

    def forward(self, xyz, point_features, point_masks, shape_id=None):
        """
        xyz: [B, 3, N] - Coordinates of the points in the batch.
        point_features: [B, emb_dim, N] - Features of the points.
        point_masks: [B, 4, N] - Binary masks for the points (4 views).
        
        Returns:
        pooled_feature: [B, 4, emb_dim] - The pooled query features for each view.
        """
        
        xyz_t = xyz.transpose(1, 2)  # [B, N, 3]
        point_features_t = point_features.transpose(1, 2)  # [B, N, emb_dim]
        xyz_expand = xyz_t.unsqueeze(1).repeat(1, 4, 1, 1)  # [B, 4, N, 3]
        masked_xyz = xyz_expand * point_masks.unsqueeze(-1)  # [B, 4, N, 3]
        masked_features = point_features_t.unsqueeze(1) * point_masks.unsqueeze(-1)  # [B, 4, N, emb_dim]
        B, V, N, emb_dim = masked_features.size()

        # relative_positions = self.norm_positions(xyz, point_masks)
        # assert not torch.isnan(relative_positions).any(), "NaN values found in relative positions!"
        # assert not torch.isinf(relative_positions).any(), "Inf values found in relative positions!"
        # Apply pts_proj1 to the relative positions without extra masking

        local_xyz = self.pts_proj1(xyz_expand)  # [32, 4, 2048, 512]
        # Compute global feature per view by max pooling valid points, ignoring masked points
        global_xyz, _ = local_xyz.max(dim=-2)  # [B, V, channel_proj]
        global_xyz_N = global_xyz.unsqueeze(-2).repeat(1, 1, N, 1)  # [B, V, N, channel_proj]
        cat_xyz = torch.cat([local_xyz, global_xyz_N], dim=-1)  # [B, V, N, 2 * channel_proj]
        weights = self.pts_proj2(cat_xyz).squeeze(-1) * 2  # [B, V, N]
        pooled_feature = (masked_features * weights.unsqueeze(-1)).sum(dim=-2)  # [B, V, emb_dim]
        
        pooled_feature = pooled_feature / ((point_masks * weights).sum(dim=-1, keepdim=True) + 1e-8) + global_xyz  # [B, V, emb_dim]
        return pooled_feature


# class GeoAwarePoolingBV(nn.Module):
#     """Pool point features to super points.
#     """
#     def __init__(self, channel_proj):
#         super().__init__()
#         self.pts_proj1 = nn.Sequential(
#             nn.Linear(3, channel_proj),
#             nn.LayerNorm(channel_proj),
#             nn.ReLU(),
#             nn.Linear(channel_proj, channel_proj),
#             nn.LayerNorm(channel_proj)
#         )
#         self.pts_proj2 = nn.Sequential(
#             nn.Linear(2 * channel_proj, channel_proj),
#             nn.LayerNorm(channel_proj),
#             nn.ReLU(),
#             nn.Linear(channel_proj, 1, bias=False),
#             nn.Sigmoid()
#         )
#     def scatter_norm(self, points, idx):
#         ''' Normalize positions of same-segment in a unit sphere of diameter 1
#         '''
#         min_segment = scatter(points, idx, dim=0, reduce='min')
#         max_segment = scatter(points, idx, dim=0, reduce='max')
#         diameter_segment = (max_segment - min_segment).max(dim=1).values
#         center_segment = scatter(points, idx, dim=0, reduce='mean')
#         print(f"points shape: {points.shape}, idx shape: {idx.shape}, center_segment shape: {center_segment.shape}")
#         print(f"idx max: {idx.max()}, idx min: {idx.min()}")
#         center = center_segment[idx]
#         diameter = diameter_segment[idx]
#         diameter = (max_segment - min_segment).max(dim=1).values
#         points = (points - center) / (diameter.view(-1, 1) + 1e-2)
#         return points, diameter_segment.view(-1, 1)

#     def geo_aware_pooling(self, x, sp_idx, all_xyz, with_xyz=False):
#         all_xyz_ = torch.cat(all_xyz)
#         all_xyz, _ = self.scatter_norm(all_xyz_, sp_idx)
#         all_xyz = self.pts_proj1(all_xyz)
#         all_xyz_segment = scatter(all_xyz, sp_idx, dim=0, reduce='max')
#         all_xyz = torch.cat([all_xyz, all_xyz_segment[sp_idx]], dim=-1)
#         all_xyz_w = self.pts_proj2(all_xyz) * 2
#         if with_xyz:
#             x = torch.cat([x * all_xyz_w, all_xyz_], dim=-1)
#             x = scatter_mean(x, sp_idx, dim=0)
#             x[:, :-3] = x[:, :-3] + all_xyz_segment
#         else:
#             x = scatter_mean(x * all_xyz_w, sp_idx, dim=0) + all_xyz_segment
#         return x, all_xyz_w

#     def forward(self, xyz, point_features, point_masks):
#         """
#         xyz: [B, 3, N] - Coordinates of the points in the batch.
#         point_features: [B, emb_dim, N] - Features of the points.
#         point_masks: [B, 4, N] - Binary masks for the points (4 views).
        
#         Returns:
#         pooled_feature: [B, 4, emb_dim] - The pooled query features for each view.
#         """
        
#         B, V, N = point_masks.size()
#         xyz_t = xyz.transpose(1, 2)  # [B, N, 3]
#         point_features_t = point_features.transpose(1, 2)  # [B, N, emb_dim]
#         point_masks = point_masks.bool()
#         query = torch.zeros(B, V, point_features.size(1), device=xyz.device)  # Initialize query feature tensor

#         for b in range(B):
#             sp_idx = []
#             all_xyz = []
#             all_features = []
#             for v in range(V):
#                 xyz_b = xyz_t[b]  # [N, 3]
#                 features_b = point_features_t[b]  # [N, emb_dim]
#                 mask = point_masks[b, v, :]  # [N]
#                 xyz_b = xyz_b[mask]
#                 features_b = features_b[mask]
#                 all_xyz.append(xyz_b)
#                 all_features.append(features_b)
#                 sp_idx.append(v * torch.ones(xyz_b.shape[0], dtype=torch.long, device=xyz.device))
#                 # Use geometric-aware pooling to get the query feature
#             all_features = torch.cat(all_features)
#             sp_idx = torch.cat(sp_idx)
#             pooled_feature = self.geo_aware_pooling(all_features, sp_idx, all_xyz)
#             query[b] = pooled_feature
#         return query