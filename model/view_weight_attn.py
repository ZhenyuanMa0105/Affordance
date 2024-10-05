import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention import MultiheadAttentionWithWeight, MultiheadAttention
class ViewTranformer(nn.Module):
    def __init__(self, emb_dim=512, num_heads=4):
        super().__init__()
        self.emb_dim = emb_dim
        self.pts_proj1 = nn.Sequential(
            nn.Linear(3, self.emb_dim),
            nn.LayerNorm(self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.LayerNorm(self.emb_dim)
        )
        self.cross_attn = MultiheadAttentionWithWeight(self.emb_dim, num_heads)
    def forward(self, xyz, point_features, point_masks):
        """
        xyz: [B, 3, N] - Coordinates of the points in the batch.
        point_features: [B, emb_dim, N] - Features of the points.
        point_masks: [B, 4, N] - Binary masks for the points (4 views).
        
        Returns:
        pooled_feature: [B, 4, emb_dim] - The pooled query features for each view.
        """
        B, C, N = point_features.shape

        point_features_t = point_features.transpose(1, 2)  # [B, N, emb_dim]
        xyz_t = xyz.transpose(1, 2)  # [B, N, 3]
        masked_xyz = xyz_t.unsqueeze(1) * point_masks.unsqueeze(-1)  # [B, 4, N, 3]

        valid_points_per_view = point_masks.sum(dim=-1, keepdim=True)  # [B, 4, 1]
        voting_ratio = valid_points_per_view.squeeze(-1) / N  # [B, 4]
        valid_points_per_view = torch.clamp(valid_points_per_view, min=1)
        center_xyz = masked_xyz.sum(dim=-2) / valid_points_per_view  # [B, 4, 3]
        center_xyz = self.pts_proj1(center_xyz)  # [B, 4, emb_dim]

        # Compute attn_weight
        attn_weight = torch.einsum('bi, bij->bj', voting_ratio, point_masks)  # [B, N]
        point_mask_sum = point_masks.sum(dim=1)  # [B, N]
        valid_mask = point_mask_sum > 0  # [B, N], True where point is valid

        # Use a large negative value instead of -inf
        mask_value = -1e9
        attn_weight = attn_weight.masked_fill(~valid_mask, mask_value)  # [B, N]

        # Handle batches with all invalid points
        batch_valid_points = valid_mask.sum(dim=-1) > 0  # [B]
        if not batch_valid_points.all():
            # For batches with no valid points, set attn_weight to zeros
            attn_weight[~batch_valid_points, :] = 0

        # Apply log_softmax
        attn_weight = torch.log_softmax(attn_weight, dim=-1)  # [B, N]

        # Apply MultiheadAttention
        output = self.cross_attn(
            query=center_xyz,  # [B, 4, emb_dim]
            key=point_features_t,  # [B, N, emb_dim]
            value=point_features_t,  # [B, N, emb_dim]
            # key_padding_mask=~valid_mask,  # [B, N]
            attn_weight=attn_weight  # [B, N]
        )  # Output shape: [B, 4, emb_dim]

        return output


class ViewGlobalSampler(nn.Module):
    def __init__(self, n_sample=20, emb_dim=512, num_heads=4):
        super().__init__()
        self.n_sample = n_sample
        self.emb_dim = emb_dim
        self.self_attn = MultiheadAttention(self.emb_dim, num_heads)
    def forward(self, point_features, point_masks, t_feat, t_mask):
        """
        point_features: [B, emb_dim, N] - Features of the points.
        point_masks: [B, 4, N] - Binary masks for the points (4 views).
        
        Returns:
        pooled_feature: [B, 4, emb_dim] - The pooled query features for each view.
        """
        B, C, N = point_features.shape

        point_features = point_features.transpose(1, 2)  # [B, N, emb_dim]
        valid_points_per_view = point_masks.sum(dim=-1, keepdim=True)  # [B, 4, 1]
        voting_ratio = valid_points_per_view.squeeze(-1) / N  # [B, 4]
        # Compute vote_weight
        vote_weight = torch.einsum('bi, bij->bj', voting_ratio, point_masks)  # [B, N]
        point_mask_sum = point_masks.sum(dim=1)  # [B, N]
        valid_mask = point_mask_sum > 0  # [B, N], True where point is valid

        # Use a large negative value instead of -inf
        mask_value = -1e9
        vote_weight = vote_weight.masked_fill(~valid_mask, mask_value)  # [B, N]

        # Handle batches with all invalid points
        batch_valid_points = valid_mask.sum(dim=-1) > 0  # [B]
        if not batch_valid_points.all():
            # For batches with no valid points, set vote_weight to zeros
            vote_weight[~batch_valid_points, :] = 0

        vote_weight = torch.softmax(vote_weight, dim=-1)  # [B, N]
        
        if self.training:
            sampled_indices = torch.multinomial(vote_weight, self.n_sample)  # [B, n_sample]
        else:
            sampled_indices = torch.topk(vote_weight, self.n_sample, dim=-1).indices  # [B, n_sample]
        sampled_features = torch.gather(point_features, 1, sampled_indices.unsqueeze(-1).expand(-1, -1, C))  # [B, n_sample, emb_dim]

        # Concatenate sampled features with text features
        combined_features = torch.cat([sampled_features, t_feat], dim=1)  # [B, n_sample + T, emb_dim]
        combined_mask = torch.cat([torch.ones(B, self.n_sample, device=point_features.device, dtype=torch.bool), t_mask], dim=1)  # [B, n_sample + T]
        
        output = self.self_attn(
            query=combined_features,  # [B, n_sample + T, emb_dim]
            key=combined_features,  # [B, n_sample + T, emb_dim]
            value=combined_features,  # [B, n_sample + T, emb_dim]
            key_padding_mask=combined_mask,  # [B, n_sample + T]
        )
        return output, combined_mask

class ViewDistanceSampler(nn.Module):
    def __init__(self, n_sample=20, emb_dim=512, num_heads=4):
        super().__init__()
        self.n_sample = n_sample
        self.emb_dim = emb_dim
        self.self_attn = MultiheadAttention(self.emb_dim, num_heads)
    def forward(self, point_features, point_masks, t_feat, t_mask, xyz):
        """
        xyz: [B, 3, N] - Coordinates of the points in the batch.
        point_features: [B, emb_dim, N] - Features of the points.
        point_masks: [B, 4, N] - Binary masks for the points (4 views).
        
        Returns:
        pooled_feature: [B, 4, emb_dim] - The pooled query features for each view.
        """
        B, C, N = point_features.shape
        _, V, _ = point_masks.shape

        point_features = point_features.transpose(1, 2)  # [B, N, emb_dim]
        xyz_t = xyz.transpose(1, 2)  # [B, N, 3]
        masked_xyz = xyz_t.unsqueeze(1) * point_masks.unsqueeze(-1)  # [B, 4, N, 3]

        valid_points_per_view = point_masks.sum(dim=-1, keepdim=True)  # [B, 4, 1]
        valid_points_per_view = torch.clamp(valid_points_per_view, min=1)
        center_xyz = masked_xyz.sum(dim=-2) / valid_points_per_view  # [B, 4, 3]
        
        distances = torch.cdist(xyz_t, center_xyz)  # [B, N, 4]

        # Select the closest 5 points for each center point
        _, closest_indices = torch.topk(-distances, torch.round(self.n_sample / V), dim=1)  # [B, 5, 4]
        closest_indices = closest_indices.transpose(1, 2).reshape(B, -1)  # [B, 20]

        # Gather the closest point features
        sampled_features = torch.gather(point_features, 1, closest_indices.unsqueeze(-1).expand(-1, -1, C))  # [B, 20, emb_dim]

        # Concatenate sampled features with text features
        combined_features = torch.cat([sampled_features, t_feat], dim=1)  # [B, n_sample + T, emb_dim]
        combined_mask = torch.cat([torch.ones(B, self.n_sample, device=point_features.device, dtype=torch.bool), t_mask], dim=1)  # [B, n_sample + T]
        
        output = self.self_attn(
            query=combined_features,  # [B, n_sample + T, emb_dim]
            key=combined_features,  # [B, n_sample + T, emb_dim]
            value=combined_features,  # [B, n_sample + T, emb_dim]
            key_padding_mask=combined_mask,  # [B, n_sample + T]
        )
        return output, combined_mask