import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention import MultiheadAttentionWithWeight, MultiheadAttention
from timm.models.layers import DropPath

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
    def __init__(self, n_sample=20, emb_dim=512, num_heads=8):
        super().__init__()
        self.n_sample = n_sample
        self.emb_dim = emb_dim
        self.self_attn = MultiheadAttention(self.emb_dim, num_heads)
    def forward(self, point_features, point_masks, t_feat, t_mask, xyz=None):
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

class ViewLocalSampler(nn.Module):
    def __init__(self, n_sample=20, emb_dim=512, num_heads=4):
        super().__init__()
        self.n_sample = n_sample
        self.emb_dim = emb_dim
        self.self_attn = MultiheadAttention(self.emb_dim, num_heads)
    def forward(self, point_features, point_masks, t_feat, t_mask, xyz=None):
        """
        point_features: [B, emb_dim, N] - Features of the points.
        point_masks: [B, 4, N] - Binary masks for the points (4 views).
        
        Returns:
        pooled_feature: [B, 4, emb_dim] - The pooled query features for each view.
        """
        B, C, N = point_features.shape
        V = point_masks.shape[1]  # Number of views
        n_sample_per_view = self.n_sample // V
        
        point_features = point_features.transpose(1, 2)  # [B, N, emb_dim]
        valid_points_per_view = point_masks.sum(dim=-1, keepdim=True)  # [B, 4, 1]
        voting_ratio = valid_points_per_view.squeeze(-1) / N  # [B, 4]
        vote_weight = torch.einsum('bi, bij->bj', voting_ratio, point_masks)  # [B, N]
        vote_weight = vote_weight.unsqueeze(1) * point_masks  # [B, 4, N]

        # Mask invalid points with a large negative value before applying softmax
        vote_weight = vote_weight.masked_fill(point_masks == 0, -1e9)  # [B, 4, N]

        # Handle edge cases where all points are invalid for a view
        all_invalid_mask = (point_masks.sum(dim=-1, keepdim=True) == 0)  # [B, 4, 1]
        vote_weight = vote_weight.masked_fill(all_invalid_mask, 0)  # Set vote_weight to 0 where all points are invalid

        # Apply softmax to create probabilities for each view
        point_weight = torch.softmax(vote_weight, dim=-1)  # [B, 4, N]

        if self.training:
            sampled_indices = torch.multinomial(point_weight.view(B * V, N), n_sample_per_view).view(B, V, n_sample_per_view)  # [B, 4, n_sample_per_view]
        else:
            sampled_indices = torch.topk(point_weight, n_sample_per_view, dim=-1).indices  # [B, 4, n_sample_per_view]

        # Gather the sampled point features for each view
        sampled_features = torch.gather(point_features.unsqueeze(1).expand(-1, V, -1, -1), 2, sampled_indices.unsqueeze(-1).expand(-1, -1, -1, C))  # [B, 4, n_sample_per_view, emb_dim]
        sampled_features = sampled_features.view(B, -1, C)  # [B, 4 * n_sample_per_view, emb_dim]

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
        distances = distances.transpose(1, 2)  # [B, 4, N]
        n_sample_per_view = self.n_sample // V
        if self.training:
            probabilities = torch.softmax(-distances, dim=-1)
            sampled_indices = torch.multinomial(probabilities.view(B * V, N), n_sample_per_view, replacement=False).view(B, V, n_sample_per_view)  # [B, 4, n_sample_per_view]
        else:
            # Select the closest 5 points for each center point
            _, sampled_indices = torch.topk(-distances, n_sample_per_view, dim=-1)  # [B, 4, 5]
        sampled_indices = sampled_indices.reshape(B, -1)  # [B, 20]

        # Gather the closest point features
        sampled_features = torch.gather(point_features, 1, sampled_indices.unsqueeze(-1).expand(-1, -1, C))  # [B, 20, emb_dim]

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

class FeatureSampler(nn.Module):
    def __init__(self, n_sample=20, emb_dim=512, num_heads=4):
        super().__init__()
        self.n_sample = n_sample
        self.emb_dim = emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 1, bias=False),
            nn.Sigmoid()
        )
        self.self_attn = MultiheadAttention(self.emb_dim, num_heads)
    def forward(self, point_features, point_masks, t_feat, t_mask, xyz=None):
        """
        point_features: [B, emb_dim, N] - Features of the points.
        point_masks: [B, 4, N] - Binary masks for the points (4 views).
        
        Returns:
        pooled_feature: [B, 4, emb_dim] - The pooled query features for each view.
        """
        B, C, N = point_features.shape
        _, V, _ = point_masks.shape
        point_features = point_features.transpose(1, 2)  # [B, N, emb_dim]
        masked_features = point_features.unsqueeze(1) * point_masks.unsqueeze(-1)
        valid_mask = point_masks.unsqueeze(-1).expand(-1, -1, -1, C).bool()  # [B, 4, N, emb_dim]
        mask_value = -1e9
        masked_features = masked_features.masked_fill(~valid_mask, mask_value)  # [B, V, N, emb_dim]
        point_weight = self.mlp(masked_features).squeeze(-1)  # [B, V, N]
        n_sample_per_view = self.n_sample // V
        if self.training:
            # [B*4, n_sample_per_view], treat (B*V) as batch for sampling
            sampled_indices = torch.multinomial(point_weight.view(B * V, N), n_sample_per_view).view(B, V, n_sample_per_view)
        else:
            # [B, 4, n_sample_per_view], get the top-k indices
            sampled_indices = torch.topk(point_weight, n_sample_per_view, dim=-1).indices  # [B, 4, n_sample_per_view]
        sampled_indices_expanded = sampled_indices.unsqueeze(-1).expand(-1, -1, -1, C)  # [B, 4, n_sample_per_view, emb_dim]
        point_features_expanded = point_features.unsqueeze(1).expand(-1, V, -1, -1)  # [B, 4, N, emb_dim]
        sampled_features = torch.gather(point_features_expanded, 2, sampled_indices_expanded)  # [B, 4, n_sample_per_view, emb_dim]
        sampled_features = sampled_features.view(B, V * n_sample_per_view, C)  # [B, n_sample, emb_dim]

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

class DualDistanceSampler(nn.Module):
    def __init__(self, n_sample=20, emb_dim=512, num_heads=4, drop_path_prob=0.1):
        super().__init__()
        self.n_sample = n_sample
        self.emb_dim = emb_dim
        self.cross_attn1 = MultiheadAttention(self.emb_dim, num_heads)
        self.cross_attn2 = MultiheadAttention(self.emb_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim)
        )
        self.drop_path = DropPath(drop_path_prob)
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
        distances = distances.transpose(1, 2)  # [B, 4, N]
        n_sample_per_view = self.n_sample // V
        if self.training:
            probabilities = torch.softmax(-distances, dim=-1)
            sampled_indices = torch.multinomial(probabilities.view(B * V, N), n_sample_per_view, replacement=False).view(B, V, n_sample_per_view)  # [B, 4, n_sample_per_view]
        else:
            # Select the closest 5 points for each center point
            _, sampled_indices = torch.topk(-distances, n_sample_per_view, dim=-1)  # [B, 4, 5]
        sampled_indices = sampled_indices.reshape(B, -1)  # [B, 20]

        # Gather the closest point features
        sampled_features = torch.gather(point_features, 1, sampled_indices.unsqueeze(-1).expand(-1, -1, C))  # [B, 20, emb_dim]
        
        cross_features1 = self.cross_attn1(
            query=sampled_features,  # [B, 20, emb_dim]
            key=t_feat,  # [B, 20, emb_dim]
            value=t_feat,  # [B, 20, emb_dim]
            key_padding_mask=t_mask,  # [B, 20]
        )
        
        cross_features2 = self.cross_attn2(
            query=t_feat,  # [B, 20, emb_dim]
            key=sampled_features,  # [B, 20, emb_dim]
            value=sampled_features,  # [B, 20, emb_dim]
        )
        
        # combined_features = torch.cat([cross_features1, cross_features2], dim=1)
        sampled_features = sampled_features + self.drop_path(self.mlp(cross_features1))
        t_feat = t_feat + self.drop_path(self.mlp(cross_features2))
        combined_features = torch.cat([sampled_features, t_feat], dim=1)  # [B, n_sample + T, emb_dim]
        combined_mask = torch.cat([torch.ones(B, self.n_sample, device=point_features.device, dtype=torch.bool), t_mask], dim=1)  # [B, n_sample + T]
        # combined_features = self.self_attn(
        #     query=combined_features,  # [B, n_sample + T, emb_dim]
        #     key=combined_features,  # [B, n_sample + T, emb_dim]
        #     value=combined_features,  # [B, n_sample + T, emb_dim]
        #     key_padding_mask=combined_mask,  # [B, n_sample + T]
        # )
        return combined_features, combined_mask