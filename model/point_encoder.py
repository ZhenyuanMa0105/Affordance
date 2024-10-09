import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from model.pointnet2_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation
# from pointnet2_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation
class PointNet_Encoder(nn.Module):
    def __init__(self, emb_dim, normal_channel, additional_channel, N_p):
        super().__init__()
        self.N_p = N_p
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3+additional_channel, 
                                             [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4, 0.8], [64, 128], 128+128+64, 
                                             [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstractionMsg(self.N_p, [0.2, 0.4], [16, 32], 256+256, 
                                             [[128, 128, 256], [128, 196, 256]])

        # Feature propagation layers
        self.fp3 = PointNetFeaturePropagation(in_channel=512+emb_dim, mlp=[768, 512])  
        self.fp2 = PointNetFeaturePropagation(in_channel=832, mlp=[768, 512]) 
        self.fp1 = PointNetFeaturePropagation(in_channel=518+additional_channel, mlp=[512, 512]) 

    def forward(self, xyz):
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz

        # First abstraction level
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  # [B, 3, npoint_sa1] --- [B, 320, npoint_sa1]

        # Second abstraction level
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # [B, 3, npoint_sa2] --- [B, 512, npoint_sa2]

        # Third abstraction level
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # [B, 3, N_p]        --- [B, 512, N_p]

        # Upsample features
        up_sample = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)   # [B, emb_dim, npoint_sa2]
        up_sample = self.fp2(l1_xyz, l2_xyz, l1_points, up_sample)    # [B, emb_dim, npoint_sa1]   
        point_features = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), up_sample)  # [B, emb_dim, N]

        return point_features