import torch
import timm
import numpy as np
from torch import nn
from model.pointnet2_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation
from . import uni3d
# from . import losses

from model.pointnet2_utils import farthest_point_sample

import logging

class Point_Encoder(nn.Module):
    def __init__(self, emb_dim, normal_channel, additional_channel, N_p):
        super().__init__()
        self.N_p = N_p
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3+additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstractionMsg(self.N_p, [0.2,0.4], [16, 32], 256+256, [[128, 128, 256], [128, 196, 256]])

    def forward(self, xyz):

        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  #[B, 3, npoint_sa1] --- [B, 320, npoint_sa1]

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  #[B, 3, npoint_sa2] --- [B, 512, npoint_sa2]

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  #[B, 3, N_p]        --- [B, 512, N_p]

        return [[l0_xyz, l0_points], [l1_xyz, l1_points], [l2_xyz, l2_points], [l3_xyz, l3_points]]


class ModelConfig:
    def __init__(self, model_name, ckpt_path, pc_model, pretrained_pc, pc_feat_dim, embed_dim, group_size, num_group, pc_encoder_dim, patch_dropout, distributed=False):
        self.model_name = model_name
        self.ckpt_path = ckpt_path
        self.pc_model = pc_model
        self.pretrained_pc = pretrained_pc
        self.pc_feat_dim = pc_feat_dim
        self.embed_dim = embed_dim
        self.group_size = group_size
        self.num_group = num_group
        self.pc_encoder_dim = pc_encoder_dim
        self.patch_dropout = patch_dropout
        self.distributed = distributed


class PointUni3d(nn.Module):
    def __init__(self, emb_dim, normal_channel, additional_channel, N_p):
        super().__init__()
        
        self.emb_dim = emb_dim
        self.N_p = N_p
        self.normal_channel = normal_channel
        self.additional_channel = additional_channel

        self.point_encoder = Point_Encoder(self.emb_dim, self.normal_channel, self.additional_channel, self.N_p)


        self.fp1 = PointNetFeaturePropagation(in_channel=518+self.additional_channel, mlp=[512, 512]) 
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        config = ModelConfig(
            model_name='create_uni3d', 
            ckpt_path='/path/to/uni3d_model/model.pt', 
            pc_model='eva_giant_patch14_560.m30m_ft_in22k_in1k', 
            pretrained_pc='/path/to/eva_giant_patch14_560/model.safetensors', 
            pc_feat_dim=1408,
            embed_dim=1024,
            group_size=32,
            num_group=512,
            pc_encoder_dim=512,
            patch_dropout=0
        )
        
        logging.info("=> creating model: {}".format(config.model_name))
        self.model = getattr(uni3d, config.model_name)(args=config)
        self.model.to('cuda')

        checkpoint = torch.load(config.ckpt_path, map_location='cuda')
        logging.info('loaded checkpoint {}'.format(config.ckpt_path))

        sd = checkpoint['module']
        if not config.distributed and next(iter(sd.items()))[0].startswith('module'):
            sd = {k[len('module.'):]: v for k, v in sd.items()}
        self.model.load_state_dict(sd)
        self.trans2encoder = nn.Linear(config.embed_dim, 512)


    def forward(self, xyz):

        '''
        xyz: [B, 3, 2048]
        '''

        B, C, N = xyz.size()
        rgb = torch.full((B, 3, N), 0.4, device=xyz.device)  
        feature = torch.cat((xyz, rgb), dim=1)  

        with torch.no_grad():
            F_p_wise = get_model(self.model).encode_pc(feature.transpose(1, 2).contiguous())
        x, p_1, center = F_p_wise
        x = x / x.norm(dim=-1, keepdim=True)
 
        x = self.trans2encoder(x).to("cuda")
        up_sample = x.permute(0, 2, 1)  

        """ 
        Decoding
        """
        points1 = torch.cat([xyz, xyz], 1)  
        points2 = up_sample  
        up_sample = self.fp1(xyz, center.permute(0, 2, 1), points1, points2) # [B, C(512), N(2048)]

        return up_sample

class Uni3D(nn.Module):
    def __init__(self, point_encoder):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.point_encoder = point_encoder

    def encode_pc(self, pc):
        xyz = pc[:,:,:3].contiguous()
        color = pc[:,:,3:].contiguous()
        pc_feat = self.point_encoder(xyz, color)
        return pc_feat

    def forward(self, pc, text, image):
        text_embed_all = text
        image_embed = image   
        pc_embed = self.encode_pc(pc)
        return {'text_embed': text_embed_all,
                'pc_embed': pc_embed,
                'image_embed': image_embed,
                'logit_scale': self.logit_scale.exp()}
        
'''
def get_filter_loss(args):
    return losses.Uni3d_Text_Image_Loss()
'''

def get_model(model):
    if isinstance(model, torch.nn.DataParallel) \
      or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    else:
        return model

def get_metric_names(model):
    return ['loss', 'uni3d_loss', 'pc_image_acc', 'pc_text_acc']

def create_uni3d(args):  
    point_transformer = timm.create_model(args.pc_model, checkpoint_path=args.pretrained_pc, drop_path_rate=0.20)
    for param in point_transformer.parameters():
        param.requires_grad = False

    point_encoder = PointcloudEncoder(point_transformer, args)

    model = Uni3D(point_encoder=point_encoder,)
    return model
    
    
def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_data = farthest_point_sample(data, number)
    fps_data = data.gather(1, fps_data.unsqueeze(-1).expand(-1, -1, 3))
    # fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data

# https://github.com/Strawberry-Eat-Mango/PCT_Pytorch/blob/main/util.py 
def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist    


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token
        logging.info("patch dropout prob is {}".format(prob))

    def forward(self, x):
        # if not self.training or self.prob == 0.:
        #     return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz, color):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = fps(xyz, self.num_group) # B G 3
        # knn to get the neighborhood
        # _, idx = self.knn(xyz, center) # B G M
        idx = knn_point(self.group_size, xyz, center) # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()

        neighborhood_color = color.view(batch_size * num_points, -1)[idx, :]
        neighborhood_color = neighborhood_color.view(batch_size, self.num_group, self.group_size, 3).contiguous()

        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)

        features = torch.cat((neighborhood, neighborhood_color), dim=-1)
        return neighborhood, center, features

class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(6, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )
    def forward(self, point_groups):
        '''
            point_groups : B G N 3(6?)
            G is num of groups, N is num in each group
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 6)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1)) 
        feature_global = torch.max(feature,dim=2,keepdim=True)[0] 
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)
        feature = self.second_conv(feature) 
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] 
        return feature_global.reshape(bs, g, self.encoder_channel)

class PointcloudEncoder(nn.Module):
    def __init__(self, point_transformer, args):
        super().__init__()
        self.trans_dim = args.pc_feat_dim 
        self.embed_dim = args.embed_dim
        self.group_size = args.group_size 
        self.num_group = args.num_group
        # grouper
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        # define the encoder
        self.encoder_dim =  args.pc_encoder_dim
        self.encoder = Encoder(encoder_channel = self.encoder_dim)
       
        # bridge encoder and transformer
        self.encoder2trans = nn.Linear(self.encoder_dim,  self.trans_dim)
        
        # bridge transformer and clip embedding
        self.trans2embed = nn.Linear(self.trans_dim,  self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )  
        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(args.patch_dropout) if args.patch_dropout > 0. else nn.Identity()
        self.visual = point_transformer


    def forward(self, pts, colors):
        # divide the point cloud in the same form. This is important
        _, center, features = self.group_divider(pts, colors)
        
        # encode the input cloud patches
        group_input_tokens = self.encoder(features)  # B G N
        trans_group_input_tokens = self.encoder2trans(group_input_tokens)
        
        # prepare cls
        cls_tokens = self.cls_token.expand(trans_group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(trans_group_input_tokens.size(0), -1, -1)

        # add pos embedding
        pos = self.pos_embed(center)  # [B, 512, 3]-->[B, 512, trans_dim]

        # final input
        x = torch.cat((cls_tokens, trans_group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = x + pos
        # x = x.half()

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.patch_dropout(x)
        x = self.visual.pos_drop(x)

        # ModuleList not support forward
        for i, blk in enumerate(self.visual.blocks):
            x = blk(x)

        x = self.visual.norm(x[:, 1:, :])
        x = self.visual.fc_norm(x)
        x = self.trans2embed(x)

        return [x, group_input_tokens, center]
