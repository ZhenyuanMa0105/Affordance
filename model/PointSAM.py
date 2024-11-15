import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from model.attention import MultiheadAttention, TransformerDecoder, TransformerDecoderLayer
from model.point_encoder import PointNet_Encoder
from model.view_weight_attn import CrossDistanceSampler
from model.uni3d import PointUni3d
from torchvision.ops import roi_align
from transformers import AutoModel, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false" # this disables a huggingface tokenizer warning (printed every epoch)


class PointSAM(nn.Module):
    def __init__(self, normal_channel=False, local_rank=None,
                N_p = 64, emb_dim = 512, proj_dim = 512, num_heads = 4, N_raw = 2048, num_affordance=18,
                freeze_text_encoder = False, text_encoder_type="roberta-base", n_groups=40, n_sample=20):
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)
        super().__init__()
        
        self.n_groups = n_groups
        self.emb_dim = emb_dim
        self.N_p = N_p
        self.N_raw = N_raw
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.local_rank = local_rank
        self.normal_channel = normal_channel
        self.num_affordance = num_affordance
        self.n_sample = n_sample
        if self.normal_channel:
            self.additional_channel = 3
        else:
            self.additional_channel = 0

        model_path = '/storage_fast/ycli/zhenyuan/LASO/model/robertabase'
        self.text_encoder = AutoModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.freeze_text_encoder = freeze_text_encoder
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)
        self.text_resizer = nn.Sequential(nn.Linear(self.text_encoder.config.hidden_size, emb_dim, bias=True),
                                          nn.LayerNorm(emb_dim, eps=1e-12))

        self.point_encoder = PointUni3d(self.n_groups, self.emb_dim, self.normal_channel, self.additional_channel, 
                                            self.N_p, self.text_encoder, self.tokenizer, self.text_resizer)
        # self.pos1d = nn.Embedding(self.n_groups, self.emb_dim)
        self.pos1d = nn.Parameter(torch.zeros(1, self.n_groups, self.emb_dim))
        nn.init.trunc_normal_(self.pos1d, std = 0.2) 
        self.pos2d = nn.Parameter(torch.zeros(1, 4, self.emb_dim))
        nn.init.trunc_normal_(self.pos2d, std = 0.2)
        self.pos3d = nn.Parameter(torch.zeros(1, self.n_sample + self.n_groups, self.emb_dim))
        nn.init.trunc_normal_(self.pos3d, std = 0.2)
        self.decoder = TransformerDecoder(TransformerDecoderLayer(self.emb_dim, nheads=num_heads, dropout=0),num_layers=1, norm=nn.LayerNorm(self.emb_dim))
        self.cross_attn = MultiheadAttention(self.emb_dim, self.num_heads)
        self.view_sampler = CrossDistanceSampler(self.n_sample, self.emb_dim, self.num_heads)
        
    def forward(self, text, xyz, view_mask):

        '''
        text: [B, L, 768]
        xyz: [B, 3, 2048]
        sub_box: bounding box of the interactive subject
        obj_box: bounding box of the interactive object
        '''

        B, C, N = xyz.size()

        point_feature, t_feat, t_mask = self.point_encoder(xyz, list(text), xyz.device)      

        query, query_mask = self.view_sampler(point_feature, view_mask, t_feat, t_mask, xyz)
        query = self.decoder(query, point_feature.transpose(-2, -1), tgt_key_padding_mask=query_mask, query_pos=self.pos3d)
        query *= query_mask.unsqueeze(-1).float()
        
        _3daffordance = torch.einsum('blc,bcn->bln', query, point_feature)
        _3daffordance = _3daffordance.sum(1)/(query_mask.float().sum(1).unsqueeze(-1))

        _3daffordance = torch.sigmoid(_3daffordance)
        return _3daffordance.squeeze(-1)

    def forward_text(self, text_queries, device):
        """
        text_queries : list of question str 
        out: text_embedding: bs, len, dim
            mask: bs, len (bool) [1,1,1,1,0,0]
        """
        # tokenized_queries = self.tokenizer.batch_encode_plus(text_queries, padding='longest', return_tensors='pt')
        tokenized_queries = self.tokenizer.batch_encode_plus(text_queries, padding='max_length', truncation=True,
                                                            max_length=self.n_groups,
                                                            return_tensors='pt')
        tokenized_queries = tokenized_queries.to(device)
        with torch.inference_mode(mode=self.freeze_text_encoder):
            encoded_text = self.text_encoder(**tokenized_queries).last_hidden_state
        # print(tokenized_queries.attention_mask.bool())

        return self.text_resizer(encoded_text), tokenized_queries.attention_mask.bool()

def get_PointSAM(normal_channel=False, local_rank=None,
    N_p = 64, emb_dim = 512, proj_dim = 512, num_heads = 4, N_raw = 2048, num_affordance=17, n_groups=40, n_sample=20):
    
    model = PointSAM( normal_channel, local_rank,
    N_p, emb_dim, proj_dim, num_heads, N_raw, num_affordance, n_groups=n_groups, n_sample=n_sample)
    return model


if __name__ == "__main__":
    import yaml
    file = open('/storage_fast/ycli/yiyang/LASO/config/default.yaml', 'r', encoding='utf-8')
    string = file.read()
    dict = yaml.safe_load(string)

    model = get_PointSAM(N_p=dict['N_p'], emb_dim=dict['emb_dim'],
                       proj_dim=dict['proj_dim'], num_heads=dict['num_heads'], N_raw=dict['N_raw'],
                       num_affordance = dict['num_affordance'], n_groups=8)

    text = ('what are three sitting on what are three sitting on', 'what are three')
    xyz = torch.rand(2, 3, 2048)
    mask = torch.rand(2, 4, 2048) > 0
    _3daffordance, logits = model(text, xyz, mask)
    print(_3daffordance.shape, logits.shape)

