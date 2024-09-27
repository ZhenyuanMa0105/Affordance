import torch
import torch.nn as nn
import torch.nn.functional as F

class QueryGenerationModule(nn.Module):
    def __init__(self, emb_dim=512, num_heads=8, dropout=0.1):
        """
        Initialize the Query Generation Module using `fs` (superpoint features) and `t_feat` (language features).
        Args:
        - emb_dim: Embedding dimension size (512 by default).
        - Nq: Number of queries to generate (default: 16).
        - num_heads: Number of attention heads (default: 8).
        - dropout: Dropout rate for multi-head attention (default: 0.1).
        """
        super(QueryGenerationModule, self).__init__()

        # Projection layers for fs (superpoint features) and t_feat (language features)
        self.linear = nn.Linear(emb_dim, emb_dim) 

        # Multi-head attention mechanism
        self.multihead_attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, 
                                                    dropout=dropout, batch_first=True)

    def forward(self, fs, t_feat, t_mask=None):
        """
        Forward pass for generating queries from `fs` and `t_feat`.
        
        Args:
        - fs: Superpoint features with shape (B, V, emb_dim), where V is the number of views.
        - t_feat: Language features with shape (B, L, emb_dim), where L is the length of the text sequence.
        - t_mask: Optional mask for attention (shape: B, L) for `t_feat`.

        Returns:
        - queries: Generated queries of shape (B, V, emb_dim).
        - attn_weights: Attention weights from the multi-head attention mechanism.
        """
        B, V, emb_dim = fs.shape  # Extract batch size, number of views, and embedding dimensions

        fs = self.linear(fs)  # (B, V, emb_dim)
        t_feat = self.linear(t_feat)  # (B, T, emb_dim)
        # Apply multi-head attention with fs as the query and t_feat as key/value
        attn_output, attn_weights = self.multihead_attn(fs, t_feat, t_feat, key_padding_mask=t_mask)

        # Generate final query vectors
        queries = self.linear(attn_output)  # (B, L, emb_dim)

        return queries, attn_weights

