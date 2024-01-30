import torch.nn as nn
from timm.models.layers import DropPath

class Part_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.linear_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_1, x_2, x_3):
        B, N, C = x_1.shape # N = 1, C = 128
        # B,1,8,128/8
        q = self.linear_q(x_1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1) #B*8*128/8*1
        k = self.linear_k(x_2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1) #B*8*128/8*1
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        del q, k
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        v = self.linear_v(x_3).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1) #B*8*128/8*1
        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # B*8*16*1 -> B*16*8*1 -> B*1*128
        # x = (attn @ v).permute(0, 3, 1, 2)  # B*8*16*1 -> B*1*8*16
        # x = x.reshape(B, N, C)
        del v, attn
        x = self.proj(x)
        x = self.proj_drop(x)
        return x # B*1*128

class FCAM2(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm3_11 = norm_layer(dim)
        self.norm3_12 = norm_layer(dim)
        self.norm3_13 = norm_layer(dim)

        self.attn_1 = Part_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, \
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x_1, x_2):
        x_1 = x_1 + self.drop_path(self.attn_1(self.norm3_11(x_2), self.norm3_12(x_1), self.norm3_13(x_1))) 

        return  x_1
