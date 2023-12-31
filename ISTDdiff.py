import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange
import math
import warnings
import os

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class Spectral_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads=1,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b,h*w,c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp))
        v = v
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b,h,w,c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out, attn

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)

class FeedForward1(nn.Module):
    def __init__(self, dim):
        super(FeedForward1, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(dim),
            GELU()
        )
        self.conv = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        residual = x
        x = self.body(x)
        x = F.relu(x+residual, True)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        return x

class SST_block(nn.Module):
    def __init__(
            self,
            dim,
            heads=1,
            num_blocks=1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                Spectral_MSA(dim=dim, dim_head=dim, heads=heads),
                FeedForward1(dim=dim)
            ]))
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        for (Spectral_attn, ff) in self.blocks:
            x_ = self.norm2(x)
            x_, spec_attn = Spectral_attn(x_)
            x = x+x_
            x_ = self.norm3(x)
            x = ff(x_) + x
        out = x.permute(0, 3, 1, 2)
        return out, spec_attn

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
 
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Time_mapping_block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.time_mlp0 = nn.Linear(8, dim)
        self.time_mlp1 = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.SiLU(),
            nn.Linear(dim*4, dim)
        )
        self.time_mlp2 = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.SiLU(),
            nn.Linear(dim*4, dim)
        )

    def forward(self, t):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        t = self.time_mlp0(t)
        t1 = self.time_mlp1(t)
        t2 = self.time_mlp2(t)
        t1 = torch.reshape(t1, [t1.shape[0],t1.shape[1],1,1])
        t2 = torch.reshape(t2, [t2.shape[0],t2.shape[1],1,1])
        return t1, t2

class NA_Spectral_MSA(nn.Module):
    def __init__(
            self,
            dim,
            heads=1,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim
        self.to_v = nn.Linear(dim, dim * heads, bias=False)
        self.proj = nn.Linear(dim * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in, attn):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b,h*w,c)
        v_inp = self.to_v(x)
        v,v_ = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (v_inp, v_inp))
        v = v.transpose(-2, -1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b,h,w,c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out

class NA_Spatial_MSA(nn.Module):
    def __init__(
            self,
            dim,
            heads=1,
            window_size=(8, 8),
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.scale = dim ** -0.5
        self.window_size = window_size

        # position embedding
        seq_l = window_size[0] * window_size[1]
        self.pos_emb = nn.Parameter(torch.Tensor(1, heads, seq_l, seq_l))
        trunc_normal_(self.pos_emb)

        inner_dim = dim * heads
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, attn):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x.shape
        w_size = self.window_size
        assert h % w_size[0] == 0 and w % w_size[1] == 0, 'fmap dimensions must be divisible by the window size'
        x_inp = rearrange(x, 'b (h b0) (w b1) c -> (b h w) (b0 b1) c', b0=w_size[0], b1=w_size[1])
        v = self.to_v(x_inp)
        v,v_ = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (v,v))
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = rearrange(out, '(b h w) (b0 b1) c -> b (h b0) (w b1) c', h=h // w_size[0], w=w // w_size[1], b0=w_size[0])
        return out

class NACSST_block(nn.Module):
    def __init__(self, dim, heads=1, num_blocks=1):
        super().__init__()
        self.time_mapping = Time_mapping_block(dim)
        self.norm = nn.LayerNorm(dim)
        
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                NA_Spectral_MSA(dim=dim),
                FeedForward(dim=dim)
            ]))
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x_t, t, attn, x):
        t1, t2 = self.time_mapping(t)
        x_t = x_t*t1+t2
        x_t = x_t.permute(0, 2, 3, 1)
        for (Spectral_attn, ff) in self.blocks:
            x_t_ = self.norm2(x_t)
            x_t = Spectral_attn(x_t_, attn) + x_t
            x_t_ = self.norm3(x_t)
            x_t = ff(x_t_) + x_t
        out = x_t.permute(0, 3, 1, 2)
        return out

class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        self.conv = nn.Conv2d(in_channels, in_channels, 1, 1, 0)

    def forward(self, x):
        residual = x
        x = self.body(x)
        out = F.relu(x+residual, True)
        out = self.conv(out)
        return out

class Upconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upconv, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, in_channels, 4, 2, 1)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        x = self.deconv(x)
        x = self.conv(x)
        return x

class Downconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downconv, self).__init__()
        self.down_conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.down_conv(x)
        return x

class Fusionconv(nn.Module):
    def __init__(self, in_channels):
        super(Fusionconv, self).__init__()
        self.conv = nn.Conv2d(in_channels*2, in_channels, 1, 1, 0)

    def forward(self, x, y):
        x = torch.concat([x,y], 1)
        x = self.conv(x)
        return x

class Head(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Head, self).__init__()
        inter_channels = in_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, out_channels, 1, 1, 0)
        )

    def forward(self, x):
        return self.block(x)

class ISTDdiff(nn.Module):
    def __init__(self, channels):
        super(ISTDdiff, self).__init__()
        self.I_stem = nn.Conv2d(3, channels[0], 3, 1, 1)
        self.N_stem = nn.Conv2d(1, channels[0], 3, 1, 1)
        self.time_encode = SinusoidalPositionEmbeddings(8)

        self.SST1 = SST_block(channels[0])
        self.SST2 = SST_block(channels[1])
        self.SST3 = SST_block(channels[2])
        self.SST4 = SST_block(channels[2])
        self.SST5 = SST_block(channels[1])
        self.SST6 = SST_block(channels[0])

        self.NASST1 = NACSST_block(channels[0])
        self.NASST2 = NACSST_block(channels[1])
        self.NASST3 = NACSST_block(channels[2])
        self.NASST4 = NACSST_block(channels[2])
        self.NASST5 = NACSST_block(channels[1])
        self.NASST6 = NACSST_block(channels[0])
        
        self.I_Down1 = Downconv(channels[0], channels[1])
        self.I_Down2 = Downconv(channels[1], channels[2])
        self.I_Up3 = Upconv(channels[2], channels[1])
        self.I_Up4 = Upconv(channels[1], channels[0])
        
        self.N_Down1 = Downconv(channels[0], channels[1])
        self.N_Down2 = Downconv(channels[1], channels[2])
        self.N_Up3 = Upconv(channels[2], channels[1])
        self.N_Up4 = Upconv(channels[1], channels[0])

        self.conv_fuse0 = Fusionconv(channels[0])
        self.conv_fuse1 = Fusionconv(channels[1])
        self.conv_fuse2 = Fusionconv(channels[1])
        self.conv_fuse3 = Fusionconv(channels[0])
        self.conv_fuse4 = Fusionconv(channels[0])

        self.last_res = ResBlock(channels[0])
        self.head = Head(channels[0], 1)

    def forward(self, noise, t, infrared):
        t = self.time_encode(t)
        infrared = self.I_stem(infrared)
        noise = self.N_stem(noise)
        noise = self.conv_fuse0(noise, infrared)
        infrared_1, attn = self.SST1(infrared)
        noise_1 = self.NASST1(noise, t, attn, infrared_1)
        infrared_2, attn = self.SST2((self.I_Down1(infrared_1)))
        noise_2 = self.NASST2(self.N_Down1(noise_1), t, attn, infrared_2)
        infrared_3, attn = self.SST3(self.I_Down2(infrared_2))
        noise_3 = self.NASST3(self.N_Down2(noise_2), t, attn, infrared_3)

        infrared_4, attn = self.SST4(infrared_3)
        noise_4 = self.NASST4(noise_3, t, attn, infrared_4)
        infrared_5, attn = self.SST5(self.conv_fuse1(self.I_Up3(infrared_4), infrared_2))
        noise_5 = self.NASST5((self.conv_fuse2(self.N_Up3(noise_4), noise_2)), t, attn, infrared_5)
        infrared_6, attn = self.SST6(self.conv_fuse3(self.I_Up4(infrared_5),infrared_1))
        noise_6 = self.NASST6((self.conv_fuse4(self.N_Up4(noise_5), noise_1)), t, attn, infrared_6)
        out = infrared_6 + noise_6
        out = self.last_res(out)
        out = self.head(out)
        return out

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    net = ISTDdiff(channels = [8, 16, 32])
    img = torch.rand(1, 3, 480, 480).cuda()
    t = torch.tensor([10] * img.shape[0],dtype=torch.int32).cuda()
    noise = img[:,0:1,:,:]
    print(net)
    net = net.cuda()
    img_out = net(noise, t, img)
    print(img_out.shape)
    print('Parameters number is ', sum(param.numel() for param in net.parameters()))