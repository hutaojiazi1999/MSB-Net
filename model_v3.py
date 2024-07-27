import torch
from torch import nn
from torch.nn import functional as F
import numbers
from einops import rearrange

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=False, norm=False, relu=True, transpose=False,
                 channel_shuffle_g=0, norm_method=nn.BatchNorm2d, groups=1):
        super(BasicConv, self).__init__()
        self.channel_shuffle_g = channel_shuffle_g
        self.norm = norm
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, groups=groups))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, groups=groups))
        if norm:
            layers.append(norm_method(out_channel))
        elif relu:
            layers.append(nn.ReLU(inplace=True))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, BasicConv=BasicConv):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = BasicConv(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, bias=bias,
                                relu=False, groups=hidden_features * 2)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, BasicConv=BasicConv):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = BasicConv(dim * 3, dim * 3, kernel_size=3, stride=1, bias=bias, relu=False, groups=dim * 3)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q.shape

        mask1 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)


        attn = (q @ k.transpose(-2, -1)) * self.temperature


        index = torch.topk(attn, k=int(C * 7 / 10), dim=-1, largest=True)[1]

        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))


        attn1 = attn1.softmax(dim=-1)


        out = (attn1 @ v)


        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, BasicConv=BasicConv):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias, BasicConv=BasicConv)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias, BasicConv=BasicConv)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)
        

class Fusion(nn.Module):
    def __init__(self, in_dim=32):
        super(Fusion, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
        self.key_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)

        self.gamma1 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
        self.gamma2 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
        self.sig = nn.Sigmoid()

    def forward(self, x, y):
        x_q = self.query_conv(x)
        y_k = self.key_conv(y)
        energy = x_q * y_k
        attention = self.sig(energy)
        attention_x = x * attention
        attention_y = y * attention

        x_gamma = self.gamma1(torch.cat((x, attention_x), dim=1))
        x_out = x * x_gamma[:, [0], :, :] + attention_x * x_gamma[:, [1], :, :]

        y_gamma = self.gamma2(torch.cat((y, attention_y), dim=1))
        y_out = y * y_gamma[:, [0], :, :] + attention_y * y_gamma[:, [1], :, :]

        x_s = x_out + y_out

        return x_s

##################################
# def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
#     return nn.Conv2d(
#         in_channels, out_channels, kernel_size,
#         padding=(kernel_size//2), bias=bias, stride = stride)
# class CALayer(nn.Module):
#     def __init__(self, channel, reduction=16, bias=False):
#         super(CALayer, self).__init__()
#         # global average pooling: feature --> point
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         # feature channel downscale and upscale --> channel weight
#         self.conv_du = nn.Sequential(
#                 nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
#                 nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         y = self.avg_pool(x)
#         y = self.conv_du(y)
#         return x * y
# class CAB(nn.Module):
#     def __init__(self, n_feat, kernel_size, reduction, bias, act):
#         super(CAB, self).__init__()
#         modules_body = []
#         modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
#         modules_body.append(act)
#         modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
#
#         self.CA = CALayer(n_feat, reduction, bias=bias)
#         self.body = nn.Sequential(*modules_body)
#
#     def forward(self, x):
#         res = self.body(x)
#         res = self.CA(res)
#         res += x
#         return res
# class CNNBlock(nn.Module):
#     def __init__(self, dim, kernel_size=3, reduction=4, act=nn.PReLU(), bias=False):
#         super(CNNBlock, self).__init__()
#
#         self.decoder_level1 = [CAB(dim,                     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
#
#         self.decoder_level1 = nn.Sequential(*self.decoder_level1)
#
#     def forward(self, outs):
#
#         dec3 = self.decoder_level1(outs)
#
#
#         return dec3
##################################


class MultiscaleNet(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=30,
                 num_blocks=[2, 3, 3],
                 heads=[1, 2, 4],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):
        super(MultiscaleNet, self).__init__()
        # self.patch_embed_small = OverlapPatchEmbed(inp_channels, dim)
        #
        # self.encoder_level1_small = nn.Sequential(*[
        #     TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
        #                      LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        #
        # self.down1_2_small = Downsample(dim)
        # #self.encoder_level2_small = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1)
        #
        # #self.down2_3_small = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1)
        # self.latent_small = nn.Sequential(*[
        #     TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
        #                      bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        #
        # #self.up3_2_small = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1)
        # #self.reduce_chan_level2_small = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        # #self.decoder_level2_small = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1)
        #
        # self.up2_1_small = Upsample(int(dim * 2 ** 1))
        # self.decoder_level1_small = nn.Sequential(*[
        #     TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
        #                      bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        #
        # self.output_small = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        #
        # self.patch_embed_mid = OverlapPatchEmbed(inp_channels, dim)
        #
        # self.encoder_level1_mid1 = nn.Sequential(*[
        #     TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
        #                      LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        #
        # self.encoder_level1_mid2 = nn.Sequential(*[
        #     TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
        #                      LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        #
        # self.down1_2_mid = Downsample(dim)
        # self.down1_2_mid2 = Downsample(dim)
        # #self.encoder_level2_mid1 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1)
        # #self.encoder_level2_mid2 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1)
        #
        # #self.down2_3_mid = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1)
        # #self.down2_3_mid2 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1)
        # self.latent_mid1 = nn.Sequential(*[
        #     TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
        #                      bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        # self.latent_mid2 = nn.Sequential(*[
        #     TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
        #                      bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        #
        # #self.up3_2_mid = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1)
        # #self.up3_2_mid2 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1)
        # #self.reduce_chan_level2_mid1 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        # #self.reduce_chan_level2_mid2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        # #self.decoder_level2_mid1 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1)
        # #self.decoder_level2_mid2 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1)
        #
        # self.up2_1_mid = Upsample(int(dim * 2 ** 1))
        # self.up2_1_mid2 = Upsample(int(dim * 2 ** 1))
        # self.decoder_level1_mid1 = nn.Sequential(*[
        #     TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
        #                      bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        # self.decoder_level1_mid2 = nn.Sequential(*[
        #     TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
        #                      bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        #
        # self.output_mid = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        # self.output_mid_context = nn.Conv2d(int(dim * 2 ** 1), dim, kernel_size=3, stride=1, padding=1, bias=bias)

        self.patch_embed_max = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1_max1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.encoder_level1_max2 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.encoder_level1_max3 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2_max = Downsample(dim)
        self.down1_2_max2 = Downsample(dim)
        self.down1_2_max3 = Downsample(dim)
        #self.encoder_level2_max1 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1)
        #self.encoder_level2_max2 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1)
        #self.encoder_level2_max3 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1)

        #self.down2_3_max = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1)
        #self.down2_3_max2 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1)
        #self.down2_3_max3 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1)
        self.latent_max1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.latent_max2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.latent_max3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        #self.up3_2_max = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1)
        #self.up3_2_max2 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1)
        #self.up3_2_max3 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1)
        #self.reduce_chan_level2_max1 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        #self.reduce_chan_level2_max2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        #self.reduce_chan_level2_max3 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        #self.decoder_level2_max1 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1)
        #self.decoder_level2_max2 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1)
        #self.decoder_level2_max3 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1)

        self.up2_1_max = Upsample(int(dim * 2 ** 1))
        self.up2_1_max2 = Upsample(int(dim * 2 ** 1))
        self.up2_1_max3 = Upsample(int(dim * 2 ** 1))
        self.decoder_level1_max1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.decoder_level1_max2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.decoder_level1_max3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.output_max = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output_max_context1 = nn.Conv2d(int(dim * 2 ** 1), dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output_max_context2 = nn.Conv2d(int(dim * 2 ** 1), dim, kernel_size=3, stride=1, padding=1, bias=bias)

        # self.BF1 = Fusion(dim*2)
        # self.BF2 = Fusion(dim*2)
        # self.BF3 = Fusion(dim*2)


        # ################################################################################################################
        # #CNN
        # #small
        # self.patch_embed_small__ = OverlapPatchEmbed(inp_channels, dim)
        #
        # self.encoder_level1_small__ = nn.Sequential(*[
        #     CNNBlock(dim=dim) for i in range(num_blocks[0])])
        #
        # self.down1_2_small__ = Downsample(dim)
        # #self.encoder_level2_small__ = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1)
        #
        # #self.down2_3_small__ = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1)
        # self.latent_small__ = nn.Sequential(*[
        #     CNNBlock(dim=int(dim * 2 ** 1)) for i in range(num_blocks[2])])
        #
        # #self.up3_2_small__ = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1)
        # #self.reduce_chan_level2_small__ = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        # #self.decoder_level2_small__ = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1)
        #
        # self.up2_1_small__ = Upsample(int(dim * 2 ** 1))
        # self.reduce_chan_level1_small__ = nn.Conv2d(int(dim * 2 ** 1), int(dim * 1 ** 1), kernel_size=1, bias=bias)
        # self.decoder_level1_small__ = nn.Sequential(*[
        #     CNNBlock(dim=int(dim * 1 ** 1)) for i in range(num_blocks[0])])
        #
        # self.output_small__ = nn.Conv2d(int(dim * 1 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        # #mid
        # self.patch_embed_mid__ = OverlapPatchEmbed(inp_channels, dim)
        #
        # self.encoder_level1_mid__ = nn.Sequential(*[
        #     CNNBlock(dim=dim) for i in range(num_blocks[0])])
        #
        # self.down1_2_mid__ = Downsample(dim)
        # #self.encoder_level2_mid__ = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1)
        # #self.down2_3_mid__ = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1)
        # self.latent_mid__ = nn.Sequential(*[
        #     CNNBlock(dim=int(dim * 2 ** 1)) for i in range(num_blocks[2])])
        #
        # #self.up3_2_mid__ = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1)
        # #self.reduce_chan_level2_mid__ = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        # #self.decoder_level2_mid__ = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1)
        #
        # self.up2_1_mid__ = Upsample(int(dim * 2 ** 1))
        # self.reduce_chan_level1_mid__ = nn.Conv2d(int(dim * 2 ** 1), int(dim * 1 ** 1), kernel_size=1, bias=bias)
        # self.decoder_level1_mid__ = nn.Sequential(*[
        #     CNNBlock(dim=int(dim * 1 ** 1)) for i in range(num_blocks[0])])
        #
        # self.output_mid__ = nn.Conv2d(int(dim * 1 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)


    def forward(self, inp_img):
        outputs = list()

        inp_img_max = inp_img
        # inp_img_mid = F.interpolate(inp_img, scale_factor=0.5)
        # inp_img_small = F.interpolate(inp_img, scale_factor=0.25)
        #####################################################################################
        # #small
        # inp_enc_level1_small__ = self.patch_embed_small__(inp_img_small)
        # out_enc_level1_small__ = self.encoder_level1_small__(inp_enc_level1_small__)
        # inp_enc_level2_small__ = self.down1_2_small__(out_enc_level1_small__)
        #
        # latent_small__ = self.latent_small__(inp_enc_level2_small__)
        #
        #
        # inp_dec_level1_small__ = self.up2_1_small__(latent_small__)
        # inp_dec_level1_small__ = torch.cat([inp_dec_level1_small__, out_enc_level1_small__], 1)
        # inp_dec_level1_small__ = self.reduce_chan_level1_small__(inp_dec_level1_small__)
        # out_dec_level1_small__ = self.decoder_level1_small__(inp_dec_level1_small__)
        # out_dec_level1_small__ = self.output_small__(out_dec_level1_small__) + inp_img_small
        # #mid
        # inp_enc_level1_mid__ = self.patch_embed_mid__(inp_img_mid)
        # out_enc_level1_mid__ = self.encoder_level1_mid__(inp_enc_level1_mid__)
        # inp_enc_level2_mid__ = self.down1_2_mid__(out_enc_level1_mid__)
        #
        # latent_mid__ = self.latent_mid__(inp_enc_level2_mid__)
        #
        #
        # inp_dec_level1_mid__ = self.up2_1_mid__(latent_mid__)
        # inp_dec_level1_mid__ = torch.cat([inp_dec_level1_mid__, out_enc_level1_mid__], 1)
        # inp_dec_level1_mid__ = self.reduce_chan_level1_mid__(inp_dec_level1_mid__)
        # out_dec_level1_mid__ = self.decoder_level1_mid__(inp_dec_level1_mid__)
        # out_dec_level1_mid__ = self.output_small__(out_dec_level1_mid__) + inp_img_mid
        #####################################################################################

        # inp_enc_level1_small = self.patch_embed_small(inp_img_small)
        # out_enc_level1_small = self.encoder_level1_small(inp_enc_level1_small)
        #
        # inp_enc_level2_small = self.down1_2_small(out_enc_level1_small)
        #
        # latent_small = self.latent_small(inp_enc_level2_small)
        #
        # inp_img_small_ = F.interpolate(out_dec_level1_small__, scale_factor=2)
        #
        # mid_img = inp_img_mid + inp_img_small_
        #
        # inp_enc_level1_mid = self.patch_embed_mid(mid_img)
        # out_enc_level1_mid = self.encoder_level1_mid1(inp_enc_level1_mid)
        #
        # inp_enc_level2_mid = self.down1_2_mid(out_enc_level1_mid)
        #
        # latent_mid = self.latent_mid1(inp_enc_level2_mid)
        #
        # mid_img_ = F.interpolate(out_dec_level1_mid__, scale_factor=2)
        
        max_img = inp_img_max

        inp_enc_level1_max = self.patch_embed_max(max_img)
        out_enc_level1_max = self.encoder_level1_max1(inp_enc_level1_max)

        inp_enc_level2_max = self.down1_2_max(out_enc_level1_max)

        latent_max = self.latent_max1(inp_enc_level2_max)
        # BFF_max_1 = latent_max


        inp_dec_level1_max = self.up2_1_max(latent_max)
        inp_dec_level1_max = torch.cat([inp_dec_level1_max, out_enc_level1_max], 1)
        out_dec_level1_max = self.decoder_level1_max1(inp_dec_level1_max)

        out_dec_level1_max = self.output_max_context1(out_dec_level1_max)
        out_enc_level1_max = self.encoder_level1_max2(out_dec_level1_max)

        inp_enc_level2_max = self.down1_2_max2(out_enc_level1_max)

        latent_max = self.latent_max2(inp_enc_level2_max)
        # BFF_max_2 = latent_max

        inp_dec_level1_max = self.up2_1_max2(latent_max)
        inp_dec_level1_max = torch.cat([inp_dec_level1_max, out_enc_level1_max], 1)
        out_dec_level1_max = self.decoder_level1_max2(inp_dec_level1_max)

        out_dec_level1_max = self.output_max_context2(out_dec_level1_max)
        out_enc_level1_max = self.encoder_level1_max3(out_dec_level1_max)

        inp_enc_level2_max = self.down1_2_max3(out_enc_level1_max)

        latent_max = self.latent_max3(inp_enc_level2_max)
        # BFF_max_3 = latent_max
        #
        # BFF1 = self.BF1(BFF_max_1, BFF_max_2)
        # BFF2 = self.BF2(BFF_max_2, BFF_max_3)
        #
        # BFF1 = F.interpolate(BFF1, scale_factor=0.5)
        # BFF2 = F.interpolate(BFF2, scale_factor=0.5)
        #
        #
        # BFF3_1 = latent_mid
        # latent_mid = latent_mid + BFF1



        # inp_dec_level1_mid = self.up2_1_mid(latent_mid)
        # inp_dec_level1_mid = torch.cat([inp_dec_level1_mid, out_enc_level1_mid], 1)
        # out_dec_level1_mid = self.decoder_level1_mid1(inp_dec_level1_mid)
        #
        # out_dec_level1_mid = self.output_mid_context(out_dec_level1_mid)
        # out_enc_level1_mid = self.encoder_level1_mid2(out_dec_level1_mid)
        #
        # inp_enc_level2_mid = self.down1_2_mid2(out_enc_level1_mid)
        #
        # latent_mid = self.latent_mid2(inp_enc_level2_mid)
        # BFF3_2 = latent_mid
        # BFF3 = self.BF3(BFF3_1, BFF3_2)
        # BFF3 = F.interpolate(BFF3, scale_factor=0.5)
        #
        # latent_mid = latent_mid + BFF2
        #
        #
        # latent_small = latent_small + BFF3
        #
        #
        # inp_dec_level1_small = self.up2_1_small(latent_small)
        # inp_dec_level1_small = torch.cat([inp_dec_level1_small, out_enc_level1_small], 1)
        # out_dec_level1_small = self.decoder_level1_small(inp_dec_level1_small)
        #
        # small_2_mid = out_dec_level1_small
        #
        # out_dec_level1_small = self.output_small(out_dec_level1_small) + inp_img_small
        #
        # outputs.append(out_dec_level1_small)
        #
        #
        #
        # inp_dec_level1_mid = self.up2_1_mid2(latent_mid)
        # inp_dec_level1_mid = torch.cat([inp_dec_level1_mid, out_enc_level1_mid], 1)
        # out_dec_level1_mid = self.decoder_level1_mid2(inp_dec_level1_mid)
        #
        # small_2_mid = F.interpolate(small_2_mid, scale_factor=2)
        # out_dec_level1_mid = out_dec_level1_mid + small_2_mid
        #
        # mid_2_max = out_dec_level1_mid
        #
        # out_dec_level1_mid = self.output_mid(out_dec_level1_mid) + inp_img_mid
        #
        # outputs.append(out_dec_level1_mid)


        inp_dec_level1_max = self.up2_1_max3(latent_max)
        inp_dec_level1_max = torch.cat([inp_dec_level1_max, out_enc_level1_max], 1)
        # mid_2_max = F.interpolate(mid_2_max, scale_factor=2)
        out_dec_level1_max = self.decoder_level1_max3(inp_dec_level1_max)

        out_dec_level1_max = self.output_max(out_dec_level1_max) + inp_img_max

        outputs.append(out_dec_level1_max)

        return outputs[::-1]


if __name__ == '__main__':
    input = torch.rand(1, 3, 256, 256)
    model = MultiscaleNet()
    output = model(input)
    print(output[0].shape)

    # from fvcore.nn import FlopCountAnalysis, parameter_count_table
    #
    # # 分析FLOPs
    # flops = FlopCountAnalysis(model, input)
    # print("FLOPs: ", flops.total())
    #
    # # 分析parameters
    # print(parameter_count_table(model))