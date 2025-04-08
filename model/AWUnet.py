"""
这是根据UNet模型搭建出的一个基本网络结构
输入和输出大小是一样的，可以根据需求进行修改
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from model import torch_wavelets
from einops import rearrange
import numbers
DWT=torch_wavelets.DWT.cuda()
IDWT=torch_wavelets.IDWT.cuda()


####################################################################################################################
####################################################################################################################
#AWmix，我需要的是把小波转换加上去
class convd(nn.Module):
    def __init__(self, inputchannel, outchannel, kernel_size, stride):
        super(convd, self).__init__()
        # self.relu = nn.ReLU()
        self.leakrelu = nn.LeakyReLU()
        self.padding = nn.ReflectionPad2d(kernel_size // 2)
        self.conv = nn.Conv2d(inputchannel, outchannel, kernel_size, stride)
        self.ins = nn.InstanceNorm2d(outchannel, affine=True)

    def forward(self, x):
        x = self.conv(self.padding(x))
        # x= self.ins(x)
        x = self.leakrelu(x)
        return x

def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc


def stride(x, stride):
    b, c, h, w = x.shape
    return x[:, :, ::stride, ::stride]


def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)


def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)
class AWmix(nn.Module):
    def __init__(self, dim,num_heads=4,bias=False):
        super(AWmix, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        #attn
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv2 = nn.Conv2d(dim * 4, dim*4, kernel_size=1)
        self.conv3 = nn.Conv2d(dim*4, dim*4, kernel_size=1)
        self.project_out = nn.Conv2d(dim*2, dim, kernel_size=1, bias=bias)
        #wave
        self.DWT_conv = convd(dim*4, dim*4, 3, 1)
        self.IDWT_conv = convd(dim, dim, 3, 1)

    def _transformer(self, DMT1_yl, DMT1_yh):
        list_tensor = []
        for i in range(3):
            list_tensor.append(DMT1_yh[0][:, :, i, :, :])
        list_tensor.append(DMT1_yl)
        return torch.cat(list_tensor, 1)

    def _Itransformer(self, out):
        yh = []
        C = int(out.shape[1] / 4)
        # print(out.shape[0])
        y = out.reshape((out.shape[0], C, 4, out.shape[-2], out.shape[-1]))
        yl = y[:, :, 0].contiguous()
        yh.append(y[:, :, 1:].contiguous())

        return yl, yh

    def forward(self, x):
        # q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)
        b, c, h, w = x.shape
        q=self.conv1(x)#b,c,h,w
        b,c,h,w=q.shape
        x_dwtl,x_dwth=DWT(x)#b,c,h/2,w/2
        x_dwt=self.DWT_conv(self._transformer(x_dwtl,x_dwth))#b,c,h/2,w/2-->b,c*4,h/2,w/2-->b,c*4,h/2,w/2
        k=self.conv2(x_dwt)
        v=self.conv3(x_dwt)
        k = k.reshape(b, c, h, w)
        v = v.reshape(b, c, h, w)
        x_idwt=IDWT(self._Itransformer(x_dwt))
        x_idwt=self.IDWT_conv(x_idwt)

        # qkv = self.qkv_dwconv(self.qkv(x))#(5,144,128,128)
        # q, k, v = qkv.chunk(3, dim=1)#(5,48,128,128)

        #attn
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)#(5,1,48,16384)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)#(5,1,48,16384)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)#(5,1,48,16384)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out_att = (attn @ v)
        out_att = rearrange(out_att, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out_att = self.project_out(torch.cat((out_att,x_idwt),dim=1))

        return out_att
####################################################################################################################
####################################################################################################################
class SelfAttention(nn.Module):
    '''这个函数定义了一个自注意力操作
    输入:要处理的注意力数据
    输出：处理后的数据
    维度：输入前与输入后维度不变
    '''
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True) #输入注意力的channel数，注意力的头数，以及输入的第一个维度是batch
        self.ln = nn.LayerNorm([channels])  #表明只对channels维度进行归一化
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2) #先改变维度到[batch_size,h*w,channels]
        #[batch_size,h*w,channels]
        x_ln = self.ln(x)         #在channels这个维度进行归一化
        #[batch_size,h*w,channels]
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)  #这个函数会返回一个注意力值和注意力权重（softmax（q*v）） 输入是kqv这里kqv是一样的
        #[batch_size,h*w,channels]
        attention_value = attention_value + x  #残差
        attention_value = self.ff_self(attention_value) + attention_value  #残差
        #[batch_size,h*w,channels]
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)
# 基本卷积块
class DoubleConv(nn.Module):
    '''这个函数定义了卷积操作具体流程是：卷积 分组归一化 激活函数 卷积 分组归一化 残差连接可选
    输入:待处理的图像 中间的channels
    输出:卷积操作后的特征图
    '''
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),  #第一个参数表示分组的数量 第二个参数表示输入的通道数
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    '''这个函数定义了下采样，同时将时间步长编码加入到图像中,流程：最大池化，
    双卷积同时进行残差连接，对时间步长编码进行一个线性变化然后升维加到卷积操作后的数据中
    输入:待下采样的特征图，时间步长编码
    输出：卷积后与时间步长融合的特征图
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), #最大池化 池化窗口是2 所以高宽都会缩小到原来的两倍
            DoubleConv(in_channels, in_channels, residual=True), #做卷积操作
            DoubleConv(in_channels, out_channels),
        )


    def forward(self, x):
        #t[batch_size,256]
        x = self.maxpool_conv(x)
        '''将时间步长编码加入到卷积操作之后的数据里'''
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        '''这个函数做了一个上采样的操作，流程是：先做双线性插值进行上采样，然后做双卷积，对时间步长进行线性变化加到卷积后的数据中
        输入：待卷积的特征图，与之对应的下采样的卷积后的特征图，时间步长编码
        输出：卷积操作后与时间步融合的特征图
        '''
        super().__init__()
        '''这里进行了一个上采样操作，使用双线性插值法 放大因子是2说明高宽会被放大到原来的两倍 角对其说明其会通过对其角上的元素避免边缘效应'''
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(   #做一个双卷积
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

    def forward(self, x, skip_x):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)  #与对应的下采样进行残差连接
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, c_in=32, c_out=32, device="cuda"):  #定义了进入模型的通道数和输出模型的通道数
        super().__init__()
        self.device = device
        # self.time_dim = time_dim   #时间的嵌入维度
        self.inc = DoubleConv(c_in, 64)      #卷积
        self.down1 = Down(64, 128)           #下采样  通道数增加到128,尺寸缩小为原来的1/2
        self.sa1 = AWmix(128)
        # self.sa1 = SelfAttention(128, 32)    #自注意力机制
        self.down2 = Down(128, 256)          #下采样  通道数增加256
        self.sa2 = AWmix(256)
        # self.sa2 = SelfAttention(256, 16)    #自注意力机制
        self.down3 = Down(256, 256)          #上采样  通道数不变256
        self.sa3 = AWmix(256)
        # self.sa3 = SelfAttention(256, 8)     #自注意力机制

        self.bot1 = DoubleConv(256, 512)     #中间的卷积层
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)             #上采样
        self.sa4 = SelfAttention(128, 16)   #自注意力机制
        self.sa4 = AWmix(128)
        self.up2 = Up(256, 64)              #上采样
        self.sa5 = SelfAttention(64, 32)    #自注意力机制
        self.sa5 = AWmix(64)
        self.up3 = Up(128, 64)              #上采样
        self.sa6 = SelfAttention(64, 64)    #自注意力机制
        self.sa6 = AWmix(64)

        self.outc = nn.Conv2d(64, c_out, kernel_size=1) #卷积出来

    def forward(self, x):
        #输入为(5,32,128,128)的话
        x1 = self.inc(x)#(5,64,128,128)
        x2 = self.down1(x1)#(5,128,64,64)
        x2 = self.sa1(x2)
        x3 = self.down2(x2)
        x3 = self.sa2(x3)
        x4 = self.down3(x3)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3)
        x = self.sa4(x)
        x = self.up2(x, x2)
        x = self.sa5(x)
        x = self.up3(x, x1)
        x = self.sa6(x)
        output = self.outc(x)
        return output

########################################################################################################################
########################################################################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

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
        return x / torch.sqrt(sigma+1e-5) * self.weight

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
        return x / torch.sqrt(sigma+1e-5) * self.weight

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
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
class NextAttentionImplZ(nn.Module):
    def __init__(self, num_dims, num_heads, bias) -> None:
        super().__init__()
        self.num_dims = num_dims
        self.num_heads = num_heads
        self.q1 = nn.Conv2d(num_dims, num_dims * 3, kernel_size=1, bias=bias)
        self.q2 = nn.Conv2d(num_dims * 3, num_dims * 3, kernel_size=3, padding=1, groups=num_dims * 3, bias=bias)
        self.q3 = nn.Conv2d(num_dims * 3, num_dims * 3, kernel_size=3, padding=1, groups=num_dims * 3, bias=bias)

        self.fac = nn.Parameter(torch.ones(1))
        self.fin = nn.Conv2d(num_dims, num_dims, kernel_size=1, bias=bias)
        return

    def forward(self, x):
        # x: [n, c, h, w]
        n, c, h, w = x.size()
        n_heads, dim_head = self.num_heads, c // self.num_heads
        reshape = lambda x: rearrange(x, "n (nh dh) h w -> (n nh h) w dh", nh=n_heads, dh=dim_head)

        qkv = self.q3(self.q2(self.q1(x)))
        q, k, v = map(reshape, qkv.chunk(3, dim=1))
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # fac = dim_head ** -0.5
        res = k.transpose(-2, -1)
        res = torch.matmul(q, res) * self.fac
        res = torch.softmax(res, dim=-1)

        res = torch.matmul(res, v)
        res = rearrange(res, "(n nh h) w dh -> n (nh dh) h w", nh=n_heads, dh=dim_head, n=n, h=h)
        res = self.fin(res)

        return res

### Axis-based Multi-head Self-Attention (row and col attention)
class NextAttentionZ(nn.Module):
    def __init__(self, num_dims, num_heads=1, bias=True) -> None:
        super().__init__()
        assert num_dims % num_heads == 0
        self.num_dims = num_dims
        self.num_heads = num_heads
        self.row_att = NextAttentionImplZ(num_dims, num_heads, bias)
        self.col_att = NextAttentionImplZ(num_dims, num_heads, bias)
        return

    def forward(self, x: torch.Tensor):
        assert len(x.size()) == 4

        x = self.row_att(x)
        x = x.transpose(-2, -1)
        x = self.col_att(x)
        x = x.transpose(-2, -1)

        return x

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x2)*x1 + F.gelu(x1)*x2
        x = self.project_out(x)
        return x


######  Axis-based Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=1, ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias'):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = NextAttentionZ(dim, num_heads)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

if __name__ == '__main__':
    a = torch.randn(5, 32, 128, 128)
    net = UNet()
    print(net(a).shape)

