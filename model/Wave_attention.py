import torch
import torch.nn as nn

from model import torch_wavelets
import utils
from einops import rearrange

DWT=torch_wavelets.DWT
IDWT=torch_wavelets.IDWT


######################################################################################################
#结合自注意力模块的res-net

######################################################################################################


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
#################################################################################################
#################################################################################################
#AWmix，我需要的是把小波转换加上去
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
# class AWmix(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1):
#         super(AWmix, self).__init__()
#         #wave
#
#         self.DWT_conv = convd(in_planes*4, out_planes*4, 3, 1)
#         self.IDWT_conv = convd(out_planes, out_planes, 3, 1)
#         self.att_conv=convd(in_planes*2,out_planes,3,1)
#         #感觉不用haar会好一些
#
#         self.in_planes = in_planes
#         self.out_planes = out_planes
#         self.head = head
#         self.kernel_att = kernel_att
#         self.kernel_conv = kernel_conv
#         self.stride = stride
#         self.dilation = dilation
#         self.rate1 = torch.nn.Parameter(torch.Tensor(1))
#         self.rate2 = torch.nn.Parameter(torch.Tensor(1))
#         self.head_dim = self.out_planes // self.head
#
#         self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
#         self.conv2 = nn.Conv2d(in_planes*4, out_planes*4, kernel_size=1)
#         self.conv3 = nn.Conv2d(in_planes*4, out_planes*4, kernel_size=1)
#         self.conv_p = nn.Conv2d(2, self.head_dim, kernel_size=1)
#
#         self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
#         self.pad_att = torch.nn.ReflectionPad2d(self.padding_att)
#         self.unfold = nn.Unfold(kernel_size=self.kernel_att, padding=0, stride=self.stride)
#         self.softmax = torch.nn.Softmax(dim=1)
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#         self.fc = nn.Conv2d(3 * self.head, self.kernel_conv * self.kernel_conv, kernel_size=1, bias=False)
#         self.dep_conv = nn.Conv2d(self.kernel_conv * self.kernel_conv * self.head_dim, out_planes,
#                                   kernel_size=self.kernel_conv, bias=True, groups=self.head_dim, padding=1,
#                                   stride=stride)
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         init_rate_half(self.rate1)
#         init_rate_half(self.rate2)
#         kernel = torch.zeros(self.kernel_conv * self.kernel_conv, self.kernel_conv, self.kernel_conv)
#         for i in range(self.kernel_conv * self.kernel_conv):
#             kernel[i, i // self.kernel_conv, i % self.kernel_conv] = 1.
#         kernel = kernel.squeeze(0).repeat(self.out_planes, 1, 1, 1)
#         self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
#         self.dep_conv.bias = init_rate_0(self.dep_conv.bias)
#
#
#     def _transformer(self, DMT1_yl, DMT1_yh):
#         list_tensor = []
#         for i in range(3):
#             list_tensor.append(DMT1_yh[0][:, :, i, :, :])
#         list_tensor.append(DMT1_yl)
#         return torch.cat(list_tensor, 1)
#
#     def _Itransformer(self, out):
#         yh = []
#         C = int(out.shape[1] / 4)
#         # print(out.shape[0])
#         y = out.reshape((out.shape[0], C, 4, out.shape[-2], out.shape[-1]))
#         yl = y[:, :, 0].contiguous()
#         yh.append(y[:, :, 1:].contiguous())
#
#         return yl, yh
#
#     def forward(self, x):
#         # q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)
#         q=self.conv1(x)#b,c,h,w
#         b,c,h,w=q.shape
#         x_dwtl,x_dwth=DWT(x)#b,c,h/2,w/2
#         x_dwt=self.DWT_conv(self._transformer(x_dwtl,x_dwth))#b,c,h/2,w/2-->b,c*4,h/2,w/2-->b,c*4,h/2,w/2
#         k=self.conv2(x_dwt)
#         k=k.reshape(b,c,h,w)
#         v=self.conv3(x_dwt)
#         v=v.reshape(b,c,h,w)
#
#
#         x_idwt=IDWT(self._Itransformer(x_dwt))
#         x_idwt=self.IDWT_conv(x_idwt)
#
#         #加入小波变化，转入到频域自注意力
#
#         #求出q,k,v
#         scaling = float(self.head_dim) ** -0.5
#         # 这行代码是在定义MultiheadAttention模块时使用的，用于计算注意力分数时的缩放因子。
#         # 具体来说，它将self.head_dim的倒数的平方作为缩放因子，
#         # 其中self.head_dim表示每个注意力头的维度大小。
#         # 这个缩放因子的作用是将注意力分数的大小控制在一个合适的范围内，
#         # 避免由于注意力头的数量或维度过大而导致的数值不稳定或梯度消失等问题。
#         # 在实际应用中，这个缩放因子通常设置为注意力头的维度大小的倒数的平方根，
#         # 这样可以保证注意力分数的期望值为1，从而更好地控制注意力的权重。
#         b, c, h, w = q.shape
#         # h_out, w_out = h // self.stride, w // self.stride
#
#         # ### att
#         # ## positional encoding
#         # pe = self.conv_p(position(h, w, x.is_cuda))
#
#         q_att = q.view(b, -1, h*w) * scaling#防止梯度爆炸或者消失
#         k_att = k.view(b, -1, h*w)
#         v_att = v.view(b, -1, h*w)
#
#         # 计算注意力矩阵
#         attention_scores = torch.bmm(q_att.permute(0,2,1), k_att)
#         attention_scores = attention_scores.clamp(-1,1)
#
#         # 计算输出张量
#         attention_output = torch.bmm(v_att, attention_scores)
#         attention_output = attention_output.view(b, -1, h, w)
#
#         out_att=self.att_conv(torch.cat((attention_output,x_idwt),dim=1))
#
#         ## conv
#         f_all = self.fc(torch.cat(
#             [q.view(b, self.head, self.head_dim, h * w), k.view(b, self.head, self.head_dim, h * w),
#              v.view(b, self.head, self.head_dim, h * w)], 1))
#         f_conv = f_all.permute(0, 2, 1, 3).reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
#         out_conv = self.dep_conv(f_conv)
#
#         return self.rate1 * out_att + self.rate2 * out_conv
####################################################################################################################
####################################################################################################################
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    # expansion = 4
    expansion = 1
    # 我应该固定死他的通道数

    def __init__(self, inplanes, planes, k_att, head, k_conv, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        # self.conv1 = conv1x1(inplanes, width)
        self.conv1 = conv1x1(inplanes, inplanes)
        # self.bn1 = norm_layer(width)
        self.conv2 = AWmix(width, width,)
        # self.bn2 = norm_layer(width)
        # self.conv3 = conv1x1(width, planes * self.expansion)
        self.conv3 = conv1x1(planes, planes )
        # self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, k_att=7, head=4, k_conv=3, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 32
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv = nn.Conv2d(32, self.inplanes, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.conv1 = nn.Conv2d(64, 32, kernel_size=7, stride=1, padding=3,
                               bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0], k_att, head, k_conv)
        self.layer2 = self._make_layer(block, 32, layers[1], k_att, head, k_conv, stride=1,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 32, layers[2], k_att, head, k_conv, stride=1,
                                       dilate=replace_stride_with_dilation[1])
        # self.layer4 = self._make_layer(block, 32, layers[3], k_att, head, k_conv, stride=1,
        #                                dilate=replace_stride_with_dilation[2])
        #这里设置stride=1是为了保证图像尺寸不发生改变
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, rate, k, head, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                # norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, rate, k, head, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, rate, k, head, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.layer1(x)
        x = self.layer2(x)#b,254,h,w->b,512,h,w
        x = self.layer3(x)#b,512,h,w->b,1024,h,w
        # x = self.layer4(x)

        return x
    #此时已经完成了输入为B,C,H,W，输出也为B,C,H,W,同时也提取到了特征

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


##这里可以改变bottlenet的个数，这里初步定为3个
def ACmix_ResNet(layers=[1,1,1], **kwargs):
    return _resnet(Bottleneck, layers, **kwargs)

#####################################################################################################################
#####################################################################################################################
if __name__ == '__main__':
    model = ACmix_ResNet()
    print(model)
    MB = utils.count_parameters_in_MB(model)
    print(MB)
    input = torch.randn([5,32,128,128])
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f'{total_params:,} total parameters.')
    # total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'{total_trainable_params:,} training parameters.')
    print(model(input).shape)
    # print(summary(model, torch.zeros((1, 3, 224, 224)).cuda()))