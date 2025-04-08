import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch_wavelets import DWTForward, DWTInverse
from model import FLW_net
# from model import Wavemix
from model import Wave_attention
import numpy as np
from model import AWUnet
def make_model(args, parent=False):
    return Enlightnet(args)


class convd(nn.Module):
    def __init__(self, inputchannel, outchannel, kernel_size, stride):
        super(convd, self).__init__()
        self.relu = nn.ReLU()
        self.padding = nn.ReflectionPad2d(kernel_size // 2)
        self.conv = nn.Conv2d(inputchannel, outchannel, kernel_size, stride)
        self.ins = nn.InstanceNorm2d(outchannel, affine=True)

    def forward(self, x):
        x = self.conv(self.padding(x))
        # x= self.ins(x)
        x = self.relu(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Upsample, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.relu = nn.LeakyReLU()

    def forward(self, x, y):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        out = self.relu(out)
        out = F.interpolate(out, y.size()[2:])
        return out

    # 这个方法接收两个参数`x`和`y`，分别表示输入的特征图和上采样后的特征图。在这个方法中，首先对输入的特征图进行反射填充，然后使用转置卷积对其进行上采样。
    # 转置卷积的参数包括输入通道数`in_channels`、输出通道数`out_channels`、卷积核大小`kernel_size`和步长`stride`。
    # 接着，使用ReLU激活函数对上采样后的特征图进行非线性变换。
    # 最后，使用`F.interpolate`方法将上采样后的特征图的大小调整为与`y`相同，得到最终的上采样结果，并将其返回。

###############################################################################
#IFM模块，用来将结构图和原始图融合
class Prior_Sp(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim=32):
        super(Prior_Sp, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
        self.key_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)

        self.gamma1 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
        # self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
        # self.softmax  = nn.Softmax(dim=-1)
        self.rlu = nn.LeakyReLU()

    def forward(self, x, prior):
        x_q = self.query_conv(x)
        prior_k = self.key_conv(prior)
        energy = x_q * prior_k
        attention = self.rlu(energy)
        #因为梯度爆炸所以改成ReLu
        # print(attention.size(),x.size())
        attention_x = x * attention
        attention_p = prior * attention

        x_gamma = self.gamma1(torch.cat((x, attention_x), dim=1))
        x_out = x * x_gamma[:, [0], :, :] + attention_x * x_gamma[:, [1], :, :]

        p_gamma = self.gamma2(torch.cat((prior, attention_p), dim=1))
        prior_out = prior * p_gamma[:, [0], :, :] + attention_p * p_gamma[:, [1], :, :]

        res = torch.cat((x_out, prior_out), dim=1)
        return res
        # return x_out, prior_out
###############################################################################

class Enlightnet(nn.Module):
    def __init__(self, args):
        super(Enlightnet, self).__init__()
        n_feats = args.n_feats
        blocks = args.n_resblocks

        scale_factor = args.scale_factor
        nbins=args.nbins
        self.FLWnet=FLW_net.enhance_net_nopool(scale_factor,nbins)
        self.upsample=Upsample(n_feats,n_feats,3,2)
        self.upsample2=Upsample(3,3,3,2)
        #convolutional Layer
        self.conv_init1_1 = convd(3, n_feats // 2, 3, 1)
        self.conv_init1_3 = convd(3, n_feats // 2, 3, 1)
        self.conv_init2_1 = convd(n_feats // 2, n_feats, 3, 1)
        self.conv_init2_3 = convd(n_feats // 2, n_feats, 3, 1)
        self.channel_resize=convd(3,32,3,1)
        # fuse res
        self.IMF_1=Prior_Sp()
        self.IMF_2=Prior_Sp()
        self.IMF_3=Prior_Sp()
        self.fuse_res_1 = convd(n_feats*2, n_feats, 3, 1)
        self.fuse_res_2 = convd(n_feats*2, n_feats, 3, 1)
        self.fuse_res_3 = convd(n_feats*2, n_feats, 3, 1)
        #多层次小波变换网络
        # self.Muti_Wave_net=Wavemix.Level3Waveblock(mult=2,in_channel=n_feats,out_channel=n_feats)
        #残差小波自注意力网络
        self.Res_Wave_net_1=Wave_attention.ACmix_ResNet()
        self.Res_Wave_net_2=Wave_attention.ACmix_ResNet()
        self.Res_Wave_net_3=Wave_attention.ACmix_ResNet()
        #小波注意U-net
        self.AWUnet=AWUnet.UNet()
        self.AWUnet_v2_1=AWUnet.UNet(3,32)
        self.AWUnet_v2_2=AWUnet.UNet(3,32)
        self.AWUnet_v2_3=AWUnet.UNet(3,32)
        #恢复通道数
        self.ag1 = convd(n_feats*2,n_feats,3,1)
        self.ag2 = convd(n_feats*3,n_feats,3,1)
        self.ag2_en = convd(n_feats*2, n_feats, 3, 1)
        self.ag_en = convd(n_feats*3, n_feats, 3, 1)

        self.output1 = nn.Conv2d(n_feats, 3, 3, 1, padding=1)
        self.output2 = nn.Conv2d(n_feats, 3, 3, 1, padding=1)
        self.output3 = nn.Conv2d(n_feats, 3, 3, 1, padding=1)

        # self._initialize_weights()

    def compute_hist(self,x, nbins=14):
        tensor_np = x.cpu().detach().numpy()
        # 计算直方图
        com_hist = []
        for i in range(tensor_np.shape[0]):
            # 计算当前图像的直方图
            low_im_filter_max = np.max(tensor_np, axis=2, keepdims=True)  # positive
            xxx, bins_of_im = np.histogram(low_im_filter_max, bins=int(nbins - 2),
                                           range=(np.nanmin(low_im_filter_max), np.nanmax(low_im_filter_max)))
            hist_c = np.reshape(xxx, [1, 1, nbins - 2])
            hist = np.zeros([1, 1, int(nbins + 1)])
            hist[:, :, 0:nbins - 2] = np.array(hist_c, dtype=np.float32) / np.sum(hist_c)
            hist[:, :, nbins - 2:nbins - 1] = np.min(low_im_filter_max)
            hist[:, :, nbins - 1:nbins] = np.max(low_im_filter_max)
            # hist[:, :, -1] = high_im_filter_max.mean()
            hist[ :, :, -1] = low_im_filter_max.mean()

            com_hist.append(hist)

        com_hist=np.array(com_hist)
        return torch.from_numpy(com_hist).float().permute(0,3,1,2)

    def forward(self, x,hist):
        ######################################################################
        #Vision-1
        ######################################################################
        # struct_x1=self.FLWnet(x,hist)#利用轻量级网络生成的简单结构图
        # # y_shape=(x.shape[0],3,128,128)
        # struct_x1=self.channel_resize(struct_x1)#16,3,128,128->16,32,128,128
        # x_init_1=self.conv_init2(self.conv_init1(x))#x_init_1:16,32,128,128
        # # x_prior1=self.IMF(x_init_1,struct_x1)
        # # x_prior_out1=self.fuse_res(x_prior1)#对结构图和原始图进行特征融合
        # # x1=self.Muti_Wave_net(self.fuse_res(self.IMF(x_init_1,struct_x1)))#使用基于小波变化的多层次网络结构对细节进行增强
        # # x1 = self.Res_Wave_net(self.fuse_res(self.IMF(x_init_1, struct_x1)))#使用基于残差结构的小波自注意力对细节进行增强
        # x1=self.AWUnet(self.fuse_res(self.IMF(x_init_1, struct_x1)))
        # # x1=self.upsample(x1,x)
        # out1=self.output1(x1)#在进行一次卷积操作
        #
        # hist=self.compute_hist(out1).cuda()
        # struct_x2=self.FLWnet(out1,hist)
        # x_init_2=self.ag1(torch.cat((x1,x_init_1),dim=1))
        # struct_x2=self.channel_resize(struct_x2)
        # # x_prior_out2 = self.upsample(x_prior_out2, torch.ones(y_shape))
        # # x2=self.Muti_Wave_net(self.fuse_res(self.IMF(x_init_2,struct_x2)))
        # # x2=self.Res_Wave_net(self.fuse_res(self.IMF(x_init_2,struct_x2)))
        # x2=self.AWUnet(self.fuse_res(self.IMF(x_init_2,struct_x2)))
        # # x2 = self.upsample(x2, x1)
        # x2_=self.ag2_en(torch.cat([x2,x1],dim=1))
        # out2=self.output2(x2_)
        #
        # hist=self.compute_hist(out2).cuda()
        # struct_x3=self.FLWnet(out2,hist)
        # struct_x3=self.channel_resize(struct_x3)
        # x_init_3=self.conv_init2(self.conv_init1(x))
        # # x_prior3=self.IMF(x_init3,struct_x3)
        # # x_prior_out3=self.fuse_res(x_prior3)
        # # x_prior_out3 = self.upsample(x_prior_out3, torch.ones(y_shape))
        # # x3=self.Muti_Wave_net(self.fuse_res(self.IMF(x_init3,struct_x3)))
        # # x3 = self.Res_Wave_net(self.fuse_res(self.IMF(x_init_3, struct_x3)))
        # x3=self.AWUnet(self.fuse_res(self.IMF(x_init_3, struct_x3)))
        # # x3 = self.upsample(x3, x2)
        # x3_=self.ag_en(torch.cat([x3,x2,x1],dim=1))
        # # x3_=self.FLWnet(x3)#因为这里最后一步进行了上采样，我觉得会损失细节，所以感觉加一个FLWnet比较好
        # out3=self.output3(x3_)
        #
        # return out1,out2,out3;
        ##########################################################################################################
        #vision-2
        ##########################################################################################################
        struct_x1=self.AWUnet_v2_1(x)#利用U-net来做结构引导
        x_init_1=self.conv_init2_1(self.conv_init1_1(x))#x_init_1:16,32,128,128
        x1=self.Res_Wave_net_1(self.fuse_res_1(self.IMF_1(x_init_1, struct_x1)))
        # x1=self.upsample(x1,x)
        out1=self.output1(x1)#在进行一次卷积操作

        struct_x2=self.AWUnet_v2_2(out1)
        x_init_2=self.ag1(torch.cat((x1,x_init_1),dim=1))
        x2=self.Res_Wave_net_2(self.fuse_res_2(self.IMF_2(x_init_2,struct_x2)))
        x2_=self.ag2_en(torch.cat([x2,x1],dim=1))
        out2=self.output2(x2_)

        struct_x3=self.AWUnet_v2_3(out2)
        x_init_3=self.conv_init2_3(self.conv_init1_3(x))
        x3=self.Res_Wave_net_3(self.fuse_res_3(self.IMF_3(x_init_3, struct_x3)))
        x3_=self.ag_en(torch.cat([x3,x2,x1],dim=1))
        out3=self.output3(x3_)

        return out1,out2,out3;


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

# sum(param.numel() for param in net.parameters())
