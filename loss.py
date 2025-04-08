import torch.nn as nn

from torchvision.models.vgg import vgg16
import pytorch_ssim
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

from IQA_pytorch import SSIM

########################################################################################################################
import numpy as np

class HistogramLoss(nn.Module):
    def __init__(self, num_bins=256):
        super(HistogramLoss, self).__init__()
        self.num_bins = num_bins

    def forward(self, input, target):
        # 将CUDA张量复制到主机内存中
        input_cpu = input.detach().cpu()
        target_cpu = target.detach().cpu()

        # 计算输入图像和目标图像的直方图
        input_hist = np.histogram(input_cpu.numpy(), bins=self.num_bins, range=(0, 1))[0]
        target_hist = np.histogram(target_cpu.numpy(), bins=self.num_bins, range=(0, 1))[0]

        # 将直方图转换为PyTorch张量
        input_hist = torch.from_numpy(input_hist).float()
        target_hist = torch.from_numpy(target_hist).float()

        # 对直方图进行归一化
        input_hist = input_hist / torch.sum(input_hist)
        target_hist = target_hist / torch.sum(target_hist)

        # 计算直方图的差异
        hist_diff = torch.abs(input_hist - target_hist)

        # 将直方图的差异作为损失函数的值
        loss = torch.sum(hist_diff)

        return loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

###########################################################################################################################

class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()
        self.l2_loss = nn.MSELoss()
        self.smooth_loss = SmoothLoss()

    def forward(self, input, illu):
        Fidelity_Loss = self.l2_loss(illu, input)
        Smooth_Loss = self.smooth_loss(input, illu)
        return 1.5*Fidelity_Loss + Smooth_Loss



class SmoothLoss(nn.Module):
    def __init__(self):
        super(SmoothLoss, self).__init__()
        self.sigma = 10

    def rgb2yCbCr(self, input_im):
        im_flat = input_im.contiguous().view(-1, 3).float()
        mat = torch.Tensor([[0.257, -0.148, 0.439], [0.564, -0.291, -0.368], [0.098, 0.439, -0.071]]).cuda()
        bias = torch.Tensor([16.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0]).cuda()
        temp = im_flat.mm(mat) + bias
        out = temp.view(input_im.shape[0], 3, input_im.shape[2], input_im.shape[3])
        return out

    # output: output      input:input
    def forward(self, input, output):
        self.output = output
        self.input = self.rgb2yCbCr(input)
        sigma_color = -1.0 / (2 * self.sigma * self.sigma)
        w1 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :] - self.input[:, :, :-1, :], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w2 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :] - self.input[:, :, 1:, :], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w3 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, 1:] - self.input[:, :, :, :-1], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w4 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, :-1] - self.input[:, :, :, 1:], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w5 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :-1] - self.input[:, :, 1:, 1:], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w6 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, 1:] - self.input[:, :, :-1, :-1], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w7 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :-1] - self.input[:, :, :-1, 1:], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w8 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, 1:] - self.input[:, :, 1:, :-1], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w9 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :] - self.input[:, :, :-2, :], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w10 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :] - self.input[:, :, 2:, :], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w11 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, 2:] - self.input[:, :, :, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w12 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, :-2] - self.input[:, :, :, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w13 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :-1] - self.input[:, :, 2:, 1:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w14 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, 1:] - self.input[:, :, :-2, :-1], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w15 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :-1] - self.input[:, :, :-2, 1:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w16 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, 1:] - self.input[:, :, 2:, :-1], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w17 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :-2] - self.input[:, :, 1:, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w18 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, 2:] - self.input[:, :, :-1, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w19 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :-2] - self.input[:, :, :-1, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w20 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, 2:] - self.input[:, :, 1:, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w21 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :-2] - self.input[:, :, 2:, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w22 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, 2:] - self.input[:, :, :-2, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w23 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :-2] - self.input[:, :, :-2, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w24 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, 2:] - self.input[:, :, 2:, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        p = 1.0

        pixel_grad1 = w1 * torch.norm((self.output[:, :, 1:, :] - self.output[:, :, :-1, :]), p, dim=1, keepdim=True)
        pixel_grad2 = w2 * torch.norm((self.output[:, :, :-1, :] - self.output[:, :, 1:, :]), p, dim=1, keepdim=True)
        pixel_grad3 = w3 * torch.norm((self.output[:, :, :, 1:] - self.output[:, :, :, :-1]), p, dim=1, keepdim=True)
        pixel_grad4 = w4 * torch.norm((self.output[:, :, :, :-1] - self.output[:, :, :, 1:]), p, dim=1, keepdim=True)
        pixel_grad5 = w5 * torch.norm((self.output[:, :, :-1, :-1] - self.output[:, :, 1:, 1:]), p, dim=1, keepdim=True)
        pixel_grad6 = w6 * torch.norm((self.output[:, :, 1:, 1:] - self.output[:, :, :-1, :-1]), p, dim=1, keepdim=True)
        pixel_grad7 = w7 * torch.norm((self.output[:, :, 1:, :-1] - self.output[:, :, :-1, 1:]), p, dim=1, keepdim=True)
        pixel_grad8 = w8 * torch.norm((self.output[:, :, :-1, 1:] - self.output[:, :, 1:, :-1]), p, dim=1, keepdim=True)
        pixel_grad9 = w9 * torch.norm((self.output[:, :, 2:, :] - self.output[:, :, :-2, :]), p, dim=1, keepdim=True)
        pixel_grad10 = w10 * torch.norm((self.output[:, :, :-2, :] - self.output[:, :, 2:, :]), p, dim=1, keepdim=True)
        pixel_grad11 = w11 * torch.norm((self.output[:, :, :, 2:] - self.output[:, :, :, :-2]), p, dim=1, keepdim=True)
        pixel_grad12 = w12 * torch.norm((self.output[:, :, :, :-2] - self.output[:, :, :, 2:]), p, dim=1, keepdim=True)
        pixel_grad13 = w13 * torch.norm((self.output[:, :, :-2, :-1] - self.output[:, :, 2:, 1:]), p, dim=1, keepdim=True)
        pixel_grad14 = w14 * torch.norm((self.output[:, :, 2:, 1:] - self.output[:, :, :-2, :-1]), p, dim=1, keepdim=True)
        pixel_grad15 = w15 * torch.norm((self.output[:, :, 2:, :-1] - self.output[:, :, :-2, 1:]), p, dim=1, keepdim=True)
        pixel_grad16 = w16 * torch.norm((self.output[:, :, :-2, 1:] - self.output[:, :, 2:, :-1]), p, dim=1, keepdim=True)
        pixel_grad17 = w17 * torch.norm((self.output[:, :, :-1, :-2] - self.output[:, :, 1:, 2:]), p, dim=1, keepdim=True)
        pixel_grad18 = w18 * torch.norm((self.output[:, :, 1:, 2:] - self.output[:, :, :-1, :-2]), p, dim=1, keepdim=True)
        pixel_grad19 = w19 * torch.norm((self.output[:, :, 1:, :-2] - self.output[:, :, :-1, 2:]), p, dim=1, keepdim=True)
        pixel_grad20 = w20 * torch.norm((self.output[:, :, :-1, 2:] - self.output[:, :, 1:, :-2]), p, dim=1, keepdim=True)
        pixel_grad21 = w21 * torch.norm((self.output[:, :, :-2, :-2] - self.output[:, :, 2:, 2:]), p, dim=1, keepdim=True)
        pixel_grad22 = w22 * torch.norm((self.output[:, :, 2:, 2:] - self.output[:, :, :-2, :-2]), p, dim=1, keepdim=True)
        pixel_grad23 = w23 * torch.norm((self.output[:, :, 2:, :-2] - self.output[:, :, :-2, 2:]), p, dim=1, keepdim=True)
        pixel_grad24 = w24 * torch.norm((self.output[:, :, :-2, 2:] - self.output[:, :, 2:, :-2]), p, dim=1, keepdim=True)

        ReguTerm1 = torch.mean(pixel_grad1) \
                    + torch.mean(pixel_grad2) \
                    + torch.mean(pixel_grad3) \
                    + torch.mean(pixel_grad4) \
                    + torch.mean(pixel_grad5) \
                    + torch.mean(pixel_grad6) \
                    + torch.mean(pixel_grad7) \
                    + torch.mean(pixel_grad8) \
                    + torch.mean(pixel_grad9) \
                    + torch.mean(pixel_grad10) \
                    + torch.mean(pixel_grad11) \
                    + torch.mean(pixel_grad12) \
                    + torch.mean(pixel_grad13) \
                    + torch.mean(pixel_grad14) \
                    + torch.mean(pixel_grad15) \
                    + torch.mean(pixel_grad16) \
                    + torch.mean(pixel_grad17) \
                    + torch.mean(pixel_grad18) \
                    + torch.mean(pixel_grad19) \
                    + torch.mean(pixel_grad20) \
                    + torch.mean(pixel_grad21) \
                    + torch.mean(pixel_grad22) \
                    + torch.mean(pixel_grad23) \
                    + torch.mean(pixel_grad24)
        total_term = ReguTerm1
        return total_term




class L_retouch_mean(nn.Module):

    def __init__(self):
        super(L_retouch_mean, self).__init__()
        self.ssim_loss = pytorch_ssim.SSIM()

    def forward(self, x, y):
        x = x.max(1, keepdim=True)[0]
        y = y.max(1, keepdim=True)[0]
        L3_retouch_mean = torch.mean(torch.pow(x - torch.mean(y, [2, 3], keepdim=True), 2))
        L4_retouch_ssim = 1 - torch.mean(self.ssim_loss(x, y))

        return L3_retouch_mean + L4_retouch_ssim


class L_recon(nn.Module):

    def __init__(self):
        super(L_recon, self).__init__()
        # self.ssim_loss = pytorch_ssim.SSIM()
    def forward(self, R_low, high):
        L1 = torch.abs(R_low - high).mean()
        # L_temp=self.ssim_loss(R_low,high)
        L_temp = ssim(R_low, high)
        L2 = (1 - L_temp).mean()
        # return L1,L2
        return L1,L2

class L_recon_low(nn.Module):

    def __init__(self):
        super(L_recon_low, self).__init__()
        # self.ssim_loss = pytorch_ssim.SSIM()

    def forward(self, R_low, high, ill):
        L1 = (R_low * ill - high * torch.log(R_low * ill + 0.0001)).mean()

        return L1


class L_color(nn.Module):
    # 这段代码定义了一个名为L_color_zy的PyTorch模块，用于计算两个张量x和y之间的颜色损失。
    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x, y):
        product_separte_color = (x * y).mean(1, keepdim=True)
        x_abs = (x ** 2).mean(1, keepdim=True) ** 0.5
        y_abs = (y ** 2).mean(1, keepdim=True) ** 0.5
        loss1 = (1 - product_separte_color / (x_abs * y_abs + 0.00001)).mean() + torch.mean(
            torch.acos(product_separte_color / (x_abs * y_abs + 0.00001)))

        return loss1


class L_grad_cosist(nn.Module):
    # 计算两个张量x和y之间的梯度一致性损失
    def __init__(self):
        super(L_grad_cosist, self).__init__()
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)

    def gradient_of_one_channel(self, x, y):
        D_org_right = F.conv2d(x, self.weight_right, padding="same")
        D_org_down = F.conv2d(x, self.weight_down, padding="same")
        D_enhance_right = F.conv2d(y, self.weight_right, padding="same")
        D_enhance_down = F.conv2d(y, self.weight_down, padding="same")
        return torch.abs(D_org_right), torch.abs(D_enhance_right), torch.abs(D_org_down), torch.abs(D_enhance_down)

    def gradient_Consistency_loss_patch(self, x, y):
        # B*C*H*W
        min_x = torch.abs(x.min(2, keepdim=True)[0].min(3, keepdim=True)[0]).detach()
        min_y = torch.abs(y.min(2, keepdim=True)[0].min(3, keepdim=True)[0]).detach()
        x = x - min_x
        y = y - min_y
        # B*1*1,3
        product_separte_color = (x * y).mean([2, 3], keepdim=True)
        x_abs = (x ** 2).mean([2, 3], keepdim=True) ** 0.5
        y_abs = (y ** 2).mean([2, 3], keepdim=True) ** 0.5
        loss1 = (1 - product_separte_color / (x_abs * y_abs + 0.00001)).mean() + torch.mean(
            torch.acos(product_separte_color / (x_abs * y_abs + 0.00001)))

        product_combine_color = torch.mean(product_separte_color, 1, keepdim=True)
        x_abs2 = torch.mean(x_abs ** 2, 1, keepdim=True) ** 0.5
        y_abs2 = torch.mean(y_abs ** 2, 1, keepdim=True) ** 0.5
        loss2 = torch.mean(1 - product_combine_color / (x_abs2 * y_abs2 + 0.00001)) + torch.mean(
            torch.acos(product_combine_color / (x_abs2 * y_abs2 + 0.00001)))
        return loss1 + loss2

    def forward(self, x, y):
        x_R1, y_R1, x_R2, y_R2 = self.gradient_of_one_channel(x[:, 0:1, :, :], y[:, 0:1, :, :])
        x_G1, y_G1, x_G2, y_G2 = self.gradient_of_one_channel(x[:, 1:2, :, :], y[:, 1:2, :, :])
        x_B1, y_B1, x_B2, y_B2 = self.gradient_of_one_channel(x[:, 2:3, :, :], y[:, 2:3, :, :])
        x = torch.cat([x_R1, x_G1, x_B1, x_R2, x_G2, x_B2], 1)
        y = torch.cat([y_R1, y_G1, y_B1, y_R2, y_G2, y_B2], 1)

        B, C, H, W = x.shape
        loss = self.gradient_Consistency_loss_patch(x, y)
        loss1 = 0
        loss1 += self.gradient_Consistency_loss_patch(x[:, :, 0:H // 2, 0:W // 2], y[:, :, 0:H // 2, 0:W // 2])
        loss1 += self.gradient_Consistency_loss_patch(x[:, :, H // 2:, 0:W // 2], y[:, :, H // 2:, 0:W // 2])
        loss1 += self.gradient_Consistency_loss_patch(x[:, :, 0:H // 2, W // 2:], y[:, :, 0:H // 2, W // 2:])
        loss1 += self.gradient_Consistency_loss_patch(x[:, :, H // 2:, W // 2:], y[:, :, H // 2:, W // 2:])

        return loss  # +loss1#+torch.mean(torch.abs(x-y))#+loss1


class L_bright_cosist(nn.Module):
    # 用于计算两个张量x和y之间的亮度一致性损失
    def __init__(self):
        super(L_bright_cosist, self).__init__()

    def gradient_Consistency_loss_patch(self, x, y):
        # B*C*H*W
        min_x = torch.abs(x.min(2, keepdim=True)[0].min(3, keepdim=True)[0]).detach()
        min_y = torch.abs(y.min(2, keepdim=True)[0].min(3, keepdim=True)[0]).detach()
        x = x - min_x
        y = y - min_y
        # B*1*1,3
        product_separte_color = (x * y).mean([2, 3], keepdim=True)
        x_abs = (x ** 2).mean([2, 3], keepdim=True) ** 0.5
        y_abs = (y ** 2).mean([2, 3], keepdim=True) ** 0.5
        loss1 = (1 - product_separte_color / (x_abs * y_abs + 0.00001)).mean() + torch.mean(
            torch.acos(product_separte_color / (x_abs * y_abs + 0.00001)))

        product_combine_color = torch.mean(product_separte_color, 1, keepdim=True)
        x_abs2 = torch.mean(x_abs ** 2, 1, keepdim=True) ** 0.5
        y_abs2 = torch.mean(y_abs ** 2, 1, keepdim=True) ** 0.5
        loss2 = torch.mean(1 - product_combine_color / (x_abs2 * y_abs2 + 0.00001)) + torch.mean(
            torch.acos(product_combine_color / (x_abs2 * y_abs2 + 0.00001)))
        return loss1 + loss2

    def forward(self, x, y):
        B, C, H, W = x.shape
        loss = self.gradient_Consistency_loss_patch(x, y)
        loss1 = 0
        loss1 += self.gradient_Consistency_loss_patch(x[:, :, 0:H // 2, 0:W // 2], y[:, :, 0:H // 2, 0:W // 2])
        loss1 += self.gradient_Consistency_loss_patch(x[:, :, H // 2:, 0:W // 2], y[:, :, H // 2:, 0:W // 2])
        loss1 += self.gradient_Consistency_loss_patch(x[:, :, 0:H // 2, W // 2:], y[:, :, 0:H // 2, W // 2:])
        loss1 += self.gradient_Consistency_loss_patch(x[:, :, H // 2:, W // 2:], y[:, :, H // 2:, W // 2:])

        return loss  # +loss1#+torch.mean(torch.abs(x-y))#+loss1


class L_diff_zy(nn.Module):

    def __init__(self):
        super(L_diff_zy, self).__init__()
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)

    def diff_zy(self, input_I, input_R):

        input_I_W_x, input_I_I_x = self.gradient_n_diff(input_I, "x")
        input_R_W_x, input_R_I_x = self.gradient_n_diff(input_R, "x")
        input_I_W_y, input_I_I_y = self.gradient_n_diff(input_I, "y")
        input_R_W_y, input_R_I_y = self.gradient_n_diff(input_R, "y")
        return torch.mean(input_I_I_x - input_R_I_x * torch.log(input_I_I_x + 0.0001)) + torch.mean(
            input_I_I_y - input_R_I_y * torch.log(input_I_I_y + 0.0001))

    def gradient_n_I(self, input_tensor, direction):
        if direction == "x":
            kernel = self.weight_right
        elif direction == "y":
            kernel = self.weight_down
        gradient_orig1 = F.conv2d(input_tensor, kernel, padding="same")

        gradient_orig_abs = torch.abs(gradient_orig1)

        grad_min1 = torch.min(torch.min(gradient_orig_abs, 2, keepdim=True), 3, keepdim=True)
        grad_max1 = torch.max(torch.max(gradient_orig_abs, 2, keepdim=True), 3, keepdim=True)
        grad_norm1 = torch.div((gradient_orig_abs - (grad_min1)), (grad_max1 - grad_min1 + 0.0001))

        gradient_orig = torch.abs(
            F.avg_pool2d(gradient_orig1, 5, stride=1, padding=2, count_include_pad=False))  # denoise

        gradient_orig_patch_max = F.max_pool2d(gradient_orig, 7, stride=1, padding=3)
        gradient_orig_patch_min = torch.abs(1 - F.max_pool2d(1 - gradient_orig, 7, stride=1, padding=3))
        gradient_orig_patch_max = F.avg_pool2d(gradient_orig_patch_max, 7, stride=1, padding=3, count_include_pad=False)
        gradient_orig_patch_min = F.avg_pool2d(gradient_orig_patch_min, 7, stride=1, padding=3, count_include_pad=False)

        grad_norm = torch.div((gradient_orig - gradient_orig_patch_min),
                              (gradient_orig_patch_max - gradient_orig_patch_min + 0.0001))
        return (grad_norm + 0.01).detach() * grad_norm1

    def gradient_n_diff(self, input_tensor, direction):
        if direction == "x":
            kernel = self.weight_right
        elif direction == "y":
            kernel = self.weight_down
        gradient_orig1 = F.conv2d(input_tensor, kernel, padding="same")
        gradient_orig1_abs = torch.abs(gradient_orig1)

        grad_min1 = torch.min(torch.min(gradient_orig1_abs, 2, keepdim=True), 3, keepdim=True)
        grad_max1 = torch.max(torch.max(gradient_orig1_abs, 2, keepdim=True), 3, keepdim=True)
        grad_norm1 = torch.div((gradient_orig1_abs - (grad_min1)), (grad_max1 - grad_min1 + 0.0001))

        input_tensor = F.avg_pool2d(input_tensor, [5, 5], stride=1, padding=2, count_include_pad=False)  # denoise
        gradient_orig = torch.abs(F.conv2d(input_tensor, kernel, padding='same'))

        gradient_orig_patch_max = F.max_pool2d(gradient_orig, [7, 7], stride=1, padding=3)
        gradient_orig_patch_min = torch.abs(1 - F.max_pool2d(1 - gradient_orig, [7, 7], stride=1, padding=3))

        gradient_orig_patch_max = F.avg_pool2d(gradient_orig_patch_max, [7, 7], stride=1, padding=3,
                                               count_include_pad=False)
        gradient_orig_patch_min = F.avg_pool2d(gradient_orig_patch_min, [7, 7], stride=1, padding=3,
                                               count_include_pad=False)

        grad_norm = torch.div((gradient_orig - gradient_orig_patch_min),
                              (gradient_orig_patch_max - gradient_orig_patch_min + 0.0001))
        # return tf.stop_gradient(grad_norm+0.05)*grad_norm1
        return (grad_norm + 0.05).detach(), grad_norm1

    def forward(self, R_low, high):
        return self.diff_zy(R_low, high)


class L_smooth4(nn.Module):

    def __init__(self):
        super(L_smooth4, self).__init__()
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)

    def gradient_n_p(self, input_tensor, direction):
        if direction == "x":
            kernel = self.weight_right
        elif direction == "y":
            kernel = self.weight_down
        gradient_orig1 = F.conv2d(input_tensor, kernel, padding="same")
        gradient_orig_abs = torch.abs(gradient_orig1)

        gradient_orig_patch_max1 = F.max_pool2d(gradient_orig_abs, kernel_size=9, stride=1, padding=4)
        gradient_orig_patch_min1 = torch.abs(
            1 - F.max_pool2d(1 - gradient_orig_abs, kernel_size=9, stride=1, padding=4))
        grad_max1 = F.avg_pool2d(gradient_orig_patch_max1, kernel_size=17, stride=1, padding=8, count_include_pad=False)
        grad_min1 = F.avg_pool2d(gradient_orig_patch_min1, kernel_size=17, stride=1, padding=8, count_include_pad=False)
        grad_norm1 = torch.div((gradient_orig_abs - (grad_min1).detach()),
                               torch.abs(grad_max1.detach() - grad_min1.detach()) + 0.0001)

        input_tensor2 = F.avg_pool2d(input_tensor, kernel_size=5, stride=1, padding=2, count_include_pad=False)

        gradient_orig = torch.abs(F.conv2d(input_tensor2, kernel, padding="same"))

        gradient_orig_patch_max = F.max_pool2d(gradient_orig, 7, stride=1, padding=3)
        gradient_orig_patch_min = torch.abs(1 - F.max_pool2d(1 - gradient_orig, 7, stride=1, padding=3))
        gradient_orig_patch_max = F.avg_pool2d(gradient_orig_patch_max, 7, stride=1, padding=3, count_include_pad=False)
        gradient_orig_patch_min = F.avg_pool2d(gradient_orig_patch_min, 7, stride=1, padding=3, count_include_pad=False)

        grad_norm = torch.div((gradient_orig - gradient_orig_patch_min),
                              (torch.abs(gradient_orig_patch_max - gradient_orig_patch_min) + 0.0001))

        return (grad_norm + 0.01).detach() * grad_norm1

    def forward(self, R_low):
        B, C, H, W = R_low.shape
        if C == 3:
            # R_low = torch.mean(R_low,1,keepdim=True)
            R = R_low[:, 0:1, :, :]
            G = R_low[:, 1:2, :, :]
            B = R_low[:, 2:3, :, :]
            R_low = 0.299 * R + 0.587 * G + 0.114 * B
        else:
            R_low = R_low
        R_low_x = self.gradient_n_p(R_low, "x")
        R_low_y = self.gradient_n_p(R_low, "y")

        return torch.mean(R_low_x * torch.exp(-10 * R_low_x)) + torch.mean(
            R_low_y * torch.exp(-10 * R_low_y))  # +another_


class L_smooth_ill(nn.Module):

    def __init__(self):
        super(L_smooth_ill, self).__init__()
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)

    def gradient_n_p(self, input_tensor, direction):
        if direction == "x":
            kernel = self.weight_right
        elif direction == "y":
            kernel = self.weight_down
        gradient_orig1 = F.conv2d(input_tensor, kernel, padding="same")
        gradient_orig_abs = torch.abs(gradient_orig1)

        gradient_orig_patch_max1 = F.max_pool2d(gradient_orig_abs, kernel_size=9, stride=1, padding=4)
        gradient_orig_patch_min1 = torch.abs(
            1 - F.max_pool2d(1 - gradient_orig_abs, kernel_size=9, stride=1, padding=4))
        grad_max1 = F.avg_pool2d(gradient_orig_patch_max1, kernel_size=17, stride=1, padding=8, count_include_pad=False)
        grad_min1 = F.avg_pool2d(gradient_orig_patch_min1, kernel_size=17, stride=1, padding=8, count_include_pad=False)
        grad_norm1 = torch.div((gradient_orig_abs - (grad_min1).detach()),
                               torch.abs(grad_max1.detach() - grad_min1.detach()) + 0.0001)

        gradient_orig = F.conv2d(input_tensor, kernel, padding="same")
        gradient_orig = torch.abs(
            F.avg_pool2d(gradient_orig, kernel_size=5, stride=1, padding=2, count_include_pad=False))

        gradient_orig_patch_max = F.max_pool2d(gradient_orig, 7, stride=1, padding=3)
        gradient_orig_patch_min = torch.abs(1 - F.max_pool2d(1 - gradient_orig, 7, stride=1, padding=3))
        gradient_orig_patch_max = F.avg_pool2d(gradient_orig_patch_max, 7, stride=1, padding=3, count_include_pad=False)
        gradient_orig_patch_min = F.avg_pool2d(gradient_orig_patch_min, 7, stride=1, padding=3, count_include_pad=False)

        grad_norm = torch.div((gradient_orig - gradient_orig_patch_min),
                              (torch.abs(gradient_orig_patch_max - gradient_orig_patch_min) + 0.0001))

        return (grad_norm + 0.01).detach() * grad_norm1

    def forward(self, R_low, low):
        B, C, H, W = R_low.shape
        if C == 3:
            # R_low = torch.mean(R_low,1,keepdim=True)
            R = R_low[:, 0:1, :, :]
            G = R_low[:, 1:2, :, :]
            B = R_low[:, 2:3, :, :]
            R_low = 0.299 * R + 0.587 * G + 0.114 * B
            R = low[:, 0:1, :, :]
            G = low[:, 1:2, :, :]
            B = low[:, 2:3, :, :]
            low = 0.299 * R + 0.587 * G + 0.114 * B

        else:
            R_low = R_low
            low = low
        R_low_x = self.gradient_n_p(R_low, "x")
        R_low_y = self.gradient_n_p(R_low, "y")
        low_x = self.gradient_n_p(low, "x")
        low_y = self.gradient_n_p(low, "y")

        return torch.mean(R_low_x * torch.exp(-10 * R_low_x) * torch.exp(-10 * low_x)) + torch.mean(
            R_low_y * torch.exp(-10 * R_low_y) * torch.exp(-10 * low_y))  # +another_
        return torch.mean(R_low_x * torch.exp(-10 * R_low_x)) + torch.mean(
            R_low_y * torch.exp(-10 * R_low_y))  # +another_



class L_spa(nn.Module):

    def __init__(self):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enhance):
        b, c, h, w = org.shape

        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)

        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        weight_diff = torch.max(
            torch.FloatTensor([1]).cuda() + 10000 * torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),
                                                              torch.FloatTensor([0]).cuda()),
            torch.FloatTensor([0.5]).cuda())
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()), enhance_pool - org_pool)

        D_org_letf = F.conv2d(org_pool, self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool, self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf, 2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)
        E = (D_left + D_right + D_up + D_down)
        # E = 25*(D_left + D_right + D_up +D_down)

        return E


class L_exp(nn.Module):

    def __init__(self, patch_size):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        # self.mean_val = mean_val

    def forward(self, x, mean_val):
        b, c, h, w = x.shape
        x = torch.max(x, 1, keepdim=True)[0]
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean - torch.FloatTensor([mean_val]).cuda(), 2))
        return d


class L_TV(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class Sa_Loss(nn.Module):
    def __init__(self):
        super(Sa_Loss, self).__init__()
        # print(1)

    def forward(self, x):
        # self.grad = np.ones(x.shape,dtype=np.float32)
        b, c, h, w = x.shape
        # x_de = x.cpu().detach().numpy()
        r, g, b = torch.split(x, 1, dim=1)
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Dr = r - mr
        Dg = g - mg
        Db = b - mb
        k = torch.pow(torch.pow(Dr, 2) + torch.pow(Db, 2) + torch.pow(Dg, 2), 0.5)
        # print(k)

        k = torch.mean(k)
        return k


class Sa_Loss(nn.Module):
    def __init__(self):
        super(Sa_Loss, self).__init__()
        # print(1)

    def forward(self, x):
        # self.grad = np.ones(x.shape,dtype=np.float32)
        b, c, h, w = x.shape
        # x_de = x.cpu().detach().numpy()
        r, g, b = torch.split(x, 1, dim=1)
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Dr = r - mr
        Dg = g - mg
        Db = b - mb
        k = torch.pow(torch.pow(Dr, 2) + torch.pow(Db, 2) + torch.pow(Dg, 2), 0.5)
        # print(k)

        k = torch.mean(k)
        return k


class perception_loss(nn.Module):
    def __init__(self):
        super(perception_loss, self).__init__()
        # vgg = vgg16(pretrained=True).cuda()
        features = vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        # out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return h_relu_4_3


class PSNR(nn.Module):
    def __init__(self, max_val=0):
        super().__init__()

        base10 = torch.log(torch.tensor(10.0))
        max_val = torch.tensor(max_val).float()

        self.register_buffer('base10', base10)
        self.register_buffer('max_val', 20 * torch.log(max_val) / base10)

    def __call__(self, a, b):
        mse = torch.mean((a.float() - b.float()) ** 2)

        if mse == 0:
            return 0

        return 10 * torch.log10((1.0 / mse))

###############################################################################################
# import lpips
# # 加载预训练的LPIPS模型
# lpips_model = lpips.LPIPS(net="alex").to('cuda:1')
# def cal_lpips(img1,img2):
#     distance = lpips_model(img1, img2)
#     return distance

##################################################################################
def validation(model, val_loader):
    psnr = PSNR()
    ssim_list = []
    psnr_list = []
    lpips_list=[]
    for i, imgs in enumerate(val_loader):
        with torch.no_grad():
            low_img, high_img, sktch = imgs[0].to('cuda:1'), imgs[1].to('cuda:1'), imgs[2].to('cuda:1')
            # res_x, res_out1, res_out2, enhanced_image1, enhanced_image2, enhanced_image3= model(low_img)
            res_x,res_out1,enhanced_image2,enhanced_image3= model(low_img)
            enhanced_image=enhanced_image3
            # print(enhanced_img.shape)
        # ssim_value = ssim(enhanced_image, high_img, as_loss=False).item()
        ssim_value = ssim(enhanced_image, high_img).item()
        psnr_value = psnr(enhanced_image, high_img).item()
        # lpips_value = cal_lpips(enhanced_image,high_img).item()
        # print('The %d image SSIM value is %d:' %(i, ssim_value))
        ssim_list.append(ssim_value)
        psnr_list.append(psnr_value)
        # lpips_list.append(lpips_value)

    SSIM_mean = np.mean(ssim_list)
    PSNR_mean = np.mean(psnr_list)
    lpips_mean = np.mean(lpips_list)
    print('The SSIM Value is:', SSIM_mean)
    print('The PSNR Value is:', PSNR_mean)
    # print('The Lpips Value is:', lpips_mean)
    return  PSNR_mean,SSIM_mean

#################################################################################################


def tv_loss(x, beta=0.5, reg_coeff=5):
    '''Calculates TV loss for an image `x`.

    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta`
    '''
    dh = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2)
    dw = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2)
    a, b, c, d = x.shape
    return reg_coeff * (torch.sum(torch.pow(dh[:, :, :-1] + dw[:, :, :, :-1], beta)) / (a * b * c * d))


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

