import torch.nn as nn
import pywt
import pytorch_wavelets.dwt.lowlevel as lowlevel
import torch

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
##############################################################################
# 多层残差
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.LeakyReLU()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
class RB(nn.Module):
    def __init__(self, n_feats, nm='in'):
        super(RB, self).__init__()
        module_body = []
        for i in range(2):
            module_body.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=True))
            module_body.append(nn.LeakyReLU())
        self.module_body = nn.Sequential(*module_body)
        self.relu = nn.LeakyReLU()
        self.se = SELayer(n_feats, 1)

    def forward(self, x):
        res = self.module_body(x)
        res = self.se(res)
        res += x
        return res

class RIR(nn.Module):
    def __init__(self, n_feats, n_blocks, nm='in'):
        super(RIR, self).__init__()
        module_body = [
            RB(n_feats) for _ in range(n_blocks)
        ]
        module_body.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=True))
        self.module_body = nn.Sequential(*module_body)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        res = self.module_body(x)
        res += x
        return self.relu(res)
#################################################################################################################

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

########################################################################################################################
class DWTForward(nn.Module):
    """ Performs a 2d DWT Forward decomposition of an image

    Args:
        J (int): Number of levels of decomposition
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays, either (h0, h1) or (h0_col, h1_col, h0_row, h1_row)
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
        """
    def __init__(self, J=1, wave='db1', mode='zero'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0_col, h1_col = wave.dec_lo, wave.dec_hi
            h0_row, h1_row = h0_col, h1_col
        else:
            if len(wave) == 2:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = h0_col, h1_col
            elif len(wave) == 4:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = wave[2], wave[3]

        # Prepare the filters
        filts = lowlevel.prep_filt_afb2d(h0_col, h1_col, h0_row, h1_row)
        self.register_buffer('h0_col', filts[0])
        self.register_buffer('h1_col', filts[1])
        self.register_buffer('h0_row', filts[2])
        self.register_buffer('h1_row', filts[3])
        self.J = J
        self.mode = mode

    def forward(self, x):
        """ Forward pass of the DWT.

        Args:
            x (tensor): Input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Returns:
            (yl, yh)
                tuple of lowpass (yl) and bandpass (yh) coefficients.
                yh is a list of length J with the first entry
                being the finest scale coefficients. yl has shape
                :math:`(N, C_{in}, H_{in}', W_{in}')` and yh has shape
                :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. The new
                dimension in yh iterates over the LH, HL and HH coefficients.

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        """
        yh = []
        ll = x
        mode = lowlevel.mode_to_int(self.mode)

        # Do a multilevel transform
        for j in range(self.J):
            # Do 1 level of the transform
            ll, high = lowlevel.AFB2D.apply(
                ll, self.h0_col, self.h1_col, self.h0_row, self.h1_row, mode)
            yh.append(high)

        return ll, yh


class DWTInverse(nn.Module):
    """ Performs a 2d DWT Inverse reconstruction of an image

    Args:
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays, either (h0, h1) or (h0_col, h1_col, h0_row, h1_row)
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
    """
    def __init__(self, wave='db1', mode='zero'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            g0_col, g1_col = wave.rec_lo, wave.rec_hi
            g0_row, g1_row = g0_col, g1_col
        else:
            if len(wave) == 2:
                g0_col, g1_col = wave[0], wave[1]
                g0_row, g1_row = g0_col, g1_col
            elif len(wave) == 4:
                g0_col, g1_col = wave[0], wave[1]
                g0_row, g1_row = wave[2], wave[3]
        # Prepare the filters
        filts = lowlevel.prep_filt_sfb2d(g0_col, g1_col, g0_row, g1_row)
        self.register_buffer('g0_col', filts[0])
        self.register_buffer('g1_col', filts[1])
        self.register_buffer('g0_row', filts[2])
        self.register_buffer('g1_row', filts[3])
        self.mode = mode

    def forward(self, coeffs):
        """
        Args:
            coeffs (yl, yh): tuple of lowpass and bandpass coefficients, where:
              yl is a lowpass tensor of shape :math:`(N, C_{in}, H_{in}',
              W_{in}')` and yh is a list of bandpass tensors of shape
              :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. I.e. should match
              the format returned by DWTForward

        Returns:
            Reconstructed input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.

        Note:
            Can have None for any of the highpass scales and will treat the
            values as zeros (not in an efficient way though).
        """
        yl, yh = coeffs
        ll = yl
        mode = lowlevel.mode_to_int(self.mode)

        # Do a multilevel inverse transform
        for h in yh[::-1]:
            if h is None:
                h = torch.zeros(ll.shape[0], ll.shape[1], 3, ll.shape[-2],
                                ll.shape[-1], device=ll.device)

            # 'Unpad' added dimensions
            if ll.shape[-2] > h.shape[-2]:
                ll = ll[...,:-1,:]
            if ll.shape[-1] > h.shape[-1]:
                ll = ll[...,:-1]
            ll = lowlevel.SFB2D.apply(
                ll, h, self.g0_col, self.g1_col, self.g0_row, self.g1_row, mode)
        return ll

from numpy.lib.function_base import hamming

DWT = DWTForward(J=1, mode='zero', wave='db1')
IDWT=DWTInverse(wave='db1')

class Level3Waveblock(nn.Module):
    def __init__(self, *, mult=2, in_channel=32, out_channel=32, dropout=0.5, ):
        super().__init__()

        self.DWT_conv1 = convd(in_channel*4, out_channel, 3, 1)
        self.DWT_conv2 = convd(in_channel*4, out_channel, 3, 1)
        self.DWT_conv3 = convd(in_channel*4, out_channel, 3, 1)

        self.IDWT_conv3 = convd(out_channel, out_channel*4, 3, 1)
        self.IDWT_conv2 = convd(out_channel, out_channel*4, 3, 1)
        self.IDWT_conv1 = convd(out_channel, out_channel*4, 3, 1)

        self.IMF=Prior_Sp()
        self.fuse_res = convd(in_channel*2, out_channel, 3, 1)

        self.feedforward1 = nn.Sequential(
            RIR(n_feats=out_channel, n_blocks=2),
            RIR(n_feats=out_channel, n_blocks=2),
            RIR(n_feats=out_channel , n_blocks=2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.feedforward2 = nn.Sequential(
            RIR(n_feats=out_channel, n_blocks=2),
            RIR(n_feats=out_channel, n_blocks=2),
            RIR(n_feats=out_channel , n_blocks=2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.feedforward3 = nn.Sequential(
            RIR(n_feats=out_channel,n_blocks=2),
            RIR(n_feats=out_channel,n_blocks=2),
            RIR(n_feats=out_channel,n_blocks=2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.reduction = nn.Conv2d(out_channel, out_channel, 1)


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

        x1 = self.reduction(x)
        y1, yh1 = DWT(x1)
        DWT1 = self._transformer(y1, yh1)
        DWT1=self.DWT_conv1(DWT1)

        y2, yh2 = DWT(DWT1)
        DWT2 = self._transformer(y2, yh2)
        DWT2=self.DWT_conv2(DWT2)

        y3, yh3 = DWT(DWT2)
        DWT3 = self._transformer(y3, yh3)
        DWT3=self.DWT_conv3(DWT3)

        #最底层
        x3 = self.IDWT_conv3(self.feedforward3(DWT3))
        x3 = self._Itransformer(x3)
        x3=IDWT(x3)+DWT2 # 进行逆小波转换，并进行残差连接

        #中间层
        x2 = torch.cat((DWT2, x3), dim=1)
        # x2=self.fuse_res(self.IMF(DWT2,x3))#特征融合
        x2 = self.IDWT_conv2(self.feedforward2(x2))
        x2 = self._Itransformer(x2)
        x2=IDWT(x2)+DWT1 #进行逆小波转换,并进行残差连接


        #最高层
        # x1 = torch.cat((DWT1, x2), dim=1)
        x1=self.fuse_res(self.IMF(DWT1, x2))#特征融合
        x1 = self.IDWT_conv1(self.feedforward1(x1))
        x1 = self._Itransformer(x1)
        x1=IDWT(x1)+x #进行逆小波转换，并进行残差连接

        return x1