import os
import numpy as np
import torch
import shutil
from torch.nn.modules.container import T
import torchvision.transforms as transforms
from torch.autograd import Variable


class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
    # para = 0.0
    # for name, v in model.named_parameters():
    #     if v.requires_grad == True:
    #         if "auxiliary" not in name:
    #             para += np.prod(v.size())
    # return para / 1e6
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6



def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.makedirs(path,exist_ok=True)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.makedirs(os.path.join(path, 'scripts'),exist_ok=True)
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)



import numpy
import cv2

def numpy_log(x):
	return numpy.log(x + 1)
def mean_std_normalize(result, dynamic=2.0):
	mean = numpy.mean(result, axis=(0, 1))
	stdvar = numpy.sqrt(numpy.var(result, axis=(0, 1)))
	min_value = mean - dynamic * stdvar
	max_value = mean + dynamic * stdvar
	result = (result - min_value) / (max_value - min_value)
	result = 255 * numpy.clip(result, 0, 1)
	return result.astype("uint8")
def MSRCR(low_light_origin, sigmas=[15, 80, 200], weights=[0.33, 0.33, 0.34], alpha=128, dynamic=2.0):
    assert len(sigmas) == len(weights), "scales are not consistent !"
    weights = numpy.array(weights) / numpy.sum(weights)
    low_light_s= low_light_origin.cpu().detach().numpy().transpose(0, 2, 3, 1)
    output_array = np.zeros_like(low_light_s)
    for i in range(low_light_s.shape[0]):
        # 图像转成 float 处理
        low_light = low_light_s[i].astype("float32")
        # 转到 log 域
        log_I = numpy_log(low_light)
        # 每个尺度下做高斯模糊, 提取不同的平滑层, 作为光照图的估计
        log_Ls = [cv2.GaussianBlur(log_I, (0, 0), sig) for sig in sigmas]
        # 多个尺度的 MSR 叠加
        log_R = numpy.stack([weights[i] * (log_I - log_Ls[i]) for i in range(len(sigmas))])
        log_R = numpy.sum(log_R, axis=0)
        # 颜色恢复
        norm_sum = numpy_log(numpy.sum(low_light, axis=2))
        result = log_R * (numpy_log(alpha * low_light) - numpy.atleast_3d(norm_sum))
        result=mean_std_normalize(result, dynamic)
        output_array[i] = result
        # result = numpy.exp(result)
        # 标准化
    output_tensor = torch.from_numpy(output_array.transpose(0, 3, 1, 2)).to(low_light_origin.device)
    return output_tensor


if __name__ == '__main__':
    input = torch.randn([2,3,256,256])
    print(MSRCR(input).shape)

