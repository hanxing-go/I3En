import sys
import time
import glob
import numpy as np
import utils
from PIL import Image
import logging
import torch.utils
import torch.backends.cudnn as cudnn
from model import *
import multi_read_data as mrd
import options
from loss import validation
import torchvision
import torch.nn.functional as F


from configs import data_configs
from datasets.LOL_real import ImagesDataset2
from torch.utils.data import DataLoader
#导入参数
args=options.args

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"

args.save = args.save + '/' + 'Train-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
model_path = args.save + '/model_epochs/'
os.makedirs(model_path, exist_ok=True)
image_path = args.save + '/image_epochs/'
os.makedirs(image_path, exist_ok=True)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

logging.info("train file name = %s", os.path.split(__file__))

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return torch.utils.data.RandomSampler(dataset)

def cross_entropy_loss_RCF(prediction, labelf, beta):
    label = labelf.long()
    mask = labelf.clone()
    num_positive = torch.sum(label==1).float()
    num_negative = torch.sum(label==0).float()

    mask[label == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[label == 0] = beta * num_positive / (num_positive + num_negative)
    mask[label == 2] = 0
    cost = F.binary_cross_entropy(prediction, labelf, weight=mask, reduction='mean')
    return cost



def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %s' % args.gpu)
    logging.info("args = %s", args)

    #####################################################################################################################
    #加载模型
    import model
    from model import utility
    checkpoint = utility.checkpoint(args)
    my_model =model.Model(args,checkpoint)


    result=sum([p.numel() for p in my_model.parameters()])
    print(result)


    optimizer = torch.optim.Adam(my_model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=3e-4)


    MB = utils.count_parameters_in_MB(my_model)
    logging.info("model size = %f", MB)
    print(MB)
    #####################################################################################################################
    # 训练数据集
    dataset_args = data_configs.DATASETS['ours_encode']
    transforms_dict = dataset_args['transforms'](args).get_transforms()

    train_dataset = ImagesDataset2(source_root_pre=dataset_args['train_source_root'],
                                   target_root_pre=dataset_args['train_target_root'],
                                   source_transform=transforms_dict['transform_source'],
                                   target_transform=transforms_dict['transform_gt_train'],
                                   opts=args, train=1)
    test_dataset = ImagesDataset2(source_root_pre=dataset_args['test_source_root'],
                                  target_root_pre=dataset_args['test_target_root'],
                                  source_transform=transforms_dict['transform_source'],
                                  target_transform=transforms_dict['transform_test'],
                                  opts=args, train=0)
    train_queue = DataLoader(train_dataset,
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       num_workers=4,
                                       drop_last=True,generator=torch.Generator(device = 'cuda:0'))
    test_queue = DataLoader(test_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=4,
                                      drop_last=True,generator=torch.Generator(device = 'cuda:0'))

    ######################################################################################################################
    # #测试
    # cnt=1
    # for iteration, (img_lowlight, img_highlight, sketch) in enumerate(test_queue):
    #     print(cnt)
    #     cnt=cnt+1
    #     img_lowlight = img_lowlight.to('cuda:1')
    #
    #     res_x, res_out1, res_out2, enhanced_image1, enhanced_image2, enhanced_image3 = my_model(img_lowlight, 1)
    #     enhanced_image = enhanced_image3
    #     # 将增亮后的图像从Tensor转换为PIL Image
    #     enhanced_image = enhanced_image.squeeze().detach().cpu().numpy()
    #     enhanced_image = np.transpose(enhanced_image, (1, 2, 0))
    #     enhanced_image = (enhanced_image * 255).clip(0, 255).astype('uint8')
    #     enhanced_image = Image.fromarray(enhanced_image)
    #     save_path = '/mnt/jxsd_jaw/motongstudio/zx/unpair_data/DICM_enhance'
    #     # 保存增亮后的图像到本地文件
    #     image_name = 'enhanced_%d' % iteration
    #     u_name = '%s.png' % (image_name)
    #     u_path = save_path + '/' + u_name
    #     enhanced_image.save(u_path)
    #
    # print('完成')
    # sys.exit(1)
    #######################################################################################################################
    #损失函数
    # Loss function一共四个损失函数
    import loss
    L_grad_cosist = loss.L_grad_cosist().to('cuda:1')#表示梯度一致性损失函数
    L_bright_cosist = loss.L_bright_cosist().to('cuda:1')#表示亮度一致性损失函数
    L_recon = loss.L_recon().to('cuda:1')# L_recon表示重构损失函数
    L_color=loss.L_color().to('cuda:1')#颜色损失
    l_TVloss=loss.TVLoss().to('cuda:1')
    l_hist=loss.HistogramLoss().to(('cuda:1'))#直方图损失，衡量颜色差异
    # #################################################################################################################
    total_step = 0
    ssim_high=0
    psnr_high=0

    for epoch in range(args.epochs):
        print('===============================================================\nepoch:')
        print(epoch)
        print('Training')
        print('===============================================================')
        my_model.train()
        losses = []
        for iteration, (img_lowlight,img_highlight,sketch) in enumerate(train_queue):
            total_step += 1
            # img_lowlight_origin = img_lowlight_origin.to('cuda:1')
            img_lowlight = img_lowlight.to('cuda:1')
            img_highlight = img_highlight.to('cuda:1')
            sketch=sketch.to('cuda:1')

            res_x,res_out1,res_out2,enhanced_image1,enhanced_image2,enhanced_image3 = my_model(img_lowlight,1)
            Loss_grad_consist1 = L_grad_cosist(enhanced_image1, img_highlight)
            Loss_bright_consist1 = L_bright_cosist(enhanced_image1, img_highlight)
            L1_1, Loss_ssim1 = L_recon(enhanced_image1, img_highlight)
            Loss_color1 = torch.mean(L_color(enhanced_image1, img_highlight))#求出颜色损失
            L_TV1=l_TVloss(enhanced_image1)
            L_Hist1=l_hist(enhanced_image1,img_highlight)

            Loss_grad_consist2 = L_grad_cosist(enhanced_image2, img_highlight)
            Loss_bright_consist2 = L_bright_cosist(enhanced_image2, img_highlight)
            L1_2, Loss_ssim2 = L_recon(enhanced_image2, img_highlight)
            Loss_color2 = torch.mean(L_color(enhanced_image2, img_highlight))#求出颜色损失
            L_TV2=l_TVloss(enhanced_image2)
            L_Hist2=l_hist(enhanced_image2,img_highlight)

            #
            Loss_grad_consist3 = L_grad_cosist(enhanced_image3, img_highlight)
            Loss_bright_consist3 = L_bright_cosist(enhanced_image3, img_highlight)
            L1_3, Loss_ssim3 = L_recon(enhanced_image3, img_highlight)
            Loss_color3 = torch.mean(L_color(enhanced_image3, img_highlight))#求出颜色损失
            L_TV3=l_TVloss(enhanced_image3)
            L_Hist3=l_hist(enhanced_image2,img_highlight)

            #
            a=1
            b=1
            c=1
            Loss_grad_consist=a*Loss_grad_consist1+b*Loss_grad_consist2+c*Loss_grad_consist3
            Loss_bright_consist=a*Loss_bright_consist1+b*Loss_bright_consist2+c*Loss_bright_consist3
            L1=a*L1_1+b*L1_2+c*L1_3
            Loss_ssim=a*Loss_ssim1+b*Loss_ssim2+c*Loss_ssim3
            Loss_color=a*Loss_color1+b*Loss_color2+c*Loss_color3
            L_TV=a*L_TV1+b*L_TV2+c*L_TV3
            L_Hist=a*L_Hist1+b*L_Hist2+c*L_Hist3

            loss_supervised = Loss_grad_consist+Loss_bright_consist+L1+Loss_ssim+Loss_color+L_TV+L_Hist
            loss = loss_supervised
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(my_model.parameters(), 5)
            optimizer.step()
            # scheduler.step()  # 动态调整学习率
            #记录当前损失
            losses.append(loss.item())#将张量变成标量
            logging.info('train-epoch %03d %03d %f', epoch, iteration, loss)


        torchvision.utils.save_image(torch.cat([img_lowlight[0],enhanced_image1[0], enhanced_image2[0],enhanced_image3[0],img_highlight[0]], dim=2),
                                                 args.sample_dir + str(epoch) + '.png')
        torchvision.utils.save_image(torch.cat([res_x[0],res_out1[0], res_out2[0],sketch[0]], dim=2),
                                                 args.sample_dir + str(epoch) +'sketch'+ '.png')
        torch.save(my_model.state_dict(), args.snapshots_folder + "Epoch" + str(epoch) + '.pth')
        # print(my_model)
        my_model.eval()
        PSNR_mean, SSIM_mean = validation(my_model, test_queue)
        if SSIM_mean > ssim_high:
            ssim_high = SSIM_mean
            print('the highest SSIM value is:', str(ssim_high))
            torch.save(my_model.state_dict(), os.path.join(args.snapshots_folder, "best_ssim_Epoch" + '.pth'))
        with open(args.snapshots_folder + 'log.txt', 'a+') as f:
            f.write('epoch' + str(epoch) + ':' + 'the SSIM is' + str(SSIM_mean) + 'the PSNR is' + str(
                            PSNR_mean) + '\n')
        if PSNR_mean > psnr_high:
            psnr_high = PSNR_mean
            print('the highest PSNR value is:', str(psnr_high))
            torch.save(my_model.state_dict(), os.path.join(args.snapshots_folder, "best_psnr_Epoch" + '.pth'))
            with open(args.snapshots_folder + 'log.txt', 'a+') as f:
                f.write('epoch' + str(epoch) + ':' + 'the SSIM is' + str(SSIM_mean) + 'the PSNR is' + str(
                            PSNR_mean) + '\n')

        # 打印当前学习率
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        print('{0}{1}'.format('Loss_grad_consist:', Loss_grad_consist3.item()))
        print('{0}{1}'.format('Loss_bright_consit:', Loss_bright_consist3.item()))
        print('{0}{1}'.format('L1:', L1_3.item()))
        print('{0}{1}'.format('Loss_ssim:', Loss_ssim3))
        print('{0}{1}'.format('Loss_color', +Loss_color3.item()))
        print('{0}{1}'.format('Loss_Hist', +L_Hist.item()))

if __name__ == '__main__':
    main()
