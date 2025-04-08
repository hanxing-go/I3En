import options
from PIL import Image
import numpy as np
from datasets.LOL_synthetic import ImagesDataset2
from torch.utils.data import DataLoader
import sys
import torch
args=options.args


from configs import data_configs
dataset_args = data_configs.DATASETS['ours_encode']
transforms_dict = dataset_args['transforms'](args).get_transforms()
test_dataset = ImagesDataset2(source_root_pre=dataset_args['test_source_root'],
                              target_root_pre=dataset_args['test_target_root'],
                              source_transform=transforms_dict['transform_source'],
                              target_transform=transforms_dict['transform_test'],
                              opts=args, train=0)
test_queue = DataLoader(test_dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=4,
                        drop_last=True, generator=torch.Generator(device='cuda:0'))


def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')


def main():
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    # 加载模型
    import model
    from model import utility
    checkpoint = utility.checkpoint(args)
    my_model = model.Model(args, checkpoint)

    my_model.eval()
    with torch.no_grad():
        for iteration, (img_lowlight,img_highlight,sketch) in enumerate(test_queue):
            print(1)
            img_lowlight = img_lowlight.to('cuda:1')

            res_x,res_out1,res_out2,enhanced_image1,enhanced_image2,enhanced_image3 = my_model(img_lowlight,1)
            enhanced_image=enhanced_image3
            # 将增亮后的图像从Tensor转换为PIL Image
            enhanced_image = enhanced_image.squeeze().detach().cpu().numpy()
            enhanced_image = (enhanced_image * 255).clip(0, 255).astype('uint8')
            enhanced_image = Image.fromarray(enhanced_image)
            save_path='/mnt/jxsd_jaw/motongstudio/zx/unpair_data/DICM_enhance'
            # 保存增亮后的图像到本地文件
            image_name = 'enhanced_%d' % iteration
            u_name = '%s.png' % (image_name)
            u_path = save_path + '/' + u_name
            enhanced_image.save(u_path)



if __name__ == '__main__':
    main()
