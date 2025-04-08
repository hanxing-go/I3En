import argparse


parser = argparse.ArgumentParser("Fisrt_work")
parser.add_argument('--cuda', default=True, type=bool, help='Use CUDA to train model')
parser.add_argument('--gpu', type=str, default='1', help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--stage', type=int, default=3, help='epochs')
parser.add_argument('--save', type=str, default='EXP/', help='location of the data corpus')
#################################################################################################
parser.add_argument('--lowlight_images_path', type=str, default="data/Train/low/",help='训练低光图像路径')
parser.add_argument('--highlight_images_path', type=str, default="data/Train/high/",help='训练正常图像路径')
parser.add_argument('--val_lowlight_images_path', type=str, default="data/Test/low/",help='验证低光图像路径')
parser.add_argument('--val_highlight_images_path', type=str, default="data/Test/high/",help='验证低光图像路劲')

parser.add_argument('--task', type=str, default="train")
parser.add_argument('--nbins', type=int, default=14,help='bins的数量')
parser.add_argument('--patch_size', type=int, default=256,help='图像的尺寸')
#根据这个来修改图像尺寸
parser.add_argument('--exp_mean', type=float, default=0.55)
parser.add_argument('--sample_dir', type=str, default="./sample/")

parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--grad_clip_norm', type=float, default=0.1)
parser.add_argument('--num_epochs', type=int, default=30000)
# parser.add_argument('--train_batch_size', type=int, default=16)
# parser.add_argument('--val_batch_size', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--display_iter', type=int, default=2)
parser.add_argument('--snapshot_iter', type=int, default=10)
parser.add_argument('--scale_factor', type=int, default=16)
parser.add_argument('--snapshots_folder', type=str, default="snapshots_My_net/")
parser.add_argument('--load_pretrain', type=bool, default=False)
parser.add_argument('--pretrain_dir', type=str, default="snapshots_My_net/best_Epoch.pth")
###################################################################################################
parser.add_argument('--debug', action='store_true',help='Enables debug mode')
parser.add_argument('--template', default='.',help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=0,help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',help='use cpu only')
# parser.add_argument('--n_GPUs', type=int, default=4,
#                     help='number of GPUs')
parser.add_argument('--n_GPUs', type=int, default=1,help='number of GPUs')
# Data specifications
parser.add_argument('--dir_data', type=str, default='../data',help='dataset directory')
parser.add_argument('--dir_demo', type=str, default='../test',help='demo image directory')
parser.add_argument('--data_train', type=str, default='RainHeavy', #'DIV2K',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default= 'RainHeavyTest', #'DIV2K',
                    help='test dataset name')
parser.add_argument('--data_range', type=str, default='1-200/1-100',help='train/test data range')
parser.add_argument('--ext', type=str, default='sep',help='dataset file extension')
parser.add_argument('--scale', type=str, default='2',help='super resolution scale')
parser.add_argument('--rgb_range', type=int, default=255,help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,help='number of color channels to use')
parser.add_argument('--chop', action='store_true',help='enable memory-efficient forward')
parser.add_argument('--no_augment', action='store_true',help='do not use data augmentation')
'''
# Model specifications
 parser.add_argument('--model', default='spdnet',
                     help='model name')
'''
parser.add_argument('--model', default='spdnet_level',help='model name')
parser.add_argument('--act', type=str, default='relu',help='activation function')
parser.add_argument('--pre_train', type=str, default='.',help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=3,help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=32,help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,help='residual scaling')
parser.add_argument('--shift_mean', default=True,help='subtract pixel mean from the input')
parser.add_argument('--dilation', action='store_true',help='use dilated convolution')
parser.add_argument('--precision', type=str, default='single',choices=('single', 'half'),help='FP precision for test (single | half)')

# Training specifications
parser.add_argument('--test_every', type=int, default=1000,help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=1000,help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=2,help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true',help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true',help='set this option to test the model')
parser.add_argument('--reset', action='store_true',help='reset the training')
# Optimization specifications
parser.add_argument('--lr_decay', type=int, default=25,help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step_100_150_200_230_260_280_300',#100_115_130_140_150_158_165_170_175_180
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',choices=('SGD', 'ADAM', 'RMSprop'),help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,help='SGD momentum')
parser.add_argument('--beta1', type=float, default=0.9,help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,help='ADAM epsilon for numerical stability')

# Loss specifications
parser.add_argument('--loss', type=str, default='1*MSE',help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e6',help='skipping batch that has large error')

# Log specifications
parser.add_argument('--load', type=str, default='.',help='file name to load')
parser.add_argument('--resume', type=int, default=0,help='resume from specific checkpoint')
parser.add_argument('--save_models', action='store_true',help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=100,help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true',help='save output results')

args = parser.parse_args()