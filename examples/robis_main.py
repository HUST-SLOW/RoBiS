import argparse
import os
import sys
sys.path.append(os.getcwd())
from models.robis import RoBiS
from models.modules._swin_cropping import crop
from models.modules._merging import merge
from models.modules._binarization import mvtec_bin
from models.modules._SAM_Finer import samfiner 

from utils.load_config import load_yaml
from utils.eval import metrics

import warnings
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser(description='RoBiS')
    parser.add_argument('--config', type=str, default='./configs/robis.yaml', help='config file path') 
    parser.add_argument('--dataset_name', type=str, default=None, help='dataset name')
    parser.add_argument('--original_data_path', type=str, default=None, help='original image dataset path') 
    parser.add_argument('--data_path', type=str, default=None, help='crop image dataset path') 
    parser.add_argument('--mvtecad2_class_list', type=str, nargs="+", default=None, help='class names')
    parser.add_argument('--test_type', type=str, default=None, help='test_public or test_private or test_private_mixed or validation')

    parser.add_argument('--window_size', type=int, default=None, help='crop window size')
    parser.add_argument('--desired_overlap', type=float, default=None, help='overlap')

    parser.add_argument('--save_dir', type=str, default=None, help='save weights path')
    parser.add_argument('--amap_savedir', type=str, default=None, help='output anomaly map path')
    parser.add_argument('--bin_savedir', type=str, default=None, help='output binary mask path')
    parser.add_argument('--encoder', type=str, default=None, help='backbone')
    parser.add_argument('--input_size', type=int, default=None, help='image size')
    parser.add_argument('--crop_size', type=int, default=None, help='crop size')
    parser.add_argument('--INP_num', type=int, default=None, help='the number of INPs') 
    parser.add_argument('--total_epochs', type=int, default=None, help='epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='batch size')
    parser.add_argument('--train_state', type=str, default=None, help='train')
    parser.add_argument('--test_state', type=str, default=None, help='test')
    parser.add_argument('--eval_pub', type=str, default=None, help='evaluate test_public or not')
    parser.add_argument('--device', type=int, default=None, help='gpu id')
    args = parser.parse_args()
    return args
    
def load_args(cfg, args):
    # If input new arguments through the script (robis.sh), the default configuration in the config file (robis.yaml) will be overwritten.
    if args.dataset_name is not None:
        cfg['dataset']['dataset_name'] = args.dataset_name
    if args.original_data_path is not None:
        cfg['dataset']['original_data_path'] = args.original_data_path
    assert os.path.exists(cfg['dataset']['original_data_path']), f"The dataset path {cfg['dataset']['original_data_path']} does not exist."
    if args.data_path is not None:
        cfg['dataset']['data_path'] = args.data_path
    if args.mvtecad2_class_list is not None:
        cfg['dataset']['mvtecad2_class_list'] = args.mvtecad2_class_list
    if args.test_type is not None:
        cfg['dataset']['test_type'] = args.test_type
    if args.window_size is not None:
        cfg['crop']['window_size'] = args.window_size
    if args.desired_overlap is not None:
        cfg['crop']['desired_overlap'] = args.desired_overlap

    if args.save_dir is not None:
        cfg['save']['save_dir'] = args.save_dir
    if args.amap_savedir is not None:
        cfg['save']['amap_savedir'] = args.amap_savedir
    if args.bin_savedir is not None:
        cfg['save']['bin_savedir'] = args.bin_savedir

    if args.encoder is not None:
        cfg['model']['encoder'] = args.encoder
    if args.input_size is not None:
        cfg['model']['input_size'] = args.input_size
    if args.crop_size is not None:
        cfg['model']['crop_size'] = args.crop_size
    if args.INP_num is not None:
        cfg['model']['INP_num'] = args.INP_num

    if args.total_epochs is not None:
        cfg['training']['total_epochs'] = args.total_epochs
    if args.batch_size is not None:
        cfg['training']['batch_size'] = args.batch_size
    if args.train_state is not None:
        if args.train_state.lower() == 'true':
            cfg['training']['train_state'] = True
        else:
            cfg['training']['train_state'] = False

    if args.test_state is not None:
        if args.test_state.lower() == 'true':
            cfg['testing']['test_state'] = True
        else:
            cfg['testing']['test_state'] = False
    if args.eval_pub is not None:
        if args.eval_pub.lower() == 'true':
            cfg['testing']['eval_pub'] = True
        else:
            cfg['testing']['eval_pub'] = False

    if args.device is not None:
        cfg['device'] = args.device
    if isinstance(cfg['device'], int):
        cfg['device'] = str(cfg['device'])
    return cfg

if __name__ == "__main__":
    args = get_args()
    cfg = load_yaml(args.config)
    cfg = load_args(cfg, args)
    os.makedirs(cfg['dataset']['data_path'], exist_ok=True)
    os.makedirs(cfg['save']['amap_savedir'], exist_ok=True)
    os.makedirs(cfg['save']['bin_savedir'], exist_ok=True)
    print(cfg)
    seed = 42
    print('Cropping images')
    crop(cfg)  # Cropping images.
    model = RoBiS(cfg, seed=seed)
    if cfg['training']['train_state']:
        print('Training model')
        model.train()  # Model training.
    if cfg['testing']['test_state']:
        print('Testing model')
        model.test()  # Model testing.
        print('Merging images')
        merge(cfg)  # Reconstruct images. 
        print('Binarizing anomaly maps')
        mvtec_bin(cfg)  # Binarize anomlay maps.
        print('Generating masks with SAM')
        samfiner(cfg)  # Using SAM to generate finer binary masks
        if cfg['testing']['eval_pub'] and cfg['dataset']['test_type']=='test_public':
            print('Evaluating')
            metrics(cfg)  # Evaluating test_public dataset.