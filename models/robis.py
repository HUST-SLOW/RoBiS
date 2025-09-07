import torch
import torch.nn as nn
import numpy as np
import os
from functools import partial
import warnings
from tqdm import tqdm
from torch.nn.init import trunc_normal_
import tifffile as tiff
from torch.nn import functional as F
import sys
sys.path.append(os.getcwd())

from models.optimizers import StableAdamW
from models.loss.loss import global_cosine_hm_adaptive
from models.scheduler.scheduler import WarmCosineScheduler
from utils.utils import setup_seed, get_gaussian_kernel

# Dataset-Related Modules
from datasets.mvtec_ad_2 import MVTec2Dataset, get_data_transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Model-Related Modules
from models.INP_models import vit_encoder
from models.INP_models.uad import INP_Former
from models.INP_models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block


class RoBiS():
    def __init__(self, cfg, seed=0):
        self.cfg = cfg
        self.seed = seed
        self.device = torch.device("cuda:{}".format(cfg['device']) if torch.cuda.is_available() else "cpu")

        self.dataset = cfg['dataset']['dataset_name']
        self.original_data_path = cfg['dataset']['original_data_path'] 
        self.data_path = cfg['dataset']['data_path']
        self.mvtecad2_class_list = cfg['dataset']['mvtecad2_class_list']
        if self.dataset == 'MVTec-AD-2':
            if self.mvtecad2_class_list == ['all']:
                self.item_list = ['sheet_metal', 'vial', 'wallplugs', 'walnuts', 'can', 'fabric', 'fruit_jelly', 'rice']
            else:
                self.item_list = self.mvtecad2_class_list
        print(self.item_list)
        self.test_type = cfg['dataset']['test_type']

        self.save_dir = cfg['save']['save_dir']
        self.amap_savedir = cfg['save']['amap_savedir']

        self.encoder = cfg['model']['encoder']
        self.input_size = cfg['model']['input_size']
        self.crop_size = cfg['model']['crop_size']
        self.INP_num = cfg['model']['INP_num']

        self.total_epochs = cfg['training']['total_epochs']
        self.batch_size = cfg['training']['batch_size']
        self.train_state = cfg['training']['train_state']
        self.test_state = cfg['testing']['test_state']

    def load_model(self):
        target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
        fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

        encoder = vit_encoder.load(self.encoder)
        if 'small' in self.encoder:
            embed_dim, num_heads = 384, 6
        elif 'base' in self.encoder:
            embed_dim, num_heads = 768, 12
        elif 'large' in self.encoder:
            embed_dim, num_heads = 1024, 16
            target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
        else:
            raise ValueError("Architecture not in small, base, large.")

        Bottleneck = nn.ModuleList([Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.)])
        INP = nn.ParameterList([nn.Parameter(torch.randn(self.INP_num, embed_dim))])
        INP_Extractor = nn.ModuleList([
            Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                              qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        ])
        INP_Guided_Decoder = nn.ModuleList([
            Prototype_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
            for _ in range(8)
        ])

        model = INP_Former(encoder=encoder, bottleneck=Bottleneck, aggregation=INP_Extractor, decoder=INP_Guided_Decoder,
                        target_layers=target_layers,  remove_class_token=True, fuse_layer_encoder=fuse_layer_encoder,
                        fuse_layer_decoder=fuse_layer_decoder, prototype_token=INP)
        model = model.to(self.device)

        return model, Bottleneck, INP_Guided_Decoder, INP_Extractor, INP

    def train(self):
        for item in self.item_list:
            setup_seed(1)
            if os.path.exists(os.path.join(self.save_dir, item, 'model.pth')):
                return None
            test_type = self.test_type
            data_transform, gt_transform = get_data_transforms(self.input_size, self.crop_size)
            data_train_transform, _ = get_data_transforms(self.input_size, self.crop_size, bright_aug=True)

            train_path = os.path.join(self.data_path, item, 'train')
            train_data = ImageFolder(root=train_path, transform=data_train_transform)
            train_dataloader = torch.utils.data.DataLoader(
                train_data, batch_size=self.batch_size, shuffle=True, num_workers=4, drop_last=True)

            model, Bottleneck, INP_Guided_Decoder, INP_Extractor, INP = self.load_model()

            trainable = nn.ModuleList([Bottleneck, INP_Guided_Decoder, INP_Extractor, INP])
            optimizer = StableAdamW([{'params': trainable.parameters()}],
                                lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
            lr_scheduler = WarmCosineScheduler(optimizer, base_value=1e-3, final_value=1e-4, total_iters=self.total_epochs*len(train_dataloader), warmup_iters=100)

            for m in trainable.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

            print('train image number:{}'.format(len(train_data)))
            for epoch in range(self.total_epochs):
                model.train()
                loss_list = []
                for img, _ in tqdm(train_dataloader, ncols=80):
                    img = img.to(self.device)
                    en, de, g_loss = model(img)
                    loss = global_cosine_hm_adaptive(en, de, y=3) + 0.2 * g_loss
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm(trainable.parameters(), max_norm=0.1)
                    optimizer.step()
                    loss_list.append(loss.item())
                    lr_scheduler.step()
                print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, self.total_epochs, np.mean(loss_list)))

            os.makedirs(os.path.join(self.save_dir, item), exist_ok=True)
            torch.save(model.state_dict(), os.path.join(self.save_dir, item, 'model.pth'))

    def test(self):
        setup_seed(1)
        gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(self.device)
        os.makedirs(self.amap_savedir, exist_ok=True)
        for item in self.item_list:
            test_type = self.test_type
            data_transform, gt_transform = get_data_transforms(self.input_size, self.crop_size)
            test_path = os.path.join(self.data_path, item)
            test_data = MVTec2Dataset(root=test_path, transform=data_transform,
                                    gt_transform=gt_transform, train_state=False, test_state=self.test_state, test_type=test_type)
            test_dataloader = torch.utils.data.DataLoader(
                test_data, batch_size=self.batch_size, shuffle=False, num_workers=4)
            model, *_ = self.load_model()
            model.load_state_dict(
                torch.load(os.path.join(self.save_dir, item, 'model.pth'), map_location='cuda:0'),
                strict=True)
            model.eval()
            save_dir = os.path.join(self.amap_savedir, item, test_type)
            img_path_list = []
            anomaly_maps = []

            with torch.no_grad():
                for img, _, _, img_path in tqdm(test_dataloader, ncols=80):
                    img = img.to(self.device)
                    output = model(img)
                    en, de = output[0], output[1]
                    out_size = img.shape[-1]
                    a_map_list = []
                    for fs, ft in zip(en, de):
                        sim = 1 - F.cosine_similarity(fs, ft).unsqueeze(1)
                        sim = F.interpolate(sim, size=(out_size, out_size),
                                            mode='bilinear', align_corners=True)
                        a_map_list.append(sim)
                    anomaly_map = torch.cat(a_map_list, dim=1).mean(dim=1, keepdim=True)
                    anomaly_map = F.interpolate(anomaly_map, size=256,
                                                mode='bilinear', align_corners=False)
                    anomaly_map = gaussian_kernel(anomaly_map)
                    anomaly_maps.append(anomaly_map.cpu())
                    img_path_list.extend(img_path)
            anomaly_maps = torch.cat(anomaly_maps, dim=0)[:, 0].numpy()
            anomaly_maps = (anomaly_maps - anomaly_maps.min()) / \
                        (anomaly_maps.max() - anomaly_maps.min() + 1e-8) * 255.0
            anomaly_maps = anomaly_maps.astype(np.float16)
            os.makedirs(save_dir, exist_ok=True)
            print('Saving anomaly maps to tiff...')
            for idx, path in enumerate(tqdm(img_path_list, ncols=80, desc='save')):
                if 'test_public' in path:
                    img_name = '/'.join(path.split('/')[-2:]).replace('.png', '.tiff')
                else:
                    img_name = os.path.basename(path).replace('.png', '.tiff')
                save_path = os.path.join(save_dir, img_name)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                tiff.imsave(save_path, anomaly_maps[idx])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    parser.add_argument("--original_data_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--mvtecad2_class_list", nargs="+", default=["all"])
    parser.add_argument("--test_type", default="test_public")
    parser.add_argument("--train_state", action="store_true", help="train")
    parser.add_argument("--test_state", action="store_true", help="test")
    parser.add_argument("--encoder", default="dinov2reg_vit_base_14")
    parser.add_argument("--input_size", type=int, default=518)
    parser.add_argument("--crop_size", type=int, default=518)
    parser.add_argument("--INP_num", type=int, default=6)
    parser.add_argument("--total_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_dir", default="./saved_weights")
    parser.add_argument("--amap_savedir", default="./anomaly_map_results")
    args = parser.parse_args()

    cfg = {
        "device": args.device,
        "dataset": {
            "dataset_name": "MVTec-AD-2",
            "original_data_path": args.original_data_path,
            "data_path": args.data_path,
            "mvtecad2_class_list": args.mvtecad2_class_list,
            "test_type": args.test_type
        },
        "model": {
            "encoder": args.encoder,
            "input_size": args.input_size,
            "crop_size": args.crop_size,
            "INP_num": args.INP_num
        },
        "training": {
            "total_epochs": args.total_epochs,
            "batch_size": args.batch_size,
            "train_state": args.train_state
        },
        "testing": {
            "test_state": args.test_state
        },
        "save": {
            "save_dir": args.save_dir,
            "amap_savedir": args.amap_savedir
        }
    }

    rob = RoBiS(cfg)
    if args.train_state:
        rob.train()
    if args.test_state:
        rob.test()