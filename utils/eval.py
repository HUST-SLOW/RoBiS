import torch
import numpy as np
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score,  precision_recall_curve, average_precision_score
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
from tqdm import tqdm

from pathlib import Path
from PIL import Image
from collections import defaultdict
from utils.utils import get_gaussian_kernel


def _metrics(binary_root, gt_root, device, test_type = 'test_public', item_list = None, img_size=None, max_ratio = 0.1, gt_name_fn=lambda name: name.replace('.png', '_mask.png')):
    device = torch.device("cuda:{}".format(device) if torch.cuda.is_available() else "cpu")
    kernel = get_gaussian_kernel(5, 4).to(device)
    root_path = Path(binary_root)
    if item_list == ['all']:
        categories = ['sheet_metal', 'vial', 'wallplugs', 'walnuts', 'can', 'fabric', 'fruit_jelly', 'rice']
    else:
        categories = item_list
    print(categories)
    cat_gt_px = defaultdict(list)
    cat_pr_px = defaultdict(list)
    cat_gt_sp = defaultdict(list)
    cat_pr_sp = defaultdict(list)

    for cat in categories:
        bin_dir = root_path / cat / test_type
        gt_dir  = Path(gt_root) / cat / test_type / 'ground_truth'

        bin_paths = sorted(bin_dir.rglob('*.png')) + sorted(bin_dir.rglob('*.jpg'))
        if not bin_paths:
            continue

        for bin_path in tqdm(bin_paths, ncols=80, desc=f"{cat}-{test_type}"):
            defect_dir = bin_path.parent.name  # good or bad
            is_normal = (defect_dir == 'good')

            pr_np = (np.array(Image.open(bin_path).convert('L')) > 0).astype(np.float32)

            if is_normal:
                h, w = pr_np.shape
                gt_np = np.zeros((h, w), dtype=np.float32)
            else:
                gt_name = gt_name_fn(bin_path.name)
                gt_path = gt_dir / 'bad' / gt_name
                if not gt_path.exists():
                    continue
                gt_np = (np.array(Image.open(gt_path).convert('L')) > 0).astype(np.float32)

            if img_size is not None:
                if isinstance(img_size, int):
                    img_size = (img_size, img_size)
                gt_np = np.array(Image.fromarray(gt_np).resize(img_size[::-1], Image.NEAREST))
                pr_np = np.array(Image.fromarray(pr_np).resize(img_size[::-1], Image.BILINEAR))

            gt_t = torch.from_numpy(gt_np)[None, None].to(device)
            pr_t = torch.from_numpy(pr_np)[None, None].to(device)
            pr_t = F.conv2d(pr_t, kernel.weight, padding=2)

            cat_gt_px[cat].append(gt_t)
            cat_pr_px[cat].append(pr_t)
            label = 0.0 if is_normal else 1.0
            cat_gt_sp[cat].append(label)

            vec = pr_t.flatten(1)
            if max_ratio == 0:
                score = vec.max(dim=1)[0].item()
            else:
                k = int(vec.numel() * max_ratio)
                score = torch.topk(vec, k, dim=1)[0].mean().item()
            cat_pr_sp[cat].append(score)

    for cat in categories:
        if cat not in cat_gt_px or len(np.unique(cat_gt_sp[cat])) < 2:
            print(f'{cat:>12}: [0.0000] * 7')
            continue

        gpx = torch.cat(cat_gt_px[cat], 0)[:, 0].cpu().numpy()
        ppx = torch.cat(cat_pr_px[cat], 0)[:, 0].cpu().numpy()
        gsp = np.array(cat_gt_sp[cat])
        psp = np.array(cat_pr_sp[cat])
        ppx = (ppx - ppx.min()) / (ppx.max() - ppx.min() + 1e-8)

        auroc_sp = roc_auc_score(gsp, psp)
        ap_sp    = average_precision_score(gsp, psp)
        prec, rec, _ = precision_recall_curve(gsp, psp)
        f1_sp = np.max((2 * prec * rec) / (prec + rec + 1e-8))

        auroc_px = roc_auc_score(gpx.ravel(), ppx.ravel())
        ap_px    = average_precision_score(gpx.ravel(), ppx.ravel())
        prec, rec, _ = precision_recall_curve(gpx.ravel(), ppx.ravel())
        f1_px = np.max((2 * prec * rec) / (prec + rec + 1e-8))
        aupro_px = compute_pro(gpx.astype(bool), ppx, num_th=200)

        print('{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, '
              'P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                  cat, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))

def compute_pro(masks: np.ndarray, amaps: np.ndarray, num_th: int = 200) -> float:
    assert masks.ndim == 3 and amaps.ndim == 3 and masks.shape == amaps.shape
    assert set(masks.flatten()) <= {0, 1}

    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / num_th

    rows = []
    for th in np.arange(min_th, max_th + 1e-8, delta):
        bin_amaps = amaps > th
        pros, fps = [], 0
        for bm, m in zip(bin_amaps, masks):
            for reg in measure.regionprops(measure.label(m)):
                pros.append(bm[reg.coords[:, 0], reg.coords[:, 1]].sum() / reg.area)
            fps += np.logical_and(1 - m, bm).sum()
        fpr = fps / (1 - masks).sum()
        rows.append({'pro': np.mean(pros), 'fpr': fpr})

    df = pd.DataFrame(rows)
    df = df[df['fpr'] < 0.3]
    df['fpr'] /= df['fpr'].max()
    return auc(df['fpr'], df['pro'])

def metrics(cfg: dict):
    _metrics(
        binary_root=cfg['save']['bin_savedir'],
        gt_root=cfg['dataset']['original_data_path'],
        test_type=cfg['dataset']['test_type'],
        item_list=cfg['dataset']['mvtecad2_class_list'],
        img_size=cfg['model']['input_size'],
        max_ratio=0.1,
        device=cfg['device']
    )