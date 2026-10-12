import os
import torch
import argparse
import numpy as np
from scipy import misc
import imageio
import cv2
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from utils import ops
from utils.dataset import test_dataset as EvalDataset
from Net.ACUC_Net import MyNet as Network  
from utils.dataset import get_loader, test_dataset
import py_sod_metrics

def val(test_loader, model):
    
    global best_metric_dict, best_epoch, best_score
    
    mae = py_sod_metrics.MAE()
    fm = py_sod_metrics.Fmeasure()
    em = py_sod_metrics.Emeasure()
    sm = py_sod_metrics.Smeasure()
    wfm = py_sod_metrics.WeightedFmeasure()

    metrics_dict = dict()

    model.eval()
    test_loss = 0
    step = 0
    with torch.no_grad():
        for i in range(test_loader.size):
            image, gt, gt_tensor, _, _ = test_loader.load_data()

            gt = np.asarray(gt, np.float32)
            image = image.cuda()

            res,p2 = model(image)

            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            res_u8 = (res * 255).astype(np.uint8)
            gt_u8 = gt

            if gt_u8.max() == 0:
                continue

            mae.step(res_u8, gt_u8)
            fm.step(res_u8, gt_u8)
            em.step(res_u8, gt_u8)
            sm.step(res_u8, gt_u8)
            wfm.step(res_u8, gt_u8)

        metrics_dict.update(Sm=sm.get_results()['sm'])
        metrics_dict.update(meanEm=em.get_results()['em']['curve'].mean())
        metrics_dict.update(wFm=wfm.get_results()['wfm'])
        metrics_dict.update(mae=mae.get_results()['mae'])

        cur_score_mae = metrics_dict['mae']
        cur_score = metrics_dict['Sm'] + metrics_dict['wFm']

        print('[Cur Epoch] Metrics ( Sm={}, wFm={}, meanEm={}, MAE={})'.format(
             metrics_dict['Sm'], metrics_dict['wFm'], metrics_dict['meanEm'], metrics_dict['mae']))

def evaluator(model, val_root, map_save_path, trainsize=384):
    val_loader = EvalDataset(image_root=val_root + 'Imgs/',
                             gt_root=val_root + 'GT/',
                             testsize=trainsize)
    model.eval()
    with torch.no_grad():
        for i in range(val_loader.size):
            image, gt, gt_tensor, name, _ = val_loader.load_data()
            gt = np.asarray(gt, np.float32)

            image = image.cuda()

            res, _ = model(image)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            res = cv2.resize(res, (gt.shape[1], gt.shape[0]))

            res = (res * 255).astype(np.uint8)
            if not cv2.imwrite(os.path.join(map_save_path, name), res):
                raise RuntimeError(f'Failed to save prediction: {name}')

            print('>>> saving prediction at: {}'.format(os.path.join(map_save_path, name)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='PVTv2-b4')
    parser.add_argument('--snap_path', type=str, default='./Checkpoints/YourModel/Net_epoch_best.pth',
                        help='train use gpu')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='train use gpu')
    opt = parser.parse_args()
    
    txt_save_path = './Result/Test_result/{}/'.format(os.path.basename(os.path.dirname(os.path.normpath(opt.snap_path))))  
    os.makedirs(txt_save_path, exist_ok=True)

    print('>>> configs:', opt)

    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')

    cudnn.benchmark = True
    model = Network(arc=opt.model).cuda()

    model.load_state_dict(torch.load(opt.snap_path))
    model.eval()

    for data_name in ['CAMO', 'COD10K', 'NC4K','CHAMELEON']:
        map_save_path = txt_save_path + "{}/".format(data_name)
        os.makedirs(map_save_path, exist_ok=True)

        evaluator(
            model=model,
            val_root='./Dataset/TestDataset/' + data_name + '/',
            map_save_path=map_save_path,
            trainsize=384)
