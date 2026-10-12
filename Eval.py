import os
import torch
import numpy as np
import py_sod_metrics as Measure
import prettytable as pt
import argparse
import cv2
from tqdm import tqdm

def get_competitors(root):
    for model_name in os.listdir(root):
        print('\'{}\''.format(model_name), end=', ')

def evaluator(gt_pth_lst, pred_pth_lst):
    
    FM = Measure.Fmeasure()
    WFM = Measure.WeightedFmeasure()
    SM = Measure.Smeasure()
    EM = Measure.Emeasure()
    MAE = Measure.MAE()

    assert len(gt_pth_lst) == len(pred_pth_lst)

    with torch.no_grad():
        
        for idx in tqdm(range(len(gt_pth_lst))):
            gt_pth = gt_pth_lst[idx]
            pred_pth = pred_pth_lst[idx]

            assert os.path.isfile(gt_pth) and os.path.isfile(pred_pth)

            pred_ary = cv2.imread(pred_pth, cv2.IMREAD_UNCHANGED)
            gt_ary = cv2.imread(gt_pth, cv2.IMREAD_GRAYSCALE)
            if pred_ary is None or gt_ary is None:
                raise RuntimeError(f'Failed to read: {pred_pth} or {gt_pth}')
            if pred_ary.ndim == 3:
                pred_ary = cv2.cvtColor(pred_ary, cv2.COLOR_BGR2GRAY)
            if pred_ary.dtype == np.uint16:
                pred_ary = np.round(pred_ary.astype(np.float64) / 65535 * 255).astype(np.uint8)
            elif pred_ary.dtype != np.uint8:
                raise TypeError(f'Unsupported prediction dtype: {pred_ary.dtype} ({pred_pth})')
            if gt_ary.shape != pred_ary.shape:
                h,w = gt_ary.shape
                pred_ary = cv2.resize(pred_ary, (w,h))
            
            FM.step(pred=pred_ary, gt=gt_ary)
            WFM.step(pred=pred_ary, gt=gt_ary)
            SM.step(pred=pred_ary, gt=gt_ary)
            EM.step(pred=pred_ary, gt=gt_ary)
            MAE.step(pred=pred_ary, gt=gt_ary)

        fm = FM.get_results()['fm']
        wfm = WFM.get_results()['wfm']
        sm = SM.get_results()['sm']
        em = EM.get_results()['em']
        mae = MAE.get_results()['mae']

    return fm, wfm, sm, em, mae

def eval_all(opt, txt_save_path):
    
    for _data_name in opt.data_lst:
        print('#' * 20, _data_name, '#' * 20)
        filename = os.path.join(txt_save_path, '{}_eval.txt'.format(_data_name))
        with open(filename, 'w+') as file_to_write:
            tb = pt.PrettyTable()
            tb.field_names = ["Dataset", "Method", "Smeasure", "wFmeasure", "MAE", "adpEm", "meanEm", "maxEm", "adpFm",
                              "meanFm", "maxFm"]
            for _model_name in opt.model_lst:
                print('#' * 10, _model_name, '#' * 10)
                gt_src = os.path.join(opt.gt_root, _data_name, 'GT')
                pred_src = os.path.join(opt.pred_root, _model_name, _data_name)

                img_name_lst = sorted(i for i in os.listdir(gt_src) if os.path.isfile(os.path.join(gt_src, i)))

                fm, wfm, sm, em, mae = evaluator(
                    gt_pth_lst=[os.path.join(gt_src, i) for i in img_name_lst],
                    pred_pth_lst=[os.path.join(pred_src, i) for i in img_name_lst]
                )
                tb.add_row([_data_name, _model_name, sm.round(3), wfm.round(3), mae.round(3), em['adp'].round(3),
                            em['curve'].mean().round(3), em['curve'].max().round(3), fm['adp'].round(3),
                            fm['curve'].mean().round(3), fm['curve'].max().round(3)])
            print(tb)
            file_to_write.write(str(tb))

def eval_super_class(opt):
    super_classes = ['Aquatic', 'Flying', 'Amphibian', 'Terrestrial', 'Other']
    dataset = 'COD10K'
    tb = pt.PrettyTable()
    tb.field_names = ['Dataset', 'Method', 'Smeasure', 'wFmeasure', 'MAE', 'adpEm', 'meanEm', 'maxEm', 'adpFm', 'meanFm', 'maxFm']
    gt_dir = os.path.join(opt.gt_root, dataset, 'GT')
    for method in opt.model_lst:
        pred_dir = os.path.join(opt.pred_root, method, dataset)
        for group in super_classes:
            names = sorted(n for n in os.listdir(gt_dir) if group in n and os.path.isfile(os.path.join(gt_dir, n)))
            if not names:
                continue
            fm, wfm, sm, em, mae = evaluator(
                [os.path.join(gt_dir, n) for n in names],
                [os.path.join(pred_dir, n) for n in names]
            )
            tb.add_row([group, method, round(float(sm), 3), round(float(wfm), 3), round(float(mae), 3),
                        round(float(em['adp']), 3), round(float(em['curve'].mean()), 3),
                        round(float(em['curve'].max()), 3), round(float(fm['adp']), 3),
                        round(float(fm['curve'].mean()), 3), round(float(fm['curve'].max()), 3)])
    print(tb)

def eval_sub_class(opt):
    raise NotImplementedError('Subclass evaluation requires a defined COD10K subclass list.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gt_root', type=str, help='ground-truth root',
        default='./Dataset/TestDataset')
    parser.add_argument(
        '--pred_root', type=str, help='prediction root',
        
        default='./Result/Test_result/') 
    parser.add_argument(
        '--gpu_id', type=str, help='gpu device',
        default='0')
    parser.add_argument(
        '--data_lst', nargs='+', help='test dataset',
        default=['CAMO', 'COD10K', 'NC4K','CHAMELEON'],
        
        choices=['CHAMELEON', 'CAMO', 'COD10K', 'NC4K'])

    parser.add_argument(
        '--model_lst', nargs='+', help='candidate competitors',
        default=['YourModel']) 
    parser.add_argument(
        '--txt_name', type=str, help='candidate competitors',
        default='YourModel') 
    parser.add_argument(
        '--check_integrity', action=argparse.BooleanOptionalAction, help='whether to check the file integrity',
        default=True)
    parser.add_argument(
        '--eval_type', type=str, help='evaluation type',
        default='eval_all',
        choices=['eval_all', 'eval_super', 'eval_sub'])
    opt = parser.parse_args()

    txt_save_path = './Result/txt_result/{}/'.format(opt.txt_name)
    os.makedirs(txt_save_path, exist_ok=True)

    if opt.check_integrity:
        for _data_name in opt.data_lst:
            for _model_name in opt.model_lst:
                gt_pth = os.path.join(opt.gt_root, _data_name, 'GT')
                pred_pth = os.path.join(opt.pred_root, _model_name, _data_name)
                if not sorted(os.listdir(gt_pth)) == sorted(os.listdir(pred_pth)):
                    print(len(sorted(os.listdir(gt_pth))), len(sorted(os.listdir(pred_pth))))
                    raise ValueError('The {} dataset of {} model does not match the ground-truth'.format(_data_name, _model_name))
    else:
        print('>>> skip check the integrity of each candidates')

    if opt.eval_type == 'eval_all':
        eval_all(opt, txt_save_path)
    elif opt.eval_type == 'eval_super':
        eval_super_class(opt)
    elif opt.eval_type == 'eval_sub':
        eval_sub_class(opt)
    else:
        raise Exception
