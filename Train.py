import os
import logging
import numpy as np
from datetime import datetime
from tensorboardX import SummaryWriter
from Net.NewNet.ACUC_Net import MyNet as Network  
import torch
import cv2
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import optim
import py_sod_metrics
from utils.clip_grad import clip_gradient
from utils.dataset import get_loader, test_dataset

def morph_gradient(mask, k=5):
    ad = k // 2
    mask_f = mask.float()
    dil = F.max_pool2d(mask_f, k, stride=1, padding=pad)
    ero = -F.max_pool2d(-mask_f, k, stride=1, padding=pad)
    return (dil - ero).clamp(0, 1)          

def beul(seg_logits, seg_gts, kernel=5):
    
    assert seg_logits.shape == seg_gts.shape
    sigmoid_x = seg_logits.sigmoid()

    loss_map = 1 - (2 * sigmoid_x - 1).abs().pow(2)   

    edge = morph_gradient(seg_gts, k=kernel)

    loss_map = loss_map * (1 - edge)

    mask_cnt = (1 - edge).sum(dim=[1, 2, 3], keepdim=True).clamp(min=1)
    ual = (loss_map.sum(dim=[1, 2, 3], keepdim=True) / mask_cnt).mean()
    return ual

def beul_Loss(preds, gts):

    return beul(seg_logits=preds, seg_gts=gts)

def structure_loss(pred, mask):
    
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def WBCE_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    return wbce.mean()

def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()

def save_checkpoint(states, is_best, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, 'checkpoint.pth')
    torch.save(states, filename)
    if is_best:
        torch.save(states['model'], os.path.join(save_dir, 'Net_epoch_best.pth'))
    print(f'Checkpoint saved: {filename}')

def train(train_loader, model, optimizer, epoch, save_path, writer):
    
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts, grads, edges) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.cuda()
            gts = gts.cuda()
            out_final, aux_logits = model(images)
            loss_main = structure_loss(out_final, gts)

            loss_aux = 0
            for aux_logit in aux_logits:
                loss_aux += structure_loss(aux_logit, gts)

            loss = loss_main + 0.25 * loss_aux

            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data

            if i % 20 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss_main: {:.4f}, Total_loss: {:.4f} '.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss_main.data, loss.data))
                logging.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss_main: {:.4f}, Total_loss: {:.4f} '.
                    format(epoch, opt.epoch, i, total_step, loss_main.data, loss.data))

                writer.add_scalar('Train/Loss', loss.item(), global_step=step)
                writer.add_scalar('Train/Loss_total', loss.item(), global_step=step)

        loss_all /= max(epoch_step, 1)
        logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', float(loss_all), global_step=epoch)

    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'Net_epoch_{}_zd.pth'.format(epoch))
        print('Save checkpoints successfully!')
        raise

def val(test_loader, model, epoch, save_path, writer):
    
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
            gts = gt_tensor.unsqueeze(1)
            gts = gts.cuda()

            gt = np.asarray(gt, np.float32)
            image = image.cuda()

            res, aux_res = model(image)
            loss_pred = structure_loss(res,gts)

            test_loss += loss_pred.data
            step += 1

            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            res = cv2.resize(res, gt.shape[::-1], interpolation=cv2.INTER_LINEAR)
            res_u8 = (res * 255).astype(np.uint8)
            gt[gt > 0] = 255
            
            if gt.max() == 0:
                continue
            
            mae.step(res_u8, gt)
            fm.step(res_u8, gt)
            em.step(res_u8, gt)
            sm.step(res_u8, gt)
            wfm.step(res_u8, gt)

        test_loss /= max(step, 1)
        writer.add_scalar('test_pred_loss', test_loss, global_step=epoch)

        metrics_dict.update(Sm=sm.get_results()['sm'])
        metrics_dict.update(meanEm=em.get_results()['em']['curve'].mean())
        metrics_dict.update(wFm=wfm.get_results()['wfm'])
        metrics_dict.update(mae=mae.get_results()['mae'])

        for k, v in metrics_dict.items():
            writer.add_scalar(f'Val/{k}', v, global_step=epoch)

        cur_score_mae = metrics_dict['mae']
        cur_score = metrics_dict['Sm'] + metrics_dict['wFm'] + metrics_dict['meanEm'] - metrics_dict['mae']

        if best_metric_dict is None:
            best_metric_dict = metrics_dict
            best_score = cur_score
            best_epoch = epoch
            torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')

            print('>>> save state_dict successfully! best epoch is {}.'.format(epoch))
            print('[Cur Epoch: {}] Metrics ( Sm={}, wFm={}, meanEm={}, MAE={})'.format(
                epoch, metrics_dict['Sm'], metrics_dict['wFm'], metrics_dict['meanEm'], metrics_dict['mae']))
            logging.info('>>> save state_dict successfully! best epoch is {}.'.format(epoch))
            logging.info('[Cur Epoch: {}] Metrics ( Sm={}, wFm={}, meanEm={},  MAE={})'.format(
                epoch, metrics_dict['Sm'], metrics_dict['wFm'], metrics_dict['meanEm'], metrics_dict['mae']))
        else:
            if cur_score > best_score:
                best_metric_dict = metrics_dict
                best_score = cur_score
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
                print('>>> save state_dict successfully! best epoch is {}.'.format(epoch))
                logging.info('>>> save state_dict successfully! best epoch is {}.'.format(epoch))

            else:
                print('>>> not find the best epoch -> continue training ...')
                logging.info('>>> not find the best epoch -> continue training ...')

            print('[Cur Epoch: {}] Metrics ( Sm={}, wFm={}, meanEm={},MAE={})\n[Best Epoch: {}] Metrics ( Sm={}, wFm={}, meanEm={},MAE={})'.format(
                    epoch, metrics_dict['Sm'], metrics_dict['wFm'], metrics_dict['meanEm'], metrics_dict['mae'],
                    best_epoch, best_metric_dict['Sm'], best_metric_dict['wFm'], best_metric_dict['meanEm'],best_metric_dict['mae']))
            logging.info('[Cur Epoch: {}] Metrics ( Sm={}, wFm={}, meanEm={},MAE={})\n[Best Epoch: {}] Metrics ( Sm={}, wFm={}, meanEm={},MAE={})'.format(
                    epoch, metrics_dict['Sm'], metrics_dict['wFm'], metrics_dict['meanEm'], metrics_dict['mae'],
                    best_epoch, best_metric_dict['Sm'], best_metric_dict['wFm'], best_metric_dict['meanEm'],best_metric_dict['mae']))

        writer.add_scalar('Test_Sm', metrics_dict['Sm'], global_step=epoch)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=50, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=2, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--model', type=str, default='PVTv2-b4')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--train_root', type=str, default='./Dataset/TrainDataset/',
                        help='the training rgb images root')
    parser.add_argument('--val_root', type=str, default='./Dataset/TestDataset/COD10K/',
                        help='the test rgb images root')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='train use gpu')
    parser.add_argument('--save_path', type=str, default='./Checkpoints/YourModel/',
                        help='the path to save model and log')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to checkpoint to resume training')
    
    opt = parser.parse_args()

    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')
    cudnn.benchmark = True

    model = Network(arc=opt.model).cuda()

    

    

    start_epoch = 0
    best_metric_dict = None
    if opt.resume is not None:  
        checkpoint = torch.load(opt.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer = torch.optim.Adam(model.parameters(), opt.lr)
        optimizer.load_state_dict(checkpoint['optimizer'])
        cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epoch, eta_min=1e-5)
        cosine_schedule.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        step = checkpoint['step']
        best_score = checkpoint.get('best_score', 0.0)
        best_epoch = checkpoint.get('best_epoch', 0)
        best_metric_dict = checkpoint.get('best_metric_dict')
        print(f'Resume training from epoch {start_epoch}')
    elif opt.load is not None:  
        model.load_state_dict(torch.load(opt.load))
        print('load model from ', opt.load)
        optimizer = torch.optim.Adam(model.parameters(), opt.lr)
        cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epoch, eta_min=1e-5)
        step = 0
        best_score = 0
        best_epoch = 0
    else:  
        optimizer = torch.optim.Adam(model.parameters(), opt.lr)
        cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epoch, eta_min=1e-5)
        step = 0
        best_score = 0
        best_epoch = 0

    save_path = os.path.join(opt.save_path, '')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print('load data...')
    train_loader = get_loader(image_root=opt.train_root + 'Imgs/',  
                              gt_root=opt.train_root + 'GT/',       
                              grad_root=opt.train_root + 'Gradient-Foreground/',
                              edge_root=opt.train_root + 'Edge/',
                              batchsize=opt.batchsize, 
                              trainsize=opt.trainsize, 
                              num_workers=4)
    val_loader = test_dataset(image_root=opt.val_root + 'Imgs/',
                              gt_root=opt.val_root + 'GT/',
                              testsize=opt.trainsize)
    total_step = len(train_loader)

    logging.basicConfig(filename=save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info(">>> current mode: network-train/val")
    logging.info('>>> config: {}'.format(opt))
    print('>>> config: : {}'.format(opt))

    writer = SummaryWriter(os.path.join(save_path, 'summary'))
    print(">>> start train...")
    for epoch in range(start_epoch, opt.epoch):

        logging.info('>>> current lr: {}'.format(cosine_schedule.get_last_lr()[0]))
        
        train(train_loader, model, optimizer, epoch, save_path, writer)
        val(val_loader, model, epoch, save_path, writer)
        cosine_schedule.step()
        writer.add_scalar('learning_rate', cosine_schedule.get_last_lr()[0], global_step=epoch)
        save_checkpoint({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': cosine_schedule.state_dict(),
            'step': step,
            'best_score': best_score,
            'best_epoch': best_epoch,
            'best_metric_dict': best_metric_dict,
        }, is_best=False, save_dir=save_path)
    writer.close()
