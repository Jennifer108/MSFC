import argparse
import datetime
import logging
import os
import random
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import OrderedDict

sys.path.append(".")

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import coco as coco
from model.losses import get_masked_ptc_loss, get_seg_loss, CTCLoss_neg, DenseEnergyLoss, get_energy_loss
from model.model_seg_neg import network
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from model.PAR import PAR
from utils import evaluate, imutils, optimizer
from utils.camutils import cam_to_label, cam_to_roi_mask2, multi_scale_cam2, label_to_aff_mask, refine_cams_with_bkg_v2, crop_from_roi_neg,multi_scale_cam1
from utils.pyutils import AverageMeter, cal_eta, format_tabs, setup_logger

os.environ['RANK'] = '0' 
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

torch.hub.set_dir("./pretrained")
parser = argparse.ArgumentParser() 

parser.add_argument("--backbone", default='deit_base_patch16_224', type=str, help="vit_base_patch16_224")
parser.add_argument("--pooling", default='gmp', type=str, help="pooling choice for patch tokens")
parser.add_argument("--pretrained", default=False, type=bool, help="use imagenet pretrained weights")
parser.add_argument("--img_folder", default='/home/newdisk/fty/LZ/MSFC/MSCOCO/coco2014', type=str, help="dataset folder")
parser.add_argument("--label_folder", default='/home/newdisk/fty/LZ/MSFC/MSCOCO/SegmentationClass', type=str, help="dataset folder")
parser.add_argument("--list_folder", default='/home/newdisk/fty/LZ/MSFC/datasets/coco', type=str, help="train/val/test list file")

parser.add_argument("--num_classes", default=81, type=int, help="number of classes")
parser.add_argument("--crop_size", default=448, type=int, help="crop_size in training")
parser.add_argument("--local_crop_size", default=96, type=int, help="crop_size for local view")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")
parser.add_argument("--work_dir", default="/home/newdisk/fty/LZ/MSFC/work_dir_coco_wseg_deit", type=str)

parser.add_argument("--train_set", default="train", type=str, help="training split")
parser.add_argument("--val_set", default="val_part", type=str, help="validation split")

parser.add_argument("--spg", default=4, type=int, help="samples_per_gpu")
parser.add_argument("--scales", default=(0.5, 2), help="random rescale in training")

parser.add_argument("--optimizer", default='PolyWarmupAdamW', type=str, help="optimizer")
parser.add_argument("--lr", default=6e-5, type=float, help="learning rate")
parser.add_argument("--warmup_lr", default=1e-6, type=float, help="warmup_lr")
parser.add_argument("--wt_decay", default=1e-2, type=float, help="weights decay")
parser.add_argument("--betas", default=(0.9, 0.999), help="betas for Adam")
parser.add_argument("--power", default=0.9, type=float, help="poweer factor for poly scheduler")

parser.add_argument("--max_iters", default=80000, type=int, help="max training iters")
parser.add_argument("--log_iters", default=200, type=int, help=" logging iters")
parser.add_argument("--eval_iters", default=2000, type=int, help="validation iters")
parser.add_argument("--warmup_iters", default=1500, type=int, help="warmup_iters")

parser.add_argument("--high_thre", default=0.65, type=float, help="high_bkg_score")
parser.add_argument("--low_thre", default=0.25, type=float, help="low_bkg_score")
parser.add_argument("--bkg_thre", default=0.45, type=float, help="bkg_score")
parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5), help="multi_scales for cam")

parser.add_argument("--w_ptc", default=0.2, type=float, help="w_ptc")
parser.add_argument("--w_ctc", default=0.5, type=float, help="w_ctc")
parser.add_argument("--w_seg", default=0.1, type=float, help="w_seg")
parser.add_argument("--w_reg", default=0.05, type=float, help="w_reg")

parser.add_argument("--temp", default=0.5, type=float, help="temp")
parser.add_argument("--momentum", default=0.9, type=float, help="temp")

parser.add_argument("--seed", default=0, type=int, help="fix random seed")
parser.add_argument("--save_ckpt", action="store_true", help="save_ckpt")

parser.add_argument("--local_rank", default=0, type=int, help="local_rank")
parser.add_argument("--num_workers", default=10, type=int, help="num_workers")
parser.add_argument('--backend', default='nccl')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def getFeatures(fts, mask):
    fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')
    masked_fts = torch.sum(fts * mask[None, ...], dim=(3, 4)) \
        / (mask[None, ...].sum(dim=(3, 4)) + 1e-5) # 1 x C
    return masked_fts

def validate(model=None, data_loader=None, args=None):

    preds, gts, cams, cams_aux = [], [], [], []
    model.eval()
    avg_meter = AverageMeter()
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):

            name, inputs, labels, cls_label = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            cls_label = cls_label.cuda()

            inputs  = F.interpolate(inputs, size=[args.crop_size, args.crop_size], mode='bilinear', align_corners=False)

            cls, segs, _, _ = model(inputs,)

            cls_pred = (cls>0).type(torch.int16)
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score": _f1})

            _cams, _cams_aux = multi_scale_cam2(model, inputs, args.cam_scales)
            resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label = cam_to_label(resized_cam, cls_label, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)

            resized_cam_aux = F.interpolate(_cams_aux, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label_aux = cam_to_label(resized_cam_aux, cls_label, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)

            cls_pred = (cls > 0).type(torch.int16)
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score": _f1})

            resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)

            preds += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
            cams += list(cam_label.cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))
            cams_aux += list(cam_label_aux.cpu().numpy().astype(np.int16))

    cls_score = avg_meter.pop('cls_score')
    seg_score = evaluate.scores(gts, preds, num_classes=args.num_classes)
    cam_score = evaluate.scores(gts, cams, num_classes=args.num_classes)
    cam_aux_score = evaluate.scores(gts, cams_aux, num_classes=args.num_classes)
    model.train()

    tab_results = format_tabs([cam_score, cam_aux_score, seg_score], name_list=["CAM", "aux_CAM", "Seg_Pred"], cat_list=coco.class_list)

    return cls_score, tab_results

def train(args=None):

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.backend, )
    logging.info("Total gpus: %d, samples per gpu: %d..."%(dist.get_world_size(), args.spg))

    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)

    train_dataset = coco.CocoClsDataset(
        img_dir=args.img_folder,
        label_dir=args.label_folder,
        name_list_dir=args.list_folder,
        split=args.train_set,
        stage='train',
        aug=True,
        rescale_range=args.scales,
        crop_size=args.crop_size,
        img_fliplr=True,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )

    val_dataset = coco.CocoSegDataset(
        img_dir=args.img_folder,
        label_dir=args.label_folder,
        name_list_dir=args.list_folder,
        split=args.val_set,
        stage='val',
        aug=False,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.spg,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
        sampler=train_sampler,
        prefetch_factor=4)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=False,
                            drop_last=False)

    device = torch.device(args.local_rank)

    model = network(
        backbone=args.backbone,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        init_momentum=args.momentum,
        aux_layer=9
    )

    data_list = []
    for i in range(80):
        file_path = f"/home/newdisk/fty/LZ/MSFC/MemoryBank/{i}.npy"
        data = np.load(file_path)
        data_list.append(data)
    ClassMemoryBank=np.vstack(data_list) 
    ClassMemoryBank = torch.from_numpy(ClassMemoryBank).to(device, non_blocking=True)  

    trained_state_dict = torch.load('/home/newdisk/fty/LZ/MSFC/results/MSFC_deit-b_coco_80k.pth', map_location="cpu")
    new_state_dict = OrderedDict()
    for k, v in trained_state_dict.items(): 
        k = k.replace('module.', '') 
        new_state_dict[k] = v
    model.load_state_dict(state_dict=new_state_dict, strict=False)

    param_groups = model.get_param_groups()
    model.to(device)

    optim = getattr(optimizer, args.optimizer)(
        params=[
            {
                "params": param_groups[0],
                "lr": args.lr,
                "weight_decay": args.wt_decay,
            },
            {
                "params": param_groups[1],
                "lr": args.lr,
                "weight_decay": args.wt_decay,
            },
            {
                "params": param_groups[2],
                "lr": args.lr * 10,
                "weight_decay": args.wt_decay,
            },
            {
                "params": param_groups[3],
                "lr": args.lr * 10,
                "weight_decay": args.wt_decay,
            },
        ],
        lr=args.lr,
        weight_decay=args.wt_decay,
        betas=args.betas,
        warmup_iter=args.warmup_iters,
        max_iter=args.max_iters,
        warmup_ratio=args.warmup_lr,
        power=args.power)

    logging.info('\nOptimizer: \n%s' % optim)
    model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    train_sampler.set_epoch(np.random.randint(args.max_iters))
    train_loader_iter = iter(train_loader)
    avg_meter = AverageMeter()

    loss_layer = DenseEnergyLoss(weight=1e-7, sigma_rgb=15, sigma_xy=100, scale_factor=0.5)
    ncrops = 10
    CTC_loss = CTCLoss_neg(ncrops=ncrops, temp=1.0).cuda()

    par = PAR(num_iter=10, dilations=[1,2,4,8,12,24],devicePAR=device).cuda()

    for n_iter in range(args.max_iters):

        try:
            img_name, inputs, cls_label, img_box, crops ,multiImg = next(train_loader_iter)

        except:
            train_sampler.set_epoch(np.random.randint(args.max_iters))
            train_loader_iter = iter(train_loader)
            img_name, inputs, cls_label, img_box, crops ,multiImg = next(train_loader_iter)

        inputs = inputs.to(device, non_blocking=True)
        inputs_denorm = imutils.denormalize_img2(inputs.clone())
        cls_label = cls_label.to(device, non_blocking=True)

        cams, cams_aux = multi_scale_cam2(model, inputs=inputs, scales=args.cam_scales)
        roi_mask = cam_to_roi_mask2(cams_aux.detach(), cls_label=cls_label, low_thre=args.low_thre, hig_thre=args.high_thre)

        local_crops, flags = crop_from_roi_neg(images=crops[2], roi_mask=roi_mask, crop_num=ncrops-2, crop_size=args.local_crop_size)
        roi_crops = crops[:2] + local_crops

        cls, segs, fmap, cls_aux, out_t, out_s,cam = model(inputs, crops=roi_crops, n_iter=n_iter)

        multiImg[0] = multiImg[0].to(device, non_blocking=True)
        multiImg[1] = multiImg[1].to(device, non_blocking=True)
        multiImg[2] = multiImg[2].to(device, non_blocking=True)
        multiImg[3] = multiImg[3].to(device, non_blocking=True)
        multiImg[4] = multiImg[4].to(device, non_blocking=True)
        multiImg[5] = multiImg[5].to(device, non_blocking=True)

        SizeCam, cams_aux1 = multi_scale_cam1(model, inputs=multiImg[0], scales=args.cam_scales)
        AngleCam, cams_aux1 = multi_scale_cam1(model, inputs=multiImg[1], scales=args.cam_scales)
        CropsCam1, cams_aux1 = multi_scale_cam1(model, inputs=multiImg[2], scales=args.cam_scales)
        CropsCam2, cams_aux1 = multi_scale_cam1(model, inputs=multiImg[3], scales=args.cam_scales)
        CropsCam3, cams_aux1 = multi_scale_cam1(model, inputs=multiImg[4], scales=args.cam_scales)
        CropsCam4, cams_aux1 = multi_scale_cam1(model, inputs=multiImg[5], scales=args.cam_scales)

        SizeCam = F.interpolate(SizeCam, size=(AngleCam.shape[2], AngleCam.shape[3]), mode='bilinear', align_corners=False)
        AngleCam = torch.flip(AngleCam, [2, 3])
        output1 = torch.cat([CropsCam1, CropsCam2], dim=2)
        output2 = torch.cat([CropsCam3, CropsCam4], dim=2)
        CropsCam = torch.cat([output1, output2], dim=3)

        SizeCam = F.normalize(SizeCam.unsqueeze(0), p=2, dim=1)
        AngleCam = F.normalize(AngleCam.unsqueeze(0), p=2, dim=1)
        CropsCam = F.normalize(AngleCam.unsqueeze(0), p=2, dim=1)

        target_error1 = torch.mean(torch.abs(SizeCam-AngleCam))
        target_error2 = torch.mean(torch.abs(SizeCam-CropsCam))
        target_error3 = torch.mean(torch.abs(AngleCam-CropsCam))
        target_error4 = torch.mean(torch.abs(SizeCam-cams))
        target_error5 = torch.mean(torch.abs(CropsCam-cams))
        target_error6 = torch.mean(torch.abs(AngleCam-cams))
        target_error_loss=(target_error1+target_error2+target_error3+target_error4+target_error5+target_error6)/6

        Pcam_label = cam_to_label(cams, cls_label, bkg_thre=args.bkg_thre)
        flattened_tensor = torch.flatten(Pcam_label)
        nonzero_elements = torch.unique(flattened_tensor[flattened_tensor != 0])
        classNum=80  
        protypeList = [[] for _ in range(classNum)]
        MeanprotypeList = [None for _ in range(classNum)]
        Pfmp=fmap
        ClassMemoryBankBefore=ClassMemoryBank
        for element in nonzero_elements:
                Pcam_labelNew=Pcam_label.clone()
                Pcam_labelNew[Pcam_label==element.item()] = 1
                Pcam_labelNew[Pcam_label!=element.item()] = 0
                Pcam_labelNew=Pcam_labelNew.unsqueeze(0)
                Pfmp=Pfmp[0,:,:,:].unsqueeze(0)
                protype=getFeatures(Pfmp,Pcam_labelNew)
                protypeList[element.item()-1].append(protype)
        
        if Iterindex==200:
            Iterindex=0
            for index in range(len(protypeList)):
                if len(protypeList[index]) != 0:
                    concatenated_tensor = torch.cat(protypeList[index], dim=0)
                    average_tensor = torch.mean(concatenated_tensor, dim=0)
                    m=0.1  
                    ClassMemoryBank[index,:]=ClassMemoryBank[index,:]*m+average_tensor.squeeze().square()*(1-m)
                    protypeList = [[] for _ in range(classNum)]


        cosine_similarity  = F.cosine_similarity(fmap, ClassMemoryBankBefore[..., None, None], dim=1).unsqueeze(0)  
        _, pseudo_label_aux2 = cam_to_label(cosine_similarity.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index) 
        aff_mask2 = label_to_aff_mask(pseudo_label_aux2)
        MemoryPTC_loss = get_masked_ptc_loss(fmap, aff_mask2)

        cls_loss = F.multilabel_soft_margin_loss(cls, cls_label)
        cls_loss_aux = F.multilabel_soft_margin_loss(cls_aux, cls_label)

        ctc_loss = CTC_loss(out_s, out_t, flags)
        
        valid_cam, _ = cam_to_label(
            cams.detach(), 
            cls_label=cls_label, 
            img_box=img_box, ignore_mid=True, 
            bkg_thre=args.bkg_thre, 
            high_thre=args.high_thre, 
            low_thre=args.low_thre, 
            ignore_index=args.ignore_index)
        valid_cam_aux, _ = cam_to_label(
            cams_aux.detach(), 
            cls_label=cls_label, 
            img_box=img_box, 
            ignore_mid=True, 
            bkg_thre=args.bkg_thre, 
            high_thre=args.high_thre, 
            low_thre=args.low_thre, 
            ignore_index=args.ignore_index)

        if n_iter <= 12000:
            refined_pseudo_label = refine_cams_with_bkg_v2(par, inputs_denorm, cams=valid_cam_aux, cls_labels=cls_label, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index, img_box=img_box, )
        else:
            refined_pseudo_label = refine_cams_with_bkg_v2(par, inputs_denorm, cams=valid_cam, cls_labels=cls_label, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index, img_box=img_box, )
        
        segs = F.interpolate(segs, size=refined_pseudo_label.shape[1:], mode='bilinear', align_corners=False)
        seg_loss = get_seg_loss(segs, refined_pseudo_label.type(torch.long), ignore_index=args.ignore_index)

        reg_loss = get_energy_loss(img=inputs, logit=segs, label=refined_pseudo_label, img_box=img_box, loss_layer=loss_layer)

        resized_cams_aux = F.interpolate(cams_aux, size=fmap.shape[2:], mode="bilinear", align_corners=False)
        _, pseudo_label_aux = cam_to_label(resized_cams_aux.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)
        aff_mask = label_to_aff_mask(pseudo_label_aux)
        ptc_loss = get_masked_ptc_loss(fmap, aff_mask)
        loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + args.w_ptc * ptc_loss + args.w_ctc * ctc_loss +args.w_seg * seg_loss + args.w_reg * reg_loss + 0.1*target_error_loss + 0.5*MemoryPTC_loss
        cls_pred = (cls > 0).type(torch.int16)
        cls_score = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])

        avg_meter.add({
            'cls_loss': cls_loss.item(),
            'ptc_loss': ptc_loss.item(),
            'ctc_loss': ctc_loss.item(),
            'cls_loss_aux': cls_loss_aux.item(),
            'seg_loss': seg_loss.item(),
            'cls_score': cls_score.item(),
        })

        optim.zero_grad()
        loss.backward()
        optim.step()

        if (n_iter + 1) % args.log_iters == 0:

            delta, eta = cal_eta(time0, n_iter + 1, args.max_iters)
            cur_lr = optim.param_groups[0]['lr']

            if args.local_rank == 0:
                logging.info("Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; cls_loss: %.4f, cls_loss_aux: %.4f, ptc_loss: %.4f, ctc_loss: %.4f, seg_loss: %.4f..." % (n_iter + 1, delta, eta, cur_lr, avg_meter.pop('cls_loss'), avg_meter.pop('cls_loss_aux'), avg_meter.pop('ptc_loss'), avg_meter.pop('ctc_loss'), avg_meter.pop('seg_loss')))

        if (n_iter + 1) % args.eval_iters == 0:
            ckpt_name = os.path.join(args.ckpt_dir, "model_iter_%d.pth" % (n_iter + 1))
            if args.local_rank == 0:
                logging.info('Validating...')
                if args.save_ckpt:
                    torch.save(model.state_dict(), ckpt_name)
            val_cls_score, tab_results = validate(model=model, data_loader=val_loader, args=args)
            if args.local_rank == 0:
                logging.info("val cls score: %.6f" % (val_cls_score))
                logging.info("\n"+tab_results)

    return True


if __name__ == "__main__":

    args = parser.parse_args()

    timestamp = "{0:%Y-%m-%d-%H-%M-%S-%f}".format(datetime.datetime.now())
    args.work_dir = os.path.join(args.work_dir, timestamp)
    args.ckpt_dir = os.path.join(args.work_dir, "checkpoints")
    args.pred_dir = os.path.join(args.work_dir, "predictions")

    if args.local_rank == 0:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.makedirs(args.pred_dir, exist_ok=True)

        setup_logger(filename=os.path.join(args.work_dir, 'train.log'))
        logging.info('Pytorch version: %s' % torch.__version__)
        logging.info("GPU type: %s"%(torch.cuda.get_device_name(0)))
        logging.info('\nargs: %s' % args)

    setup_seed(args.seed)
    train(args=args)
