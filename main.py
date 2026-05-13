from dataset.semi import SemiDataset
from model.semseg.deeplabv2 import DeepLabV2
from model.semseg.deeplabv3plus import DeepLabV3Plus, DeepLabV3Plus_Aux, DeepLabV3Plus_Aux_RGL
from model.semseg.pspnet import PSPNet
from utils import count_params, meanIOU, color_map

from augmentation import *
from dataloader import GPSDataset, GPSDataset2, GPSDataset3, GPSDatasetT
import torch.nn as nn

import argparse
from copy import deepcopy
import numpy as np
import os
from PIL import Image
import torch
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
# from tqdm.notebook import tqdm
import random


MODE = None


def parse_args():
    parser = argparse.ArgumentParser(description='ST and ST++ Framework')

    # basic settings
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--crop-size', type=int, default=None)
    parser.add_argument('--backbone', type=str, choices=['resnet50', 'resnet50_whitening'], default='resnet50')
    parser.add_argument('--model', type=str, choices=['deeplabv3plusaux', 'deeplabv3plus', 'pspnet', 'deeplabv2', 'deeplabv3plusauxrgl'],
                        default='deeplabv3plusaux')

    # semi-supervised settings
    parser.add_argument('--labeled-id-path', type=str, required=True)
    parser.add_argument('--unlabeled-id-path', type=str, required=True)
    parser.add_argument('--pseudo-mask-path', type=str, required=True)

    parser.add_argument('--save-path', type=str, required=True)

    # arguments for ST++
    parser.add_argument('--reliable-id-path', type=str)
    parser.add_argument('--plus', dest='plus', default=True, action='store_true',
                        help='whether to use ST++')
    
    parser.add_argument('--class_name', default='label', type=str, help='landcover class name')
    parser.add_argument('--labeled_num', default=1000, type=int, help='the number of labeled sample')
    
    parser.add_argument('--nogram', dest='nogram', action='store_true', help='whether to use gram matrix')

    args = parser.parse_args()
    return args

class Metrics:
    def __init__(self, num_classes, ignore_label):
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.hist = torch.zeros(num_classes, num_classes)

    def update(self, pred, target):
        # pred = pred.argmax(dim=1)
        keep = target != self.ignore_label
        self.hist += torch.bincount(target[keep] * self.num_classes + pred[keep], minlength=self.num_classes**2).view(self.num_classes, self.num_classes)

    def compute_iou(self):
        ious = self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1) - self.hist.diag())
        miou = ious[~ious.isnan()].mean().item()
        return ious.cpu().numpy().round(2).tolist(), round(miou, 2)

    def compute_f1(self):
        f1 = 2 * self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1))
        mf1 = f1[~f1.isnan()].mean().item()
        return f1.cpu().numpy().round(2).tolist(), round(mf1, 2)

    def compute_pixel_acc(self):
        acc = self.hist.diag() / self.hist.sum(1)
        macc = acc[~acc.isnan()].mean().item()
        return acc.cpu().numpy().round(2).tolist(), round(macc, 2)
    

        
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

SEEDS = [42, 43, 44]
set_seed(42)


def get_train_augmentation(size, seg_fill):
    return Compose([
#         ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        # RandomAdjustSharpness(sharpness_factor=0.1, p=0.5),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        # RandomGaussianBlur((3, 3), p=0.5),
#         RandomGrayscale(p=0.2),
        RandomRotation(degrees=10, p=0.3, seg_fill=seg_fill),
        RandomResizedCrop(size, scale=(0.5, 2.0), seg_fill=seg_fill),
    ])


def get_normalize():
    return Compose([
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


def get_strong_augmentation(size, seg_fill):
    return Compose([
#         ColorJitter(brightness=0.001, contrast=0.001, saturation=0.00, hue=0.000),
        # RandomAdjustSharpness(sharpness_factor=0.1, p=0.5),
        # RandomAutoContrast(p=0.2),
        RandomAutoContrast(p=0.2),
        RandomGaussianBlur((3, 3), p=0.2),
    ])

def get_val_augmentation(size):
    return Compose([
        Resize(size),
    ])


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

global_step = 0

traintransform = get_train_augmentation([256, 256], 255)
strongtransform = get_strong_augmentation([256, 256], 255)
valtransform = get_val_augmentation([256, 256])
normalize = get_normalize()
CE_loss = nn.CrossEntropyLoss()


device = 'cuda' 
class_num = 2

def main(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(args.pseudo_mask_path):
        os.makedirs(args.pseudo_mask_path)
    if args.plus and args.reliable_id_path is None:
        exit('Please specify reliable-id-path in ST++.')
    
    if args.nogram:
        print("This is no-gram baseline.\n")
    else:
        print("GRAM Matrix included.\n")
    criterion = CrossEntropyLoss(ignore_index=255)

    valset = SemiDataset(args.dataset, args.data_root, 'val', None, 
                         transform=traintransform, s_transform=strongtransform, normalize=normalize)
    valloader = DataLoader(valset, batch_size=4 if args.dataset == 'cityscapes' else 1,
                           shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    # <====================== Supervised training with labeled images (SupOnly) ======================>
    print('\n================> Total stage 1/%i: '
          'Supervised training on labeled images (SupOnly)' % (6 if args.plus else 3))

    global MODE
    MODE = 'train'

    

    

    trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size, args.labeled_id_path,
                           transform=traintransform, s_transform=strongtransform, normalize=normalize)
    trainset.ids = 2 * trainset.ids if len(trainset.ids) < 200 else trainset.ids
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=4, drop_last=True)

    model, optimizer = init_basic_elems(args)
    print('\nParams: %.1fM' % count_params(model))
    best_model, checkpoints = train(model, trainloader, valloader, criterion, optimizer, args)

    """
        ST framework without selective re-training
    """
    if not args.plus:
        # <============================= Pseudo label all unlabeled images =============================>
        print('\n\n\n================> Total stage 2/3: Pseudo labeling all unlabeled images')

        dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, args.unlabeled_id_path,
                              transform=traintransform, s_transform=strongtransform, normalize=normalize)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=2, drop_last=False)

        label(best_model, dataloader, args)

        # <======================== Re-training on labeled and unlabeled images ========================>
        print('\n\n\n================> Total stage 3/3: Re-training on labeled and unlabeled images')

        MODE = 'semi_train'

        trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size,
                               args.labeled_id_path, args.unlabeled_id_path, args.pseudo_mask_path,
                               transform=traintransform, s_transform=strongtransform, normalize=normalize)
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                 pin_memory=True, num_workers=4, drop_last=True)

        model, optimizer = init_basic_elems(args)

        train(model, trainloader, valloader, criterion, optimizer, args)

        return

    """
        ST++ framework with selective re-training
    """
    # <===================================== Select Reliable IDs =====================================>
    print('\n\n\n================> Total stage 2/6: Select reliable images for the 1st stage re-training')

    dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, args.unlabeled_id_path,
                          transform=traintransform, s_transform=strongtransform, normalize=normalize)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=2, drop_last=False)

    select_reliable(checkpoints, dataloader, args)

    # <================================ Pseudo label reliable images =================================>
    print('\n\n\n================> Total stage 3/6: Pseudo labeling reliable images')

    cur_unlabeled_id_path = os.path.join(args.reliable_id_path, 'reliable_ids.txt')
    dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, cur_unlabeled_id_path,
                          transform=traintransform, s_transform=strongtransform, normalize=normalize)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=2, drop_last=False)

    label(best_model, dataloader, args)

    # <================================== The 1st stage re-training ==================================>
    print('\n\n\n================> Total stage 4/6: The 1st stage re-training on labeled and reliable unlabeled images')

    MODE = 'semi_train'

    trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size,
                           args.labeled_id_path, cur_unlabeled_id_path, args.pseudo_mask_path,
                           transform=traintransform, s_transform=strongtransform, normalize=normalize)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=4, drop_last=True)

    model, optimizer = init_basic_elems(args)

    best_model = train(model, trainloader, valloader, criterion, optimizer, args)

    # <=============================== Pseudo label unreliable images ================================>
    print('\n\n\n================> Total stage 5/6: Pseudo labeling unreliable images')

    cur_unlabeled_id_path = os.path.join(args.reliable_id_path, 'unreliable_ids.txt')
    dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, cur_unlabeled_id_path,
    transform=traintransform, s_transform=strongtransform, normalize=normalize)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=2, drop_last=False)

    label(best_model, dataloader, args)

    # <================================== The 2nd stage re-training ==================================>
    print('\n\n\n================> Total stage 6/6: The 2nd stage re-training on labeled and all unlabeled images')

    trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size,
                           args.labeled_id_path, args.unlabeled_id_path, args.pseudo_mask_path,
                           transform=traintransform, s_transform=strongtransform, normalize=normalize)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=4, drop_last=True)

    model, optimizer = init_basic_elems(args)

    train(model, trainloader, valloader, criterion, optimizer, args)


def init_basic_elems(args):
    model_zoo = {'deeplabv3plusaux':DeepLabV3Plus_Aux, 'deeplabv3plus': DeepLabV3Plus, 'pspnet': PSPNet, 'deeplabv2': DeepLabV2, 'deeplabv3plusauxrgl':DeepLabV3Plus_Aux_RGL}
    if args.model == 'deeplabv3plusaux' or args.model == 'deeplabv3plusauxrgl':
        model = model_zoo[args.model](args.backbone, dilations=[6, 12, 18], nclass=2)
    else:
        model = model_zoo[args.model](args.backbone, nclass=2)#21 if args.dataset == 'pascal' else 19)

    head_lr_multiple = 10.0
    if args.model == 'deeplabv2':
        assert args.backbone == 'resnet101'
        model.load_state_dict(torch.load('pretrained/deeplabv2_resnet101_coco_pretrained.pth'))
        head_lr_multiple = 1.0

    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': args.lr},
                     {'params': [param for name, param in model.named_parameters()
                                 if 'backbone' not in name],
                      'lr': args.lr * head_lr_multiple}],
                    lr=args.lr, momentum=0.9, weight_decay=1e-4)

    model = DataParallel(model).cuda()

    return model, optimizer


def train(model, trainloader, valloader, criterion, optimizer, args):
    iters = 0
    total_iters = len(trainloader) * args.epochs
    
    previous_best = 0.0
    best_loss = 10000000000

    global MODE

    if MODE == 'train':
        checkpoints = []

    for epoch in range(args.epochs):
        print("\n==> Epoch %i, learning rate = %.4f\t\t\t\t\t previous best = %.2f" %
              (epoch, optimizer.param_groups[0]["lr"], previous_best))

        model.train()
        total_loss = 0.0

        try:
            for i, (img, img_w, mask) in enumerate(trainloader):
                img, mask = img.cuda(), mask.cuda()

                pred = model(img)
                pred_w = model(img_w)
                if args.model == 'deeplabv3plusaux':
                    pred_, gram = pred[0], pred[1]
                    pred_w_, gram_w = pred_w[0], pred_w[1]
                
                if args.nogram:
                    loss = criterion(pred_, mask.to(torch.int64))
                else:
                    loss = criterion(pred_, mask.to(torch.int64)) + 1e-3*torch.mean((gram_w.detach() - gram)**2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                iters += 1
                lr = args.lr * (1 - iters / total_iters) ** 0.9
                optimizer.param_groups[0]["lr"] = lr
                optimizer.param_groups[1]["lr"] = lr * 1.0 if args.model == 'deeplabv2' else lr * 10.0
        except ValueError as e:
            print(e)
        metrics = Metrics(2, 255) 
        # metric = meanIOU(num_classes=21 if args.dataset == 'pascal' else 19)

        model.eval()

        with torch.no_grad():
            for img, mask, _ in valloader:
                img = img.cuda()
                pred = model(img)
                if args.model == 'deeplabv3plusaux':
                    pred = pred[0]
                pred = torch.argmax(pred, dim=1)

                metrics.update(pred.cpu(), mask.to(torch.int64))
                # metric.add_batch(pred.cpu().numpy(), mask.numpy())
                # mIOU = metric.evaluate()[-1]
                _, mIOU = metrics.compute_iou()

        
        ious, mIOU = metrics.compute_iou()
        acc, macc = metrics.compute_pixel_acc()
        f1, mf1 = metrics.compute_f1()

        print("f1 : ", f1)
        print("miou : ", ious)
        print("Pixel Acc : ", acc)
        print("best_mIOU : ", previous_best)

        mIOU *= 100.0
        if total_loss < best_loss:
            print('Saving...')
            if previous_best != 0:
                os.remove(os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, previous_best)))
            best_loss = total_loss
            torch.save(model.module.state_dict(),
                       os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, mIOU)))

            best_model = deepcopy(model)

        if MODE == 'train' and ((epoch + 1) in [args.epochs // 3, args.epochs * 2 // 3, args.epochs]):
            checkpoints.append(deepcopy(model))

    if MODE == 'train':
        return best_model, checkpoints

    return best_model


def select_reliable(models, dataloader, args):
    if not os.path.exists(args.reliable_id_path):
        os.makedirs(args.reliable_id_path)

    for i in range(len(models)):
        models[i].eval()

    id_to_reliability = []

    with torch.no_grad():
        for img, mask, id in dataloader:
            id = id[0]
            img = img.cuda().float()

            preds = []
            for model in models:
                if args.model == 'deeplabv3plusaux':
                    preds.append(torch.argmax(model(img)[0], dim=1).cpu())
                    # preds.append(torch.argmax(model(img)[0], dim=1).cpu().numpy())
                else:
                    preds.append(torch.argmax(model(img), dim=1).cpu())
                    # preds.append(torch.argmax(model(img), dim=1).cpu().numpy())

            mIOU = []
            for i in range(len(preds) - 1):
                metric = Metrics(2, 255)
                metric.update(preds[i], preds[-1])
                # metric = meanIOU(num_classes=21 if args.dataset == 'pascal' else 19)
                # metric.add_batch(preds[i], preds[-1])
                mIOU.append(metric.compute_iou()[-1])
                # mIOU.append(metric.evaluate()[-1])

            reliability = sum(mIOU) / len(mIOU)
            id_to_reliability.append((id, reliability))

    id_to_reliability.sort(key=lambda elem: elem[1], reverse=True)
    with open(os.path.join(args.reliable_id_path, 'reliable_ids.txt'), 'w') as f:
        for elem in id_to_reliability[:len(id_to_reliability) // 2]:
            f.write(elem[0] + '\n')
    with open(os.path.join(args.reliable_id_path, 'unreliable_ids.txt'), 'w') as f:
        for elem in id_to_reliability[len(id_to_reliability) // 2:]:
            f.write(elem[0] + '\n')


def label(model, dataloader, args):
    model.eval()

    metric = Metrics(2, 255)
    # metric = meanIOU(num_classes=21 if args.dataset == 'pascal' else 19)
    cmap = color_map(args.dataset)

    with torch.no_grad():
        for img, mask, id in dataloader:
            mask, id = mask[0], id[0]
            img = img.cuda()
            pred = model(img, True)
            pred = torch.argmax(pred[0], dim=1).cpu()
            
#             metric.update(pred, mask.to(torch.int64))
#             _, mIOU = metric.compute_iou()
            # metric.add_batch(pred.numpy(), mask.numpy())
            # mIOU = metric.evaluate()[-1]
            
            pred = Image.fromarray(pred.squeeze(0).numpy().astype(np.uint8), mode='P')
            pred.putpalette(cmap)

            pred.save(r'%s/%s' % (args.pseudo_mask_path, os.path.basename(id+'.tif')))

#             tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))


if __name__ == '__main__':
    args = parse_args()

    if args.epochs is None:
        args.epochs = {'pascal': 80, 'cityscapes': 240}[args.dataset]
    if args.lr is None:
        args.lr = {'pascal': 0.001, 'cityscapes': 0.004}[args.dataset] / 16 * args.batch_size
    if args.crop_size is None:
        args.crop_size = 256#{'pascal': 321, 'cityscapes': 721, 'Karachi': 256}[args.dataset]

    print()
    print(args)

    main(args)