# Copyright (c) EEEM071, University of Surrey
# Modified: Added ArcFace, Circle Loss, EfficientNet, Swin Transformer

import datetime
import os
import os.path as osp
import sys
import time
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from args import argument_parser, dataset_kwargs, optimizer_kwargs, lr_scheduler_kwargs
from src import models
from src.data_manager import ImageDataManager
from src.eval_metrics import evaluate
from src.losses import CrossEntropyLoss, TripletLoss, DeepSupervision
from src.lr_schedulers import init_lr_scheduler
from src.optimizers import init_optimizer
from src.utils.avgmeter import AverageMeter
from src.utils.generaltools import set_random_seed
from src.utils.iotools import check_isfile
from src.utils.loggers import Logger, RankLogger
from src.utils.torchtools import (
    count_num_param,
    accuracy,
    load_pretrained_weights,
    save_checkpoint,
    resume_from_checkpoint,
)
from src.utils.visualtools import visualize_ranked_results

# Import new loss functions
from src_losses_arcface import ArcFaceLoss
from src_losses_circle import CircleLoss
from imbalance_circle_loss import ImbalanceAwareCircleLoss

# Import new backbone models
from src_models_efficientnet_reid import efficientnet_b3_reid, efficientnetv2_s_reid
from src_models_swin_reid import swin_base_reid, swin_small_reid

import uuid

# global variables
parser = argument_parser()
args = parser.parse_args()

# ============================================================
# ARCHITECTURE REGISTRY
# Maps arch name strings to model builder functions
# Includes both original and new backbones
# ============================================================
EXTRA_MODELS = {
    "efficientnet_b3": efficientnet_b3_reid,
    "efficientnetv2_s":  efficientnetv2_s_reid,
    "swin_base":       swin_base_reid,
    "swin_small":      swin_small_reid,
}


def build_model(arch, num_classes, loss, pretrained, use_gpu):
    """
    Build model by architecture name.
    Checks original models registry first, then EXTRA_MODELS.
    """
    if arch in EXTRA_MODELS:
        print(f"  => Loading new backbone: {arch}")
        model = EXTRA_MODELS[arch](
            num_classes=num_classes,
            loss=loss,
            pretrained=pretrained,
        )
    else:
        # Original models (resnet50_fc512 etc.)
        model = models.init_model(
            name=arch,
            num_classes=num_classes,
            loss=loss,
            pretrained=pretrained,
            use_gpu=use_gpu,
        )
    return model


def main():
    global args

    set_random_seed(args.seed)
    if not args.use_avai_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False

    log_name = "log_test.txt" if args.evaluate else "log_train.txt"
    sys.stdout = Logger(osp.join(args.save_dir, log_name))

    print("==========")
    student_id = os.environ.get('STUDENT_ID', '6967472')
    student_name = os.environ.get('STUDENT_NAME', 'Saunak Ghosh')
    print(f"Student ID:{student_id}")
    print(f"Student name:{student_name}")
    print(f"UUID:{uuid.uuid4()}")
    print(f"Experiment time:{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print("==========")
    print(f"==========\nArgs:{args}\n==========")
    print(f"Loss type: {args.loss_type}")
    print(f"Architecture: {args.arch}")

    if use_gpu:
        print(f"Currently using GPU {args.gpu_devices}")
        cudnn.benchmark = True
    else:
        warnings.warn("Currently using CPU, however, GPU is highly recommended")

    print("Initializing image data manager")
    dm = ImageDataManager(use_gpu, **dataset_kwargs(args))
    trainloader, testloader_dict = dm.return_dataloaders()

    print(f"Initializing model: {args.arch}")
    model = build_model(
        arch=args.arch,
        num_classes=dm.num_train_pids,
        loss={"xent", "htri"},
        pretrained=not args.no_pretrained,
        use_gpu=use_gpu,
    )
    print("Model size: {:.3f} M".format(count_num_param(model)))

    if args.load_weights and check_isfile(args.load_weights):
        load_pretrained_weights(model, args.load_weights)

    model = nn.DataParallel(model).cuda() if use_gpu else model

    # ============================================================
    # LOSS FUNCTION SETUP
    # Based on --loss-type argument:
    #   triplet  → CrossEntropy + Triplet Loss  (default)
    #   arcface  → ArcFace     + Triplet Loss
    #   circle   → CrossEntropy + Circle Loss
    # ============================================================
    print(f"\n=== Loss Function: {args.loss_type.upper()} ===")

    if args.loss_type == "arcface":
        print(f"  ArcFace: scale={args.arcface_scale}, margin={args.arcface_margin}")
        print(f"  Triplet: margin={args.margin} (hard mining)")
        criterion_xent = ArcFaceLoss(
            num_classes=dm.num_train_pids,
            embedding_size=512,
            scale=args.arcface_scale,
            margin=args.arcface_margin,
            use_gpu=use_gpu,
            label_smooth=args.label_smooth,
        )
        criterion_htri = TripletLoss(margin=args.margin)
        use_arcface = True
        use_circle = False

    elif args.loss_type == "imbalance_circle":
        print(f"  CrossEntropy: label_smooth={args.label_smooth}")
        print(f"  Imbalance-Aware Circle Loss: base_margin={args.circle_margin}")
        from collections import Counter
        from src.datasets import init_imgreid_dataset
        _tmp_dataset = init_imgreid_dataset(root=args.root, name=args.source_names[0])
        class_counts = Counter()
        for img_path, pid, camid in _tmp_dataset.train:
            class_counts[pid] += 1
        criterion_xent = CrossEntropyLoss(
            num_classes=dm.num_train_pids,
            use_gpu=use_gpu,
            label_smooth=args.label_smooth,
        )
        criterion_htri = ImbalanceAwareCircleLoss(
            class_counts=dict(class_counts),
            base_margin=args.circle_margin,
            gamma=80,
            power=0.5
        )
        use_arcface = False
        use_circle = True

    elif args.loss_type == "circle":
        print(f"  CrossEntropy: label_smooth={args.label_smooth}")
        print(f"  Circle Loss: scale={args.circle_scale}, margin={args.circle_margin}")
        criterion_xent = CrossEntropyLoss(
            num_classes=dm.num_train_pids,
            use_gpu=use_gpu,
            label_smooth=args.label_smooth,
        )
        criterion_htri = CircleLoss(
            scale=args.circle_scale,
            margin=args.circle_margin,
        )
        use_arcface = False
        use_circle = True

    else:
        # Default: triplet
        print(f"  CrossEntropy: label_smooth={args.label_smooth}")
        print(f"  Triplet Loss: margin={args.margin} (hard mining)")
        criterion_xent = CrossEntropyLoss(
            num_classes=dm.num_train_pids,
            use_gpu=use_gpu,
            label_smooth=args.label_smooth,
        )
        criterion_htri = TripletLoss(margin=args.margin)
        use_arcface = False
        use_circle = False

    print("=" * 40)

    optimizer = init_optimizer(model, **optimizer_kwargs(args))
    scheduler = init_lr_scheduler(optimizer, **lr_scheduler_kwargs(args))

    if args.resume and check_isfile(args.resume):
        args.start_epoch = resume_from_checkpoint(
            args.resume, model, optimizer=optimizer
        )

    if args.evaluate:
        print("Evaluate only")
        for name in args.target_names:
            print(f"Evaluating {name} ...")
            queryloader = testloader_dict[name]["query"]
            galleryloader = testloader_dict[name]["gallery"]
            distmat = test(
                model, queryloader, galleryloader, use_gpu, return_distmat=True
            )
            if args.visualize_ranks:
                visualize_ranked_results(
                    distmat,
                    dm.return_testdataset_by_name(name),
                    save_dir=osp.join(args.save_dir, "ranked_results", name),
                    topk=20,
                )
        return

    time_start = time.time()
    ranklogger = RankLogger(args.source_names, args.target_names)
    print("=> Start training")

    for epoch in range(args.start_epoch, args.max_epoch):
        train(
            epoch,
            model,
            criterion_xent,
            criterion_htri,
            optimizer,
            trainloader,
            use_gpu,
            use_arcface=use_arcface,
        )

        scheduler.step()

        if (
            (epoch + 1) > args.start_eval
            and args.eval_freq > 0
            and (epoch + 1) % args.eval_freq == 0
            or (epoch + 1) == args.max_epoch
        ):
            print("=> Test")
            for name in args.target_names:
                print(f"Evaluating {name} ...")
                queryloader = testloader_dict[name]["query"]
                galleryloader = testloader_dict[name]["gallery"]
                rank1 = test(model, queryloader, galleryloader, use_gpu)
                ranklogger.write(name, epoch + 1, rank1)

            save_checkpoint(
                {
                    "state_dict": model.state_dict(),
                    "rank1": rank1,
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "optimizer": optimizer.state_dict(),
                    "loss_type": args.loss_type,
                },
                args.save_dir,
            )

    elapsed = round(time.time() - time_start)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print(f"Elapsed {elapsed}")
    ranklogger.show_summary()


def train(
    epoch,
    model,
    criterion_xent,
    criterion_htri,
    optimizer,
    trainloader,
    use_gpu,
    use_arcface=False,
):
    """
    Training loop for one epoch.

    Supports three loss combinations:
    - Standard:  CrossEntropy + TripletLoss
    - ArcFace:   ArcFaceLoss  + TripletLoss
    - Circle:    CrossEntropy + CircleLoss

    For ArcFace: the classification criterion now operates on raw
    embeddings (f_t from bottleneck) rather than classifier logits,
    as ArcFace computes its own angular softmax internally.
    """
    xent_losses = AverageMeter()
    htri_losses = AverageMeter()
    accs = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    for p in model.parameters():
        p.requires_grad = True

    end = time.time()
    for batch_idx, (imgs, pids, _, _) in enumerate(trainloader):
        data_time.update(time.time() - end)

        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()

        outputs, features = model(imgs)

        # --------------------------------------------------------
        # Compute classification loss
        # --------------------------------------------------------
        if use_arcface:
            # ArcFace operates on raw embeddings (not classifier logits)
            # features = f_t from bottleneck (before BN)
            if isinstance(features, (tuple, list)):
                xent_loss = DeepSupervision(criterion_xent, features, pids)
            else:
                xent_loss = criterion_xent(features, pids)
        else:
            # Standard CrossEntropy on classifier logits
            if isinstance(outputs, (tuple, list)):
                xent_loss = DeepSupervision(criterion_xent, outputs, pids)
            else:
                xent_loss = criterion_xent(outputs, pids)

        # --------------------------------------------------------
        # Compute metric loss (triplet or circle)
        # --------------------------------------------------------
        if isinstance(features, (tuple, list)):
            htri_loss = DeepSupervision(criterion_htri, features, pids)
        else:
            htri_loss = criterion_htri(features, pids)

        # Combined loss
        loss = args.lambda_xent * xent_loss + args.lambda_htri * htri_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        xent_losses.update(xent_loss.item(), pids.size(0))
        htri_losses.update(htri_loss.item(), pids.size(0))
        accs.update(accuracy(outputs, pids)[0])

        if (batch_idx + 1) % args.print_freq == 0:
            loss_name = "Arc" if use_arcface else "Xent"
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.4f} ({data_time.avg:.4f})\t"
                "{loss_name} {xent.val:.4f} ({xent.avg:.4f})\t"
                "Metric {htri.val:.4f} ({htri.avg:.4f})\t"
                "Acc {acc.val:.2f} ({acc.avg:.2f})\t".format(
                    epoch + 1,
                    batch_idx + 1,
                    len(trainloader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss_name=loss_name,
                    xent=xent_losses,
                    htri=htri_losses,
                    acc=accs,
                )
            )

        end = time.time()


def test(
    model,
    queryloader,
    galleryloader,
    use_gpu,
    ranks=[1, 5, 10, 20],
    return_distmat=False,
):
    """Evaluation using CMC and mAP metrics."""
    batch_time = AverageMeter()
    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(queryloader):
            if use_gpu:
                imgs = imgs.cuda()
            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)
            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
        print("Extracted features for query set, obtained {}-by-{} matrix".format(
            qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(galleryloader):
            if use_gpu:
                imgs = imgs.cuda()
            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)
            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)
        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(
            gf.size(0), gf.size(1)))

    print(f"=> BatchTime(s)/BatchSize(img): {batch_time.avg:.3f}/{args.test_batch_size}")

    m, n = qf.size(0), gf.size(0)
    distmat = (
        torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n)
        + torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    )
    distmat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    distmat = distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results ----------")
    print(f"mAP: {mAP:.1%}")
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")

    if return_distmat:
        return distmat
    return cmc[0]


if __name__ == "__main__":
    main()
