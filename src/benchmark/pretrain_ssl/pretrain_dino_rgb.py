# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path
import builtins

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

from models.dino import utils
import models.dino.vision_transformer as vits
from models.dino.vision_transformer import DINOHead
import wandb
from datasets.SSL4EO.ssl4eo_dataset import SSL4EO, random_subset
from datasets.SSL4EO.ssl4eo_dataset_lmdb import LMDBDataset
from cvtorchvision import cvtransforms
from models.rs_transforms_uint8 import (
    RandomChannelDrop,
    GaussianBlur,
    Solarize,
    RandomBrightness,
    RandomContrast,
    ToGray,
    RandomSensorDrop_S1S2,
)
from models.dino.utils import load_pretrained_weights

### end of change ###
import pdb

from torch.utils.tensorboard import SummaryWriter

# import warnings
# warnings.filterwarnings("error")
from models.dino.vision_transformer import PatchEmbed


def update_patch_embed(model):
    """
    Update the patch embedding layer of the model with a new one.

    Args:
        model: The model whose patch embedding layer needs to be updated.
        new_patch_embed: The new patch embedding layer to be used.
    """
    # Replace the model's patch embedding layer with the new one
    new_patch_embed = PatchEmbed(
        img_size=model.patch_embed.img_size,
        patch_size=model.patch_embed.patch_size,
        in_chans=3,
        embed_dim=model.patch_embed.proj.weight.shape[0],
    )

    # Define the RGB channel indices from the pre-trained 13-channel weights.
    # Adjust these indices based on your dataset specifics.
    rgb_idx = [2, 3, 4]

    with torch.no_grad():
        new_patch_embed.proj.weight.copy_(
            model.patch_embed.proj.weight[:, rgb_idx, :, :]
        )
        if model.patch_embed.proj.bias is not None:
            new_patch_embed.proj.bias.copy_(model.patch_embed.proj.bias)

    # Replace the model's patch embedding layer with the new one
    model.patch_embed = new_patch_embed

    return model


torchvision_archs = sorted(
    name
    for name in torchvision_models.__dict__
    if name.islower()
    and not name.startswith("__")
    and callable(torchvision_models.__dict__[name])
)


def get_args_parser():
    parser = argparse.ArgumentParser("DINO", add_help=False)

    # Model parameters
    parser.add_argument(
        "--arch",
        default="vit_small",
        type=str,
        choices=["vit_tiny", "vit_small", "vit_base", "xcit", "deit_tiny", "deit_small"]
        + torchvision_archs,
        help="""Name of architecture to train. For quick experiments with ViTs, we recommend using vit_tiny or vit_small.""",
    )
    parser.add_argument(
        "--patch_size",
        default=16,
        type=int,
        help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""",
    )
    parser.add_argument(
        "--out_dim",
        default=65536,
        type=int,
        help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""",
    )
    parser.add_argument(
        "--norm_last_layer",
        default=True,
        type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""",
    )
    parser.add_argument(
        "--momentum_teacher",
        default=0.996,
        type=float,
        help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""",
    )
    parser.add_argument(
        "--use_bn_in_head",
        default=False,
        type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)",
    )

    # Temperature teacher parameters
    parser.add_argument(
        "--warmup_teacher_temp",
        default=0.04,
        type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""",
    )
    parser.add_argument(
        "--teacher_temp",
        default=0.04,
        type=float,
        help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""",
    )
    parser.add_argument(
        "--warmup_teacher_temp_epochs",
        default=0,
        type=int,
        help="Number of warmup epochs for the teacher temperature (Default: 30).",
    )

    # Training/Optimization parameters
    parser.add_argument(
        "--use_fp16",
        type=utils.bool_flag,
        default=True,
        help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.04,
        help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""",
    )
    parser.add_argument(
        "--weight_decay_end",
        type=float,
        default=0.4,
        help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""",
    )
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=3.0,
        help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""",
    )
    parser.add_argument(
        "--batch_size_per_gpu",
        default=64,
        type=int,
        help="Per-GPU batch-size : number of distinct images loaded on one GPU.",
    )
    parser.add_argument(
        "--epochs", default=100, type=int, help="Number of epochs of training."
    )
    parser.add_argument(
        "--freeze_last_layer",
        default=1,
        type=int,
        help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""",
    )
    parser.add_argument(
        "--lr",
        default=0.0005,
        type=float,
        help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""",
    )
    parser.add_argument(
        "--warmup_epochs",
        default=10,
        type=int,
        help="Number of epochs for the linear learning-rate warm up.",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""",
    )
    parser.add_argument(
        "--optimizer",
        default="adamw",
        type=str,
        choices=["adamw", "sgd", "lars"],
        help="""Type of optimizer. We recommend using adamw with ViTs.""",
    )
    parser.add_argument(
        "--drop_path_rate", type=float, default=0.1, help="stochastic depth rate"
    )

    # Multi-crop parameters
    parser.add_argument(
        "--global_crops_scale",
        type=float,
        nargs="+",
        default=(0.4, 1.0),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""",
    )
    parser.add_argument(
        "--local_crops_number",
        type=int,
        default=8,
        help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """,
    )
    parser.add_argument(
        "--local_crops_scale",
        type=float,
        nargs="+",
        default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""",
    )

    # Misc
    parser.add_argument(
        "--checkpoints_dir",
        default=".",
        type=str,
        help="Path to save logs and checkpoints.",
    )
    parser.add_argument(
        "--saveckp_freq", default=20, type=int, help="Save checkpoint every x epochs."
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    parser.add_argument(
        "--num_workers",
        default=10,
        type=int,
        help="Number of data loading workers per GPU.",
    )
    parser.add_argument(
        "--dist_url",
        default="env://",
        type=str,
        help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""",
    )
    parser.add_argument(
        "--local_rank",
        default=0,
        type=int,
        help="Please ignore and do not set this argument.",
    )

    # new
    parser.add_argument(
        "--data",
        default="/path/to/imagenet/",
        type=str,
        help="Please specify path to the ImageNet folder.",
    )
    parser.add_argument("--bands", type=str, default="all", help="input bands")
    parser.add_argument("--lmdb", action="store_true", help="use lmdb dataset")
    parser.add_argument("--is_slurm_job", action="store_true", help="running in slurm")
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint")

    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--mode", nargs="*", default=["s2c"])
    parser.add_argument("--dtype", type=str, default="uint8")
    parser.add_argument("--season", type=str, default="augment")
    parser.add_argument("--strategy", type=str, default="average")
    parser.add_argument("--in_size", type=int, default=224)

    parser.add_argument(
        "--wandb_project", type=str, default="SSL4EO-RUNS", help="WandB project name"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="msc_thesis_proj",
        help="WandB entity (username or team)",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="ssl4eo_vits_rgb_finetune",
        help="Optional run name for WandB",
    )
    parser.add_argument(
        "--wandb_log", action="store_true", help="Enable logging to WandB"
    )

    return parser


def train_dino(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)

    if args.wandb_log and utils.is_main_process():
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=vars(args),
        )

    # suppress printing if not master
    if args.is_slurm_job and args.rank != 0:

        def print_pass(*args):
            pass

        builtins.print = print_pass

    print("git:\n  {}\n".format(utils.get_sha()))
    print(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = DataAugmentationDINO_RGB(
        args.in_size,
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
        args.season,
        args.strategy,
    )

    bands = ["B04", "B03", "B02"]
    args.n_channels = 3

    dataset = SSL4EO(
        root=args.data,
        normalize=args.normalize,
        mode=args.mode,
        dtype=args.dtype,
        transform=transform,
    )
    dataset = random_subset(dataset, 0.1, 42)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print()
    # sample = next(iter(data_loader))
    # print(type(sample), type(sample[0]))
    # print(len(sample), len(sample[0]))
    # print(sample[0][0].shape)
    # dist.destroy_process_group()
    # quit(0)
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        # For other bands (unchanged original behavior)
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,
            in_chans=args.n_channels,
        )
        teacher = vits.__dict__[args.arch](
            patch_size=args.patch_size, in_chans=args.n_channels
        )

        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models, [yi:need to adjust for more in_channels]
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        load_pretrained_weights(
            student,
            r"/ix/cs2770_2025s/dtk28/checkpts/ssl4eo/B13_vits16_dino_0099_ckpt.pth",
            "student",
            "vits",
            16,
        )
        load_pretrained_weights(
            teacher,
            r"/ix/cs2770_2025s/dtk28/checkpts/ssl4eo/B13_vits16_dino_0099_ckpt.pth",
            "teacher",
            "vits",
            16,
        )
        student = update_patch_embed(student)
        teacher = update_patch_embed(teacher)
        embed_dim = student.fc.weight.shape[1]
        student.conv1 = nn.Conv2d(
            args.n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        teacher.conv1 = nn.Conv2d(
            args.n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

    # if the network is a XCiT, [yi:need to adjust for more in_channels]
    elif args.arch in torch.hub.list("facebookresearch/xcit:main"):
        student = torch.hub.load(
            "facebookresearch/xcit:main",
            args.arch,
            pretrained=False,
            drop_path_rate=args.drop_path_rate,
        )
        teacher = torch.hub.load(
            "facebookresearch/xcit:main", args.arch, pretrained=False
        )
        embed_dim = student.embed_dim
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(
        student,
        DINOHead(
            embed_dim,
            args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
        ),
    )
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False

    #

    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number
        + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params_groups, lr=0, momentum=0.9
        )  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = torch.amp.GradScaler(enabled=args.use_fp16)

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr
        * (args.batch_size_per_gpu * utils.get_world_size())
        / 256.0,  # linear scaling rule
        args.min_lr,
        args.epochs,
        len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs,
        len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(
        args.momentum_teacher, 1, args.epochs, len(data_loader)
    )
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    if args.resume:
        utils.restart_from_checkpoint(
            os.path.join(args.checkpoints_dir, "checkpoint.pth"),
            run_variables=to_restore,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            dino_loss=dino_loss,
        )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training !")

    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(
            student,
            teacher,
            teacher_without_ddp,
            dino_loss,
            data_loader,
            optimizer,
            lr_schedule,
            wd_schedule,
            momentum_schedule,
            epoch,
            fp16_scaler,
            args,
        )
        if args.wandb_log and utils.is_main_process():
            wandb.log(data={"epoch": epoch})
        # ============ writing logs ... ============
        save_dict = {
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "args": args,
            "dino_loss": dino_loss.state_dict(),
        }

        save_dict["fp16_scaler"] = fp16_scaler.state_dict()

        utils.save_on_master(
            save_dict, os.path.join(args.checkpoints_dir, "checkpoint.pth")
        )
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(
                save_dict,
                os.path.join(args.checkpoints_dir, f"checkpoint{epoch:04}.pth"),
            )
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }
        if utils.is_main_process():
            with (Path(args.checkpoints_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


def train_one_epoch(
    student,
    teacher,
    teacher_without_ddp,
    dino_loss,
    data_loader,
    optimizer,
    lr_schedule,
    wd_schedule,
    momentum_schedule,
    epoch,
    fp16_scaler,
    args,
):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Epoch: [{}/{}]".format(epoch, args.epochs)
    for it, images in enumerate(metric_logger.log_every(data_loader, 10, header)):

        # images = [torch.cat((images_s2[i],images_s1[i]),axis=1) for i in range(len(images_s2))]

        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        # teacher and student forward passes + compute dino loss
        with torch.amp.autocast(
            enabled=args.use_fp16, dtype=torch.bfloat16, device_type="cuda"
        ):
            teacher_output = teacher(
                images[:2]
            )  # only the 2 global views pass through the teacher
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None

        fp16_scaler.scale(loss).backward()
        if args.clip_grad:
            fp16_scaler.unscale_(
                optimizer
            )  # unscale the gradients of optimizer's assigned params in-place
            param_norms = utils.clip_gradients(student, args.clip_grad)
        utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
        fp16_scaler.step(optimizer)
        fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(
                student.module.parameters(), teacher_without_ddp.parameters()
            ):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        if args.wandb_log and utils.is_main_process():
            wandb.log(
                data={
                    "loss": loss.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "weight_decay": optimizer.param_groups[0]["weight_decay"],
                }
            )
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim,
        ncrops,
        warmup_teacher_temp,
        teacher_temp,
        warmup_teacher_temp_epochs,
        nepochs,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate(
            (
                np.linspace(
                    warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs
                ),
                np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
            )
        )

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = cvtransforms.Compose(
            [
                cvtransforms.RandomHorizontalFlip(p=0.5),
                cvtransforms.RandomApply(
                    [RandomBrightness(0.4), RandomContrast(0.4)], p=0.8
                ),
                cvtransforms.RandomApply([ToGray(14)], p=0.2),
            ]
        )
        normalize = cvtransforms.Compose(
            [
                cvtransforms.ToTensor(),
                # cvtransforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        sensor_drop = cvtransforms.RandomApply([RandomSensorDrop_S1S2()], p=0.5)

        # first global crop
        self.global_transfo1 = cvtransforms.Compose(
            [
                cvtransforms.RandomResizedCrop(
                    112, scale=global_crops_scale, interpolation="BICUBIC"
                ),
                flip_and_color_jitter,
                cvtransforms.RandomApply([GaussianBlur([0.1, 2.0])], p=1.0),
                normalize,
                sensor_drop,
            ]
        )
        # second global crop
        self.global_transfo2 = cvtransforms.Compose(
            [
                cvtransforms.RandomResizedCrop(
                    112, scale=global_crops_scale, interpolation="BICUBIC"
                ),
                flip_and_color_jitter,
                cvtransforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.1),
                cvtransforms.RandomApply([Solarize(128)], p=0.2),
                normalize,
                sensor_drop,
            ]
        )
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = cvtransforms.Compose(
            [
                cvtransforms.RandomResizedCrop(
                    48, scale=local_crops_scale, interpolation="BICUBIC"
                ),
                flip_and_color_jitter,
                cvtransforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                normalize,
                sensor_drop,
            ]
        )

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


class DataAugmentationDINO_S2(object):
    def __init__(
        self,
        in_size,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        season="fixed",
    ):
        flip_and_color_jitter = cvtransforms.Compose(
            [
                cvtransforms.RandomHorizontalFlip(p=0.5),
                cvtransforms.RandomApply(
                    [RandomBrightness(0.4), RandomContrast(0.4)], p=0.8
                ),
                cvtransforms.RandomApply([ToGray(13)], p=0.2),
            ]
        )
        normalize = cvtransforms.Compose(
            [
                cvtransforms.ToTensor(),
                # cvtransforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        # first global crop
        self.global_transfo1 = cvtransforms.Compose(
            [
                cvtransforms.RandomResizedCrop(
                    in_size, scale=global_crops_scale, interpolation="BICUBIC"
                ),
                flip_and_color_jitter,
                cvtransforms.RandomApply([GaussianBlur([0.1, 2.0])], p=1.0),
                normalize,
            ]
        )
        # second global crop
        self.global_transfo2 = cvtransforms.Compose(
            [
                cvtransforms.RandomResizedCrop(
                    in_size, scale=global_crops_scale, interpolation="BICUBIC"
                ),
                flip_and_color_jitter,
                cvtransforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.1),
                cvtransforms.RandomApply([Solarize(128)], p=0.2),
                normalize,
            ]
        )
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = cvtransforms.Compose(
            [
                cvtransforms.RandomResizedCrop(
                    96, scale=local_crops_scale, interpolation="BICUBIC"
                ),
                flip_and_color_jitter,
                cvtransforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                normalize,
            ]
        )

        self.season = season

    def __call__(self, image):

        if self.season == "augment":
            season1 = np.random.choice([0, 1, 2, 3])
            season2 = np.random.choice([0, 1, 2, 3])
            season3 = np.random.choice([0, 1, 2, 3])
        elif self.season == "fixed":
            np.random.seed(42)
            season1 = np.random.choice([0, 1, 2, 3])
            season2 = season1
            season3 = season1
        elif self.season == "random":
            season1 = np.random.choice([0, 1, 2, 3])
            season2 = season1
            season3 = season1

        x1 = np.transpose(image[season1, :, :, :], (1, 2, 0))
        x2 = np.transpose(image[season2, :, :, :], (1, 2, 0))
        x3 = np.transpose(image[season3, :, :, :], (1, 2, 0))

        crops = []
        crops.append(self.global_transfo1(x1))
        crops.append(self.global_transfo2(x2))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(x3))
        return crops


class DataAugmentationDINO_RGB(object):
    def __init__(
        self,
        in_size,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        season="fixed",
        strategy="single",
    ):
        """
        Args:
            in_size (int): Image input size.
            global_crops_scale (tuple): Scale range for global crops.
            local_crops_scale (tuple): Scale range for local crops.
            local_crops_number (int): Number of local crops per image.
            season (str): 'fixed', 'random', or 'augment' for seasonal selection.
            strategy (str): 'single' (select one season) or 'average' (combine all seasons).
        """
        self.strategy = strategy  # Multi-season strategy: 'single' or 'average'
        self.season = season  # Season selection strategy
        if self.strategy == "average":
            self.season = "fixed"
        flip_and_color_jitter = cvtransforms.Compose(
            [
                cvtransforms.RandomHorizontalFlip(p=0.5),
                cvtransforms.RandomApply(
                    [RandomBrightness(0.4), RandomContrast(0.4)], p=0.8
                ),
                cvtransforms.RandomApply([ToGray(3)], p=0.2),
            ]
        )
        normalize = cvtransforms.Compose(
            [
                cvtransforms.ToTensor(),
                # Uncomment below if ImageNet normalization is needed
                cvtransforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        # Global crops
        self.global_transfo1 = cvtransforms.Compose(
            [
                cvtransforms.RandomResizedCrop(
                    in_size, scale=global_crops_scale, interpolation="BICUBIC"
                ),
                flip_and_color_jitter,
                cvtransforms.RandomApply([GaussianBlur([0.1, 2.0])], p=1.0),
                normalize,
            ]
        )

        self.global_transfo2 = cvtransforms.Compose(
            [
                cvtransforms.RandomResizedCrop(
                    in_size, scale=global_crops_scale, interpolation="BICUBIC"
                ),
                flip_and_color_jitter,
                cvtransforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.1),
                cvtransforms.RandomApply([Solarize(128)], p=0.2),
                normalize,
            ]
        )

        # Local crops
        self.local_crops_number = local_crops_number
        self.local_transfo = cvtransforms.Compose(
            [
                cvtransforms.RandomResizedCrop(
                    96, scale=local_crops_scale, interpolation="BICUBIC"
                ),
                flip_and_color_jitter,
                cvtransforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                normalize,
            ]
        )

    def __call__(self, image):
        """
        Args:
            image (numpy.ndarray): Input RGB image in shape (T, 3, H, W), where T=4 is the number of seasonal images.

        Returns:
            List of augmented crops.
        """
        assert (
            image.shape[1] == 3
        ), f"Expected an RGB image with shape (T, 3, H, W), but got {image.shape}"

        # **Option 1: Single Season Selection**
        if self.strategy == "single":
            if self.season == "augment":
                season1, season2, season3 = np.random.choice(4, 3, replace=True)
            elif self.season == "fixed":
                np.random.seed(42)
                season1 = np.random.choice(4)
                season2, season3 = season1, season1
            elif self.season == "random":
                season1 = np.random.choice(4)
                season2, season3 = season1, season1
            else:
                raise ValueError(f"Invalid season setting: {self.season}")

            # Extract selected seasonal images and convert to (H, W, C)
            x1 = np.transpose(image[season1], (1, 2, 0))
            x2 = np.transpose(image[season2], (1, 2, 0))
            x3 = np.transpose(image[season3], (1, 2, 0))

        # **Option 2: Average Over All Seasons**
        elif self.strategy == "average":
            averaged_image = image.mean(
                axis=0
            )  # Mean across seasons -> shape (3, H, W)
            x1 = x2 = x3 = np.transpose(averaged_image, (1, 2, 0))  # (H, W, C)

        else:
            raise ValueError(
                f"Invalid strategy: {self.strategy}. Choose 'single' or 'average'."
            )

        # Apply transformations
        crops = []
        crops.append(self.global_transfo1(x1))
        crops.append(self.global_transfo2(x2))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(x3))

        return crops


class DataAugmentationDINO_S1(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = cvtransforms.Compose(
            [
                # cvtransforms.RandomHorizontalFlip(p=0.5),
                # cvtransforms.RandomApply([
                #    RandomBrightness(0.4),
                #    RandomContrast(0.4)
                # ], p=0.8),
                # cvtransforms.RandomApply([ToGray(2)], p=0.2),
            ]
        )
        normalize = cvtransforms.Compose(
            [
                cvtransforms.ToTensor(),
                # cvtransforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        # first global crop
        self.global_transfo1 = cvtransforms.Compose(
            [
                cvtransforms.RandomResizedCrop(
                    112, scale=global_crops_scale, interpolation="BICUBIC"
                ),
                # flip_and_color_jitter,
                # cvtransforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
                normalize,
            ]
        )
        # second global crop
        self.global_transfo2 = cvtransforms.Compose(
            [
                cvtransforms.RandomResizedCrop(
                    112, scale=global_crops_scale, interpolation="BICUBIC"
                ),
                # flip_and_color_jitter,
                # cvtransforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
                # cvtransforms.RandomApply([Solarize(128)], p=0.2),
                normalize,
            ]
        )
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = cvtransforms.Compose(
            [
                cvtransforms.RandomResizedCrop(
                    48, scale=local_crops_scale, interpolation="BICUBIC"
                ),
                # flip_and_color_jitter,
                # cvtransforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                normalize,
            ]
        )

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DINO", parents=[get_args_parser()])
    args = parser.parse_args()

    Path(args.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)
