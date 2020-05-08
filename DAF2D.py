import os
import pdb
import subprocess
import time
import types

import torch
import torchvision
from fastai import *
from fastai import vision
from fastai.basic_data import *
from fastai.basic_train import *
from fastai.callbacks import CSVLogger
from fastai.core import *
from fastai.torch_core import *
from fastai.vision import *
from fastai.vision.learner import cnn_config, create_head, num_features_model

# import torchsnooper
from IPython.core.debugger import set_trace
from sklearn.model_selection import KFold
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader

from BackBone3D import BackBone3D
from DAF3D import DAF3D, ASPP_module
from DataOperate import MySet, get_data_list
from Utils import DiceLoss, dice_ratio


class ASPP_module_2d(ASPP_module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module_2d, self).__init__(inplanes, planes, rate)
        rate_list = (1, rate)  # this ? 3d and 2d?
        self.atrous_convolution = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=1,
            padding=rate_list,
            dilation=rate_list,
        )
        self.group_norm = nn.GroupNorm(32, planes)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class AfterRefine(nn.Module):
    def __init__(self):
        super(AfterRefine, self).__init__()
        rates = (1, 6, 12, 18)
        self.aspp1 = ASPP_module_2d(64, 64, rate=rates[0])
        self.aspp2 = ASPP_module_2d(64, 64, rate=rates[1])
        self.aspp3 = ASPP_module_2d(64, 64, rate=rates[2])
        self.aspp4 = ASPP_module_2d(64, 64, rate=rates[3])

        self.aspp_conv = nn.Conv2d(256, 64, 1)
        self.aspp_gn = nn.GroupNorm(32, 64)

        self.predict = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, *input, **kwargs):
        refine = input[0]

        aspp1 = self.aspp1(refine)
        aspp2 = self.aspp2(refine)
        aspp3 = self.aspp3(refine)
        aspp4 = self.aspp4(refine)

        aspp = torch.cat((aspp1, aspp2, aspp3, aspp4), dim=1)

        aspp = self.aspp_gn(self.aspp_conv(aspp))

        predict = self.predict(aspp)
        return predict


class DAF2D(DAF3D):
    def __init__(self, body):
        super(DAF2D, self).__init__()
        self.backbone = BackBone2D(body)

        self.down4 = nn.Sequential(
            nn.Conv2d(2048, 128, kernel_size=1), nn.GroupNorm(
                32, 128), nn.PReLU(),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, 128, kernel_size=1), nn.GroupNorm(
                32, 128), nn.PReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1), nn.GroupNorm(
                32, 128), nn.PReLU()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1), nn.GroupNorm(
                32, 128), nn.PReLU()
        )

        self.fuse1 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1),
            nn.GroupNorm(32, 64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(32, 64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(32, 64),
            nn.PReLU(),
        )
        self.attention4 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=1),
            nn.GroupNorm(32, 64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(32, 64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        self.attention3 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=1),
            nn.GroupNorm(32, 64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(32, 64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.attention2 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=1),
            nn.GroupNorm(32, 64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(32, 64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.attention1 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=1),
            nn.GroupNorm(32, 64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(32, 64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        self.refine4 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=1),
            nn.GroupNorm(32, 64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(32, 64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(32, 64),
            nn.PReLU(),
        )
        self.refine3 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=1),
            nn.GroupNorm(32, 64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(32, 64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(32, 64),
            nn.PReLU(),
        )
        self.refine2 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=1),
            nn.GroupNorm(32, 64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(32, 64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(32, 64),
            nn.PReLU(),
        )
        self.refine1 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=1),
            nn.GroupNorm(32, 64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(32, 64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(32, 64),
            nn.PReLU(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1), nn.GroupNorm(
                32, 64), nn.PReLU(),
        )
        self.after_refine = AfterRefine()

        self.predict1_4 = nn.Conv2d(128, 1, kernel_size=1)
        self.predict1_3 = nn.Conv2d(128, 1, kernel_size=1)
        self.predict1_2 = nn.Conv2d(128, 1, kernel_size=1)
        self.predict1_1 = nn.Conv2d(128, 1, kernel_size=1)

        self.predict2_4 = nn.Conv2d(64, 1, kernel_size=1)
        self.predict2_3 = nn.Conv2d(64, 1, kernel_size=1)
        self.predict2_2 = nn.Conv2d(64, 1, kernel_size=1)
        self.predict2_1 = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        layer0 = self.backbone.layer0(x)
        layer1 = self.backbone.layer1(layer0)
        layer2 = self.backbone.layer2(layer1)
        layer3 = self.backbone.layer3(layer2)
        layer4 = self.backbone.layer4(layer3)

        # Top-down
        down4 = self.down4(layer4)
        down3 = torch.add(
            F.interpolate(down4, size=layer3.size()[2:], mode="bilinear"),
            self.down3(layer3),
        )
        down2 = torch.add(
            F.interpolate(down3, size=layer2.size()[2:], mode="bilinear"),
            self.down2(layer2),
        )
        down1 = torch.add(
            F.interpolate(down2, size=layer1.size()[2:], mode="bilinear"),
            self.down1(layer1),
        )
        down4 = F.interpolate(down4, size=layer1.size()[2:], mode="bilinear")
        down3 = F.interpolate(down3, size=layer1.size()[2:], mode="bilinear")
        down2 = F.interpolate(down2, size=layer1.size()[2:], mode="bilinear")

        predict1_4 = self.predict1_4(down4)
        predict1_3 = self.predict1_3(down3)
        predict1_2 = self.predict1_2(down2)
        predict1_1 = self.predict1_1(down1)

        fuse1 = self.fuse1(torch.cat((down4, down3, down2, down1), 1))

        attention4 = self.attention4(torch.cat((down4, fuse1), 1))
        attention3 = self.attention3(torch.cat((down3, fuse1), 1))
        attention2 = self.attention2(torch.cat((down2, fuse1), 1))
        attention1 = self.attention1(torch.cat((down1, fuse1), 1))

        refine4 = self.refine4(torch.cat((down4, attention4 * fuse1), 1))
        refine3 = self.refine3(torch.cat((down3, attention3 * fuse1), 1))
        refine2 = self.refine2(torch.cat((down2, attention2 * fuse1), 1))
        refine1 = self.refine1(torch.cat((down1, attention1 * fuse1), 1))

        refine = self.refine(
            torch.cat((refine1, refine2, refine3, refine4), 1))
        predict_undersample = self.after_refine(refine)

        predict2_4 = self.predict2_4(refine4)
        predict2_3 = self.predict2_3(refine3)
        predict2_2 = self.predict2_2(refine2)
        predict2_1 = self.predict2_1(refine1)

        predict1_1 = F.interpolate(
            predict1_1, size=x.size()[2:], mode="bilinear")
        predict1_2 = F.interpolate(
            predict1_2, size=x.size()[2:], mode="bilinear")
        predict1_3 = F.interpolate(
            predict1_3, size=x.size()[2:], mode="bilinear")
        predict1_4 = F.interpolate(
            predict1_4, size=x.size()[2:], mode="bilinear")

        predict2_1 = F.interpolate(
            predict2_1, size=x.size()[2:], mode="bilinear")
        predict2_2 = F.interpolate(
            predict2_2, size=x.size()[2:], mode="bilinear")
        predict2_3 = F.interpolate(
            predict2_3, size=x.size()[2:], mode="bilinear")
        predict2_4 = F.interpolate(
            predict2_4, size=x.size()[2:], mode="bilinear")

        predict = F.interpolate(predict_undersample, size=x.size()[
                                2:], mode="bilinear")
        if self.training:
            return (
                predict1_1,
                predict1_2,
                predict1_3,
                predict1_4,
                predict2_1,
                predict2_2,
                predict2_3,
                predict2_4,
                predict,
            )
        else:
            return predict


class BackBone2D(BackBone3D):
    def __init__(self, body):
        super(BackBone2D, self).__init__()
        net = body
        # resnext3d-101 is [3, 4, 23, 3]
        # we use the resnet3d-50 with [3, 4, 6, 3] blocks
        # and if we use the resnet3d-101, change the block list with [3, 4, 23, 3]
        net = list(net.children())
        self.layer0 = nn.Sequential(*net[:3])
        # the layer0 contains the first convolution, bn and relu
        self.layer1 = nn.Sequential(*net[3:5])
        # the layer1 contains the first pooling and the first 3 bottle blocks
        self.layer2 = net[5]
        # the layer2 contains the second 4 bottle blocks
        self.layer3 = net[6]
        # the layer3 contains the media bottle blocks
        # with 6 in 50-layers and 23 in 101-layers
        self.layer4 = net[7]
        # the layer4 contains the final 3 bottle blocks
        # according the backbone the next is avg-pooling and dense with num classes uints
        # but we don't use the final two layers in backbone networks


def PSLearner(
    data: DataBunch,
    base_arch: Callable,
    cut: Union[int, Callable] = None,
    pretrained: bool = True,
    lin_ftrs: Optional[Collection[int]] = None,
    ps: Floats = 0.5,
    custom_head: Optional[nn.Module] = None,
    split_on: Optional[SplitFuncOrIdxList] = None,
    bn_final: bool = False,
    init=nn.init.kaiming_normal_,
    concat_pool: bool = True,
    attention=False,
    **kwargs: Any
) -> Learner:
    "Build convnet style learner."
    meta = cnn_config(base_arch)

    "Create custom convnet architecture"
    body = create_body(base_arch, pretrained, cut)
    modified = DAF2D(body)

    # if custom_head is None:  # quadnet head
    #    nf = num_features_model(nn.Sequential(*body.children())) * (2 if concat_pool else 1)
    #    if attention:
    #        pass
    #    head = create_head(nf, data.c+2, lin_ftrs, ps=ps, concat_pool=concat_pool, bn_final=bn_final)
    # else:
    #    head = custom_head

    # model = nn.Sequential(body, head)
    nets = list(modified.children())

    learn = Learner(data, modified, **kwargs)

    add_custom_learner_function(learn)

    learn.split((nets[1],))

    if pretrained:
        learn.freeze_backbone()

    if init:
        for g in learn.layer_groups[1:]:
            chd = children(g)
            for l in chd:
                if (
                    isinstance(l, ASPP_module_2d)
                    or isinstance(l, ASPP_module)
                    or isinstance(l, nn.GroupNorm)
                    or isinstance(l, nn.PReLU)
                ):
                    continue
                apply_init(l, init)
    return learn


def update_instance_function(func, instance, func_name_of_instance):
    attr = instance.__setattr__(
        func_name_of_instance, types.MethodType(func, instance))


def freeze_backbone(self) -> None:
    "Freeze first layer group."
    assert len(self.layer_groups) > 1
    self.freeze_to(1)


def add_custom_learner_function(learn):
    update_instance_function(freeze_backbone, learn, "freeze_backbone")


class PSKernel:
    def __init__(self, stage="dev"):
        super(PSKernel, self).__init__()
        # The original images, provided in this competition, have 1024x1024 resolution. To prevent additional overhead on image loading, the datasets composed of 128x128 and 256x256 scaled down images are prepared separately and used as an input. Check [this keknel](https://www.kaggle.com/iafoss/data-repack-and-image-statistics) for more details on image rescaling and mask generation. Also In that kernel I apply image normalization based on histograms (exposure.equalize_adapthist) that provides some improvement of image appearance as well as a small boost of the model performance. The corresponding pixel statistics are computed in the kernel.
        self.sz = 256
        self.bs = 16
        self.n_acc = 64 // self.bs  # gradinet accumulation steps
        self.nfolds = 4
        self.stage_2 = False

        self.SEED = 2019
        self.stage = stage

        # eliminate all predictions with a few (self.noise_th) pixesls
        self.noise_th = (
            75.0 * (self.sz / 128.0) ** 2
        )  # threshold for the number of predicted pixels
        self.best_thr0 = (
            0.2  # preliminary value of the threshold for metric calculation
        )
        self.best_thr0_sigmoid = (
            0.6  # preliminary value of the threshold for metric calculation
        )

        if self.sz == 256 and self.stage_2:
            self.stats = ([0.540, 0.540, 0.540], [0.264, 0.264, 0.264])
            self.TRAIN = "../input/siimacr-pneumothorax-segmentation-data-256/train"
            # self.TRAIN = "../input/siim-acr-pneumothorax-segmentation/stage_2_train.csv"
            self.TEST = "../input/siim-acr-pneumothorax-segmentation/stage_2_sample_submission.csv"
            self.MASKS = "../input/siimacr-pneumothorax-segmentation-data-256/masks"
        elif self.sz == 256:
            self.stats = ([0.540, 0.540, 0.540], [0.264, 0.264, 0.264])
            self.TRAIN = "../input/siimacr-pneumothorax-segmentation-data-256/train"
            self.TEST = "../input/siimacr-pneumothorax-segmentation-data-256/test"
            self.MASKS = "../input/siimacr-pneumothorax-segmentation-data-256/masks"
        elif self.sz == 128:
            self.stats = ([0.615, 0.615, 0.615], [0.291, 0.291, 0.291])
            self.TRAIN = "../input/siimacr-pneumothorax-segmentation-data-128/train"
            self.TEST = "../input/siimacr-pneumothorax-segmentation-data-128/test"
            self.MASKS = "../input/siimacr-pneumothorax-segmentation-data-128/masks"

        self.best_thr = 0.21

    def _get_fold_data(self, fold):
        kf = KFold(n_splits=self.nfolds, shuffle=True, random_state=self.SEED)
        valid_idx = list(kf.split(list(range(len(Path(self.TRAIN).ls())))))[
            fold][1]
        # Create databunch
        data = (
            SegmentationItemList.from_folder(self.TRAIN)
            .split_by_idx(valid_idx)
            .label_from_func(lambda x: str(x).replace("train", "masks"), classes=[0, 1])
            .transform(get_transforms(), size=self.sz, tfm_y=True)
            .databunch(path=Path("."), bs=self.bs)
            .normalize(self.stats)
        )
        return data

    # A slight modification of the default dice metric to make it comparable with the competition metric: dice is computed for each image independently, and dice of empty image with zero prediction is 1. Also I use noise removal and similar threshold as in my prediction pipline.
    def _dice(
        self,
        input: Tensor,
        targs: Tensor,
        thr: float = None,
        iou: bool = False,
        eps: float = 1e-8,
    ) -> Rank0Tensor:
        n = targs.shape[0]
        input = input[-1]
        input = torch.sigmoid(input).view(n, -1)
        # input = torch.softmax(input, dim=1)[:, 0, ...].view(n, -1)
        if thr:
            input = (input > thr).long()
        else:
            input = (input > self.best_thr0_sigmoid).long()

        input[input.sum(-1) < self.noise_th, ...] = 0.0
        # input = input.argmax(dim=1).view(n,-1)
        targs = targs.view(n, -1)
        intersect = (input * targs).sum(-1).float()
        union = (input + targs).sum(-1).float()
        if not iou:
            return ((2.0 * intersect + eps) / (union + eps)).mean()
        else:
            return ((intersect + eps) / (union - intersect + eps)).mean()


class PSMultiLoss(nn.Module):
    def __init__(self, gamma: float = 0.0, eps=1e-7, from_logits=True, cls_cnt=None):
        super(PSMultiLoss, self).__init__()

        self.criterion_bce = torch.nn.BCELoss()
        self.criterion_dice = DiceLoss()
        self.eps = 1e-9

    def forward(self, input, target, **kwargs):
        # return predict1_1, predict1_2, predict1_3, predict1_4, \
        #       predict2_1, predict2_2, predict2_3, predict2_4, predict
        (
            outputs1,
            outputs2,
            outputs3,
            outputs4,
            outputs1_1,
            outputs1_2,
            outputs1_3,
            outputs1_4,
            output,
        ) = input

        label = target.float()
        label = label.clamp(self.eps, 1.0 - self.eps)

        output = torch.sigmoid(output)
        outputs1 = torch.sigmoid(outputs1)
        outputs2 = torch.sigmoid(outputs2)
        outputs3 = torch.sigmoid(outputs3)
        outputs4 = torch.sigmoid(outputs4)
        outputs1_1 = torch.sigmoid(outputs1_1)
        outputs1_2 = torch.sigmoid(outputs1_2)
        outputs1_3 = torch.sigmoid(outputs1_3)
        outputs1_4 = torch.sigmoid(outputs1_4)

        loss0_bce = self.criterion_bce(output, label)
        loss1_bce = self.criterion_bce(outputs1, label)
        loss2_bce = self.criterion_bce(outputs2, label)
        loss3_bce = self.criterion_bce(outputs3, label)
        loss4_bce = self.criterion_bce(outputs4, label)
        loss5_bce = self.criterion_bce(outputs1_1, label)
        loss6_bce = self.criterion_bce(outputs1_2, label)
        loss7_bce = self.criterion_bce(outputs1_3, label)
        loss8_bce = self.criterion_bce(outputs1_4, label)

        loss0_dice = self.criterion_dice(output, label)
        loss1_dice = self.criterion_dice(outputs1, label)
        loss2_dice = self.criterion_dice(outputs2, label)
        loss3_dice = self.criterion_dice(outputs3, label)
        loss4_dice = self.criterion_dice(outputs4, label)
        loss5_dice = self.criterion_dice(outputs1_1, label)
        loss6_dice = self.criterion_dice(outputs1_2, label)
        loss7_dice = self.criterion_dice(outputs1_3, label)
        loss8_dice = self.criterion_dice(outputs1_4, label)

        loss = (
            loss0_bce
            + 0.4 * loss1_bce
            + 0.5 * loss2_bce
            + 0.7 * loss3_bce
            + 0.8 * loss4_bce
            + 0.4 * loss5_bce
            + 0.5 * loss6_bce
            + 0.7 * loss7_bce
            + 0.8 * loss8_bce
            + loss0_dice
            + 0.4 * loss1_dice
            + 0.5 * loss2_dice
            + 0.7 * loss3_dice
            + 0.8 * loss4_dice
            + 0.4 * loss5_dice
            + 0.7 * loss6_dice
            + 0.8 * loss7_dice
            + 1 * loss8_dice
        )

        return loss
        # epoch_loss += loss.item()


def mask2rle(img, width, height):
    rle = []
    lastColor = 0
    currentPixel = 0
    runStart = -1
    runLength = 0

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel
                    runLength = 1
                else:
                    rle.append(str(runStart))
                    rle.append(str(runLength))
                    runStart = -1
                    runLength = 0
                    currentPixel = 0
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor
            currentPixel += 1

    return " ".join(rle)


def rle2mask(rle, width, height):
    mask = np.zeros(width * height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position: current_position + lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)
