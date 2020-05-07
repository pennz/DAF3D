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


def ps_learner(
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


class psKernel:
    def __init__(self, stage="dev"):
        super(psKernel, self).__init__()
        # The original images, provided in this competition, have 1024x1024 resolution. To prevent additional overhead on image loading, the datasets composed of 128x128 and 256x256 scaled down images are prepared separately and used as an input. Check [this keknel](https://www.kaggle.com/iafoss/data-repack-and-image-statistics) for more details on image rescaling and mask generation. Also In that kernel I apply image normalization based on histograms (exposure.equalize_adapthist) that provides some improvement of image appearance as well as a small boost of the model performance. The corresponding pixel statistics are computed in the kernel.
        self.sz = 256
        self.bs = 16
        self.n_acc = 64 // self.bs  # gradinet accumulation steps
        self.nfolds = 4

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

        if self.sz == 256:
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


class ps_multi_Loss(nn.Module):
    def __init__(self, gamma: float = 0.0, eps=1e-7, from_logits=True, cls_cnt=None):
        super(ps_multi_Loss, self).__init__()

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


# -


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


to_train = False
debug_trace = False


if __name__ == "__main__":

    print(os.listdir("../input"))

    k = psKernel()
    data = k._get_fold_data(0)

    learner = ps_learner(
        data,
        torchvision.models.resnext50_32x4d,
        cut=8,  # only used up to 7
        # loss_func=ps_multi_Loss(),
        metrics=[
            partial(k._dice, thr=0.5),
            partial(k._dice, thr=0.6),
            partial(k._dice, thr=0.7),
            partial(k._dice, thr=0.8),
            partial(k._dice, thr=0.9),
        ],
        loss_func=ps_multi_Loss(),
        callback_fns=[partial(CSVLogger, append=True)],
    )

    # learner.summary()

    # torch.cuda.set_device(0)
    # net = DAF3D().cuda()
    # learner.lr_find()  # for freezed one, max_lr=1e-2

    # for stage 2  slice(1e-6, 1e-2/10)

    if to_train:
        # learner.recorder.plot(suggestion=True)
        # learner.recorder.plot(suggestion=True)
        s1_lr = 1e-4

        learner.freeze()
        learner.fit_one_cycle(4, max_lr=s1_lr, wd=0.1)
        learner.recorder.plot_losses()
        learner.recorder.plot_metrics()
        learner.save("dr-stage1")

        learner.unfreeze()
        # learner.lr_find()
        # learner.recorder.plot(suggestion=True)

        learner.fit_one_cycle(6, max_lr=slice(1e-6, s1_lr / 5), wd=0.1)

        learner.recorder.plot_losses()
        learner.recorder.plot_metrics()
        learner.save("dr-stage2")

        learner.freeze()
        learner.fit_one_cycle(4, max_lr=s1_lr / 2, wd=0.1)
        learner.recorder.plot_losses()
        learner.recorder.plot_metrics()
        learner.save("dr-stage1_2")

        learner.unfreeze()
        # learner.lr_find()
        # learner.recorder.plot(suggestion=True)

        learner.fit_one_cycle(6, max_lr=slice(1e-6, s1_lr / 10), wd=0.1)

        learner.recorder.plot_losses()
        learner.recorder.plot_metrics()
        learner.save("dr-stage2_2")

        learner.freeze()
        learner.fit_one_cycle(4, max_lr=s1_lr / 4, wd=0.1)
        learner.recorder.plot_losses()
        learner.recorder.plot_metrics()
        learner.save("dr-stage1_2")

        learner.unfreeze()
        # learner.lr_find()
        # learner.recorder.plot(suggestion=True)

        learner.fit_one_cycle(6, max_lr=slice(1e-6, s1_lr / 20), wd=0.1)

        learner.recorder.plot_losses()
        learner.recorder.plot_metrics()
        learner.save("dr-stage2_2")

        learner.freeze()
        learner.fit_one_cycle(4, max_lr=s1_lr / 5, wd=0.1)
        learner.recorder.plot_losses()
        learner.recorder.plot_metrics()
        learner.save("dr-stage1_2")

        learner.unfreeze()
        # learner.lr_find()
        # learner.recorder.plot(suggestion=True)

        learner.fit_one_cycle(6, max_lr=slice(1e-6, s1_lr / 25), wd=0.1)

        learner.recorder.plot_losses()
        learner.recorder.plot_metrics()
        learner.save("dr-stage2_2")
        learner.freeze()
        learner.fit_one_cycle(4, max_lr=s1_lr / 5, wd=0.1)
        learner.recorder.plot_losses()
        learner.recorder.plot_metrics()
        learner.save("dr-stage1_2")

        learner.unfreeze()
        # learner.lr_find()
        # learner.recorder.plot(suggestion=True)

        learner.fit_one_cycle(6, max_lr=slice(1e-6, s1_lr / 25), wd=0.1)

        learner.recorder.plot_losses()
        learner.recorder.plot_metrics()
        learner.save("dr-stage2_2")
        learner.freeze()
        learner.fit_one_cycle(4, max_lr=s1_lr / 5, wd=0.1)
        learner.recorder.plot_losses()
        learner.recorder.plot_metrics()
        learner.save("dr-stage1_2")

        learner.unfreeze()
        # learner.lr_find()
        # learner.recorder.plot(suggestion=True)

        learner.fit_one_cycle(6, max_lr=slice(1e-6, s1_lr / 25), wd=0.1)

        learner.recorder.plot_losses()
        learner.recorder.plot_metrics()
        learner.save("dr-stage2_2")
    # net = DAF3D()

    # optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    # train_list, test_list = get_data_list("Data/Original", ratio=0.8)

    # best_dice = 0.

    # if not os.path.exists("checkpoints"):
    #    os.mkdir("checkpoints")

    # information_line = '='*20 + ' DAF3D ' + '='*20 + '\n'
    # open('Log.txt', 'w').write(information_line)

    # train_set = MySet(train_list)
    # train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    # test_dataset = MySet(test_list)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # for epoch in range(1, 21):
    #    epoch_start_time = time.time()
    #    print("Epoch: {}".format(epoch))
    #    epoch_loss = 0.
    #    net.train()
    #    start_time = time.time()
    #    for batch_idx, (image, label) in enumerate(train_loader):
    #        image = Variable(image.cuda())
    #        label = Variable(label.cuda())

    #        optimizer.zero_grad()

    #    if batch_idx % 10 == 0:
    #        print_line = 'Epoch: {} | Batch: {} -----> Train loss: {:4f} Cost Time: {}\n' \
    #                     'Batch bce  Loss: {:4f} || ' \
    #                     'Loss1: {:4f}, Loss2: {:4f}, Loss3: {:4f}, Loss4: {:4f}, ' \
    #                     'Loss5: {:4f}, Loss6: {:4f}, Loss7: {:4f}, Loss8: {:4f}\n' \
    #                     'Batch dice Loss: {:4f} || ' \
    #                     'Loss1: {:4f}, Loss2: {:4f}, Loss3: {:4f}, Loss4: {:4f}, ' \
    #                     'Loss5: {:4f}, Loss6: {:4f}, Loss7: {:4f}, Loss8: {:4f}\n' \
    #            .format(epoch, batch_idx, epoch_loss / (batch_idx + 1), time.time() - start_time,
    #                    loss0_bce.item(), loss1_bce.item(), loss2_bce.item(), loss3_bce.item(), loss4_bce.item(),
    #                    loss5_bce.item(), loss6_bce.item(), loss7_bce.item(), loss8_bce.item(),
    #                    loss0_dice.item(), loss1_dice.item(), loss2_dice.item(), loss3_dice.item(),
    #                    loss4_dice.item(), loss5_dice.item(), loss6_dice.item(), loss7_dice.item(),
    #                    loss8_dice.item())
    #        print(print_line)
    #        start_time = time.time()

    #    loss.backward()
    #    optimizer.step()

    # print('Epoch {} Finished ! Loss is {:4f}'.format(epoch, epoch_loss / (batch_idx + 1)))
    # open('Log.txt', 'a') \
    # .write("Epoch {} Loss: {}".format(epoch, epoch_loss / (batch_idx + 1)))

    # print("Epoch time: ", time.time() - epoch_start_time)
    # begin to eval
    # net.eval()

    # dice = 0.

    # for batch_idx, (image, label) in enumerate(test_loader):
    #    image = Variable(image.cuda())
    #    label = Variable(label.cuda())

    #    predict = net(image)
    #    predict = F.sigmoid(predict)

    #    predict = predict.data.cpu().numpy()
    #    label = label.data.cpu().numpy()

    #    dice_tmp = dice_ratio(predict, label)
    #    dice = dice + dice_tmp

    # dice = dice / (1 + batch_idx)
    # print("Eva Dice Result: {}".format(dice))
    # open('Log.txt', 'a').write("Epoch {} Dice Score: {}\n".format(epoch, dice))

    # if dice > best_dice:
    #    best_dice = dice
    #    torch.save(net.state_dict(), 'checkpoints/Best_Dice.pth')

    # torch.save(net.state_dict(), 'checkpoints/model_{}.pth'.format(epoch))
    else:
        learner.load("dr-stage1_2")
        k.learn = learner  # !!! set learner here

        k.learn.model.eval()

        kf = KFold(n_splits=k.nfolds, shuffle=True, random_state=k.SEED)
        valid_idx = list(kf.split(list(range(len(Path(k.TRAIN).ls())))))[0][1]
        k.learn.data = (
            SegmentationItemList.from_folder(k.TRAIN)
            .split_by_idx(valid_idx)
            .label_from_func(lambda x: str(x).replace("train", "masks"), classes=[0, 1])
            .add_test(Path(k.TEST).ls()[:32], label=None)
            .databunch(path=Path("."), bs=k.bs)
            .normalize(k.stats)
        )

        k.learn.data
        k.learn.model.eval()

        output_WIP = k.learn.get_preds(DatasetType.Test, with_loss=False)

        sys.path.insert(0, "../input/siim-acr-pneumothorax-segmentation")

        # !head -n 3  submissio*.csv

        def predict_on_test_post_process(self, output, thr=None):  # monkey-patch
            # ### Submission
            # convert predictions to byte type and save
            output_prob = torch.sigmoid(output)

            # here it is changed to 255 range
            preds_save = (output_prob * 255.0).byte()
            torch.save(preds_save, "preds_test.pt")

            output_WIP = output_prob

            # Generate rle encodings (images are first converted to the original size)
            thr = self.best_thr if thr is None else thr
            output_WIP = (output_WIP > thr).int()

            # If any pixels are predicted for an empty mask, the corresponding image gets zero score during evaluation. While prediction of no pixels for an empty mask gives perfect score. Because of this penalty it is resonable to set masks to zero if the number of predicted pixels is small. This trick was quite efective in [Airbus Ship Detection Challenge](https://www.kaggle.com/iafoss/unet34-submission-tta-0-699-new-public-lb).
            output_WIP[
                output_WIP.view(
                    output_WIP.shape[0], -1).sum(-1) < self.noise_th, ...
            ] = 0.0
            return output_WIP

        # output_WIP = output_WIP.numpy()
        if debug_trace:
            set_trace()

        predict_on_test_post_process(k, output_WIP[0], thr=0.9)

        rles = []
        rlesNoT = []
        t = torchvision.transforms.ToPILImage()
        for p in progress_bar(output_WIP[0]):
            im = t(p)
            # p_1_channel=(p.T * 255).astype(np.uint8)
            im = im.resize((1024, 1024))
            im = np.asarray(im)
            if debug_trace:
                set_trace()
            # im = im * 255  # the mask2rle will use this-> It is 255 range already

            im_4_rle = np.copy(np.transpose(im))
            if debug_trace:
                set_trace()
            rles.append(mask2rle(im_4_rle, 1024, 1024))
            rlesNoT.append(mask2rle(im, 1024, 1024))
            del im
            del im_4_rle

        ids = [o.stem for o in k.learn.data.test_ds.items]
        sub_df = pd.DataFrame({"ImageId": ids, "EncodedPixels": rles})
        sub_df.loc[sub_df.EncodedPixels == "", "EncodedPixels"] = "-1"
        sub_df.to_csv("submission_.csv", index=False)

        sub_df_noT = pd.DataFrame({"ImageId": ids, "EncodedPixels": rlesNoT})
        sub_df_noT.loc[sub_df.EncodedPixels == "", "EncodedPixels"] = "-1"
        sub_df_noT.to_csv("submission_not_T_.csv", index=False)

        print(sub_df.head())

        k.preds_test.shape
        preds, ys = k.learn.get_preds(DatasetType.Valid)
        # preds.shape
        preds.max()
        data.valid_ds[3][0].data
        outprobs = torch.sigmoid(preds)
        ys.shape
        data
        # !head -n 6 submission_not_T_.csv submission_.csv
        # !gdrive upload submission_not_T_.csv
        ys.shape
        ys.sum((2, 3)).sort(descending=True, dim=0).indices
        kdata.valid_ds[987][0]
        plot_idx[1]
        kdata.valid_ds[987][1].data.dtype
        ys[987].data.numpy().transpose(1, 2, 0).shape
        outprobs_h = outprobs > 0.9
        outprobs_h.max()
        rows_start = 1500
        rows_end = 1520
        plot_idx = (
            ys.sum((2, 3)).sort(descending=True,
                                dim=0).indices[rows_start:rows_end]
        )
        # print(plot_idx)
        for idx in plot_idx:
            idd = idx.data.numpy()[0]
            fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(12, 4))
            ax0.imshow(kdata.valid_ds[idd][0].data.numpy().transpose(1, 2, 0))
            ax1.imshow(
                ys[idd].data.numpy().transpose(1, 2, 0).reshape(256, 256),
                vmin=0,
                vmax=1,
            )
            ax2.imshow(
                outprobs_h[idd].data.numpy().transpose(
                    1, 2, 0).reshape(256, 256),
                vmin=0,
                vmax=1,
            )
            ax1.set_title("Targets")
            ax2.set_title("Predictions")
# -

# gdrive upload submission_not_T_.csv
# ls ../input/siim-acr-pneumothorax-segmentation
# !pip install kaggle
# !mkdir $HOME/.kaggle
# !echo '{"username":"k1gaggle","key":"f51513f40920d492e9d81bc670b25fa9"}' > $HOME/.kaggle/kaggle.json
# !chmod 600 $HOME/.kaggle/kaggle.json

# !kaggle competitions submit  -f submission_not_T_.csv -m 'daf_no_tta_0.9_th_first_100_noiseth_no_T' siim-acr-pneumothorax-segmentation


# !kaggle competitions submissions siim-acr-pneumothorax-segmentation
