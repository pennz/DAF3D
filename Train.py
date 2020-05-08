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
from DAF2D import PSKernel, mask2rle, PSLearner, PSMultiLoss
from DataOperate import MySet, get_data_list
from Utils import DiceLoss, dice_ratio
import utils


to_train = False
debug_trace = False

os.chdir("..")

if __name__ == "__main__":
    print(os.listdir("../input"))

    k = PSKernel()
    data = k._get_fold_data(0)

    learner = PSLearner(
        data,
        torchvision.models.resnext50_32x4d,
        cut=8,  # only used up to 7
        # loss_func=PSMultiLoss(),
        metrics=[
            partial(k._dice, thr=0.5),
            partial(k._dice, thr=0.6),
            partial(k._dice, thr=0.7),
            partial(k._dice, thr=0.8),
            partial(k._dice, thr=0.9),
        ],
        loss_func=PSMultiLoss(),
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

        kf = KFold(n_splits=k.nfolds, shuffle=True, random_state=k.SEED)
        valid_idx = list(kf.split(list(range(len(Path(k.TRAIN).ls())))))[0][1]

        def _get_test_stage2():  # just use database for stage two. Faster!!!
            pass

        k.learn.data = (
            SegmentationItemList.from_folder(k.TRAIN)
            .split_by_idx(valid_idx)  # will use k.nfolds
            .label_from_func(lambda x: str(x).replace("train", "masks"), classes=[0, 1])
            .add_test(Path(k.TEST).ls()[:32], label=None)
            .databunch(path=Path("."), bs=k.bs)
            .normalize(k.stats)
        )

        print(k.learn.data)
        utils.logger.debug(k.learn.data)
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
        sub_df.to_csv("submission.csv", index=False)

        sub_df_noT = pd.DataFrame({"ImageId": ids, "EncodedPixels": rlesNoT})
        sub_df_noT.loc[sub_df.EncodedPixels == "", "EncodedPixels"] = "-1"
        sub_df_noT.to_csv("submission_not_T.csv", index=False)

        print(sub_df.head())

    #    preds, ys = k.learn.get_preds(DatasetType.Valid)
    #    # preds.shape
    #    preds.max()
    #    data.valid_ds[3][0].data
    #    outprobs = torch.sigmoid(preds)
    #    ys.shape
    #    data
    #    # !head -n 6 submission_not_T_.csv submission_.csv
    #    # !gdrive upload submission_not_T_.csv
    #    ys.shape
    #    ys.sum((2, 3)).sort(descending=True, dim=0).indices
    #    kdata.valid_ds[987][0]
    #    plot_idx[1]
    #    kdata.valid_ds[987][1].data.dtype
    #    ys[987].data.numpy().transpose(1, 2, 0).shape
    #    outprobs_h = outprobs > 0.9
    #    outprobs_h.max()
    #    rows_start = 1500
    #    rows_end = 1520
    #    plot_idx = (
    #        ys.sum((2, 3)).sort(descending=True,
    #                            dim=0).indices[rows_start:rows_end]
    #    )
    #    # print(plot_idx)
    #  for idx in plot_idx:
    #      idd = idx.data.numpy()[0]
    #      fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(12, 4))
    #      ax0.imshow(kdata.valid_ds[idd][0].data.numpy().transpose(1, 2, 0))
    #      ax1.imshow(
    #          ys[idd].data.numpy().transpose(1, 2, 0).reshape(256, 256),
    #          vmin=0,
    #          vmax=1,
    #      )
    #      ax2.imshow(
    #          outprobs_h[idd].data.numpy().transpose(
    #              1, 2, 0).reshape(256, 256),
    #          vmin=0,
    #          vmax=1,
    #      )
    #      ax1.set_title("Targets")
    #      ax2.set_title("Predictions")
# -

# gdrive upload submission_not_T_.csv
# ls ../input/siim-acr-pneumothorax-segmentation
# !pip install kaggle
# !mkdir $HOME/.kaggle
# !echo '{"username":"k1gaggle","key":"f51513f40920d492e9d81bc670b25fa9"}' > $HOME/.kaggle/kaggle.json
# !chmod 600 $HOME/.kaggle/kaggle.json

# !kaggle competitions submit  -f submission_not_T_.csv -m 'daf_no_tta_0.9_th_first_100_noiseth_no_T' siim-acr-pneumothorax-segmentation


# !kaggle competitions submissions siim-acr-pneumothorax-segmentation
