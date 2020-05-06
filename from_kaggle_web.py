# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5"
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import os
import pdb

# +
# # %load ../input/siim-acr-pneumothorax-segmentation/mask_functions.py
import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# %reload_ext autoreload
# %autoreload 2

# #!wget http://23.105.212.181:8000/daf.tar.gz -O daf.tar.gz
# #!tar xvf daf.tar.gz
# -

# +
# ! git clone --depth=1 https://github.com/pennz/DAF3D

# ! (cd DAF3D & & mv * .* ..)

# -

# %run Train.py ## it hsoud create the learner


# +
# !./gdrive download 1meRU32M5cm0RXDI2_GaeDCHaGvISDCZE
# -

# !gdrive download 11XnBpIo8bEofmKLuJLxy3nJo4H5dkNfB

# !gdrive list

# !mkdir models ; mv *pth models

# #!wget https://www.kaggleusercontent.com/kf/17274750/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..iW7qeXH6bvVzU4hfnmXOBA.OnKmLhEQZiG7BQC0speCnPF-mbQc0iuxXpc-gEmSJ-X9WKniYzV1X9pSKvO1EM9LhCrCwGY6RuBUdlIfZ7xk39eQlHWUlpfcPV7c_30clh-FAbgxuohA8a7ld1r3bj1N.PIYIts3bSaPNlNf2XrRkAA/models/dr-stage1_2.pth


learner.load('dr-stage1_2')
k.learn.model.eval()

k.learn.data

k.learn = learner  # !!! set learner here

kf = KFold(n_splits=k.nfolds, shuffle=True, random_state=k.SEED)
valid_idx = list(kf.split(list(range(len(Path(k.TRAIN).ls())))))[0][1]
k.learn.data = (SegmentationItemList.from_folder(k.TRAIN)
                .split_by_idx(valid_idx)
                .label_from_func(lambda x: str(x).replace('train', 'masks'), classes=[0, 1])
                .add_test(Path(k.TEST).ls(), label=None)
                .databunch(path=Path('.'), bs=k.bs)
                .normalize(k.stats))


k.learn.data

k.learn.model.eval()

output_WIP = k.learn.get_preds(DatasetType.Test, with_loss=False)


def predict_on_test_post_process(self, output, thr=None):  # monkey-patch
    # ### Submission
    # convert predictions to byte type and save
    output_prob = torch.sigmoid(output)

    preds_save = (output_prob * 255.0).byte()
    torch.save(preds_save, 'preds_test.pt')

    output_WIP = output_prob

    # Generate rle encodings (images are first converted to the original size)
    thr = self.best_thr if thr is None else thr
    output_WIP = (output_WIP > thr).int()

    # If any pixels are predicted for an empty mask, the corresponding image gets zero score during evaluation. While prediction of no pixels for an empty mask gives perfect score. Because of this penalty it is resonable to set masks to zero if the number of predicted pixels is small. This trick was quite efective in [Airbus Ship Detection Challenge](https://www.kaggle.com/iafoss/unet34-submission-tta-0-699-new-public-lb).
    output_WIP[output_WIP.view(
        output_WIP.shape[0], -1).sum(-1) < self.noise_th, ...] = 0.0

    # output_WIP = output_WIP.numpy()

    rles = []
    rlesNoT = []
    t = torchvision.transforms.ToPILImage()
    for p in progress_bar(output_WIP):
        im = t(p)
        # p_1_channel=(p.T * 255).astype(np.uint8)
        im = im.resize((1024, 1024))
        im = np.asarray(im)
        if debug_trace:
            set_trace()
        im = im*255  # the mask2rle will use this

        im_4_rle = np.copy(np.transpose(im))
        if debug_trace:
            set_trace()
        rles.append(mask2rle(im_4_rle, 1024, 1024))
        rlesNoT.append(mask2rle(im, 1024, 1024))
        del im
        del im_4_rle

    ids = [o.stem for o in self.learn.data.test_ds.items]
    sub_df = pd.DataFrame({'ImageId': ids, 'EncodedPixels': rles})
    sub_df.loc[sub_df.EncodedPixels == '', 'EncodedPixels'] = '-1'
    sub_df.to_csv('submission_.csv', index=False)

    sub_df_noT = pd.DataFrame({'ImageId': ids, 'EncodedPixels': rlesNoT})
    sub_df_noT.loc[sub_df.EncodedPixels == '', 'EncodedPixels'] = '-1'
    sub_df_noT.to_csv('submission_not_T_.csv', index=False)


print(sub_df.head())


k.preds_test.shape


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
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)

# -

# ls ../input/siim-acr-pneumothorax-segmentation


sys.path.insert(0, '../input/siim-acr-pneumothorax-segmentation')

debug_trace = False

# !head -n 3  submissio*.csv

predict_on_test_post_process(k, output_WIP[0], thr=0.9)

# !head -n 3 submission_not_T.csv

# !head b

# !diff submission.csv submission_not_T.csv

!./gdrive upload submission_not_T_.csv

pdb.pm()

#

# ls

ddd = torch.load('preds_test.pt')

ddd

preds_on_test, _ = k.learn.get_preds(DatasetType.Test, with_loss=False)

preds_on_test.min()


# !find ../input -name "util*"

# ls -lh

k.data0 = k.learn.data

predict_on_test_post_process(k, preds_on_test, 0.7)

pdb.pm()

# !pip install kaggle
# !mkdir $HOME/.kaggle
# !echo '{"username":"k1gaggle","key":"f51513f40920d492e9d81bc670b25fa9"}' > $HOME/.kaggle/kaggle.json
# !chmod 600 $HOME/.kaggle/kaggle.json

# !kaggle competitions submit  -f submission_not_T_.csv -m 'daf_no_tta_0.9_th_first_100_noiseth_no_T' siim-acr-pneumothorax-segmentation


# !kaggle competitions submissions siim-acr-pneumothorax-segmentation

8006/2669

preds, ys = k.learn.get_preds(DatasetType.Valid)

preds.shape

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

outprobs_h = (outprobs > 0.9)

outprobs_h.max()

rows_start = 1500
rows_end = 1520
plot_idx = ys.sum((2, 3)).sort(
    descending=True, dim=0).indices[rows_start:rows_end]
# print(plot_idx)
for idx in plot_idx:
    idd = idx.data.numpy()[0]
    fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(12, 4))
    ax0.imshow(kdata.valid_ds[idd][0].data.numpy().transpose(1, 2, 0))
    ax1.imshow(ys[idd].data.numpy().transpose(
        1, 2, 0).reshape(256, 256), vmin=0, vmax=1)
    ax2.imshow(outprobs_h[idd].data.numpy().transpose(
        1, 2, 0).reshape(256, 256), vmin=0, vmax=1)
    ax1.set_title('Targets')
    ax2.set_title('Predictions')

# pdb.pm()
