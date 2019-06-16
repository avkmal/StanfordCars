## Training code
from fastai import *
from fastai.vision import *
import pandas as pd

labels_df = pd.read_csv('labels.csv')

path = 'data/'

SZ = 224
LABEL = 'class_name'

car_tfms = get_transforms()

trn_labels_df = labels_df.loc[labels_df['is_test']==0, ['filename', 'class_name', 'class_id']].copy()

src = (ImageList.from_df(trn_labels_df, path, folder='cars_train', cols='filename')
                    .split_by_rand_pct(valid_pct=0.2, seed=100)
                    .label_from_df(cols=LABEL))

data = (src.transform(car_tfms, size=SZ)
            .databunch()
            .normalize(imagenet_stats))

## stage 1

arch = models.resnet152
data.batch_size = 32
learn = cnn_learner(data, arch, metrics=[accuracy])
lr = 1e-2
learn.fit_one_cycle(16, max_lr = lr)

## stage 2

learn.unfreeze()
data.batch_size = 16
lr = 1e-3
lrs = np.array([lr/9,lr/6,lr])
learn.fit_one_cycle(15, lrs)

## stage 3

SZ = 299
LABEL = 'class_name'

car_tfms = get_transforms()

trn_labels_df = labels_df.loc[labels_df['is_test']==0, ['filename', 'class_name', 'class_id']].copy()

src = (ImageList.from_df(trn_labels_df, path, folder='cars_train', cols='filename')
                    .split_by_rand_pct(valid_pct=0.2, seed=100)
                    .label_from_df(cols=LABEL))

data = (src.transform(car_tfms, size=SZ)
            .databunch()
            .normalize(imagenet_stats))

learn.data = data

learn.freeze()
lr=1e-3
learn.fit_one_cycle(8, lr)



