from fastai import *
from fastai.vision import *
import pandas as pd

labels_df = pd.read_csv('labels.csv')

path = 'data/'

SZ = 299
LABEL = 'class_name'

car_tfms = get_transforms()


src = (ImageList.from_df(labels_df, path, folder='cars_test', cols='filename')
       # the 'is_test' column has values of 1 for the test set
       .split_from_df(col='is_test')
       .label_from_df(cols=LABEL))

data_test = (src.transform(car_tfms, size=SZ)
            .databunch(no_check=True)
            .normalize(imagenet_stats))

arch = models.resnet152
learn = cnn_learner(data_test, arch, metrics=[accuracy])
learn.load('stage3-8epoch')
test_preds, test_ys = learn.TTA()
accuracy(test_preds, test_ys)
