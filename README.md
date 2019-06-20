# Stanford Cars Image Classification 

Creating CNN image classifier on Stanford Cars dataset using fastai library.

## Summary

Summary of best model based on experiments in [Solution_notebook.ipynb](https://github.com/avkmal/StanfordCars/blob/master/Solution_notebook.ipynb):

**Stage 1**:
- initialize the model(resnet152 with pretrained weights on ImageNet).
- train final layer
- best learning rate: 0.001
- number of epochs: 16
- train loss: 0.364173 	
- validation loss: 0.630648
- validation accuracy: 0.810811

**Stage 2**:
- unfreeze all layers and train.
- best learning rate: 0.001
- number of epochs: 15
- using differential learning rate: [lr/9, lr/6, lr]. The earlier layers was trained on very small learning rate, 0.001/9, the middle layers trained on a  bit higher learning rate, 0.001/6, and final layers trained using learning rate, 0.001
- train loss: 0.250652 	
- validation loss: 0.475406
- validation accuracy: 0.867322

**Stage 3**:
- resize image to 299, freeze the layers and train the final layer.
- best learning rate: 0.001
- number of epochs: 8
- train loss: 0.213713 	
- validation loss: 0.444467
- validation accuracy: 0.878378

**Final Testing**: <br>
- Use Test-Time-Augmentation(TTA). Taking series of different versions of the original image(cropping different areas, changing the zoom) and passing them through the model to get predictions. The average output predictions is then calculated based on different versions of the image and this is given as final output precition for the image
- Accuracy on test set: **0.8827**

**Testing Steps**:
- dependency: PyTorch and `fastai` library. Install:
```
conda install -c pytorch -c fastai fastai
```
or

```
pip install fastai
```
  
- data should be put inside `data` folder using `cars_train` and `cars_test` as folder names.
- run `testing.py`

Model file can be found [here](https://github.com/avkmal/StanfordCars/tree/master/data/models). It's stored in git LFS as the size is about 200 mb.

