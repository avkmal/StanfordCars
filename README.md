# Stanford Cars Image classification (Grab AI for Sea Challenge)

Creating CNN image classifier on Stanford Cars dataset using fastai library

## Summary

Summary of best models based on experiment above:

**Stage 1**:
- Initialize the model(resnet152 with pretrained weights on ImageNet).
- Train final layer
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
Accuracy on test set: **0.8827**
