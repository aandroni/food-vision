# Food Vision

In this project we develop a deep convolutional neural network that beats the 77.4% top-1 accuracy reported in the [DeepFood article](https://www.researchgate.net/publication/304163308_DeepFood_Deep_Learning-Based_Food_Image_Recognition_for_Computer-Aided_Dietary_Assessment).

The data comes from the [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) dataset, which contains 101,000 images belonging to 101 different food classes. The machine learning task is therefore a multi-class classification problem with images as input and labels as output.

To achieve high accuracy we take advantage of transfer learning, which consists in reusing the weights of a model that has been pretrained on a different dataset. In order adapt the model to the Food-101 dataset we proceed in two steps:

1. Replace the topmost layer of the pretrained model with a new layer, then train for a few epochs while keeping the lower layers frozen;

2. Unfreeze the lower layers and train for a few more epochs (this fine tunes the weights of the entire network).

The pretrained model is the [EfficientNetB0](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB0) network trained on ImageNet.

### Dependencies

The project is a single Jupyter Notebook file (*food_vision.ipynb*) and depends on a few libraries:

* `numpy` and `pandas` for data manipulation.
* `tensorflow` for deep learning.
* `sklearn` for computing performance metrics.
* `matplotlib` and `seaborn` for visualization.
