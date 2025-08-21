### Exercise 0: Python Boot Camp

Please open the google colab notebook by clicking on this [link](https://colab.research.google.com/github/ai-mbl/boot/blob/main/exercise.ipynb). Feel free to make a copy of it so that you can save your edits.

You can open the solutions in colab by clicking [here](https://colab.research.google.com/github/ai-mbl/boot/blob/main/solution.ipynb).

### Overview

In this notebook, we will go through some basic image processing in Python, come across standard tasks required while setting up deep learning pipelines, and familiarize ourselves with popular packages such as `glob`, `tifffile`, `tqdm` and more.

We will learn about:

- Loading images (This is important, as images serve as the primary input for many deep learning models in computer vision)
- Normalizing images (This is important as it helps in faster convergence of models because it helps in reducing the scale of the input data and hence the scale of the gradients)
- Cropping images (This is important as it reduces image size by removing peripheral regions, creating smaller inputs that improve memory efficiency during training.)
- Downsampling images (This is important as it reduces image size by lowering its resolution—scaling down the number of pixels—unlike cropping, which removes parts of the image.)
- Flipping images (This is important as it  creates mirrored versions of the originals, effectively augmenting the dataset with new examples and improving model generalisation without increasing memory requirements.)
- Batching images (Since training is typically performed using Stochastic Gradient Descent, which updates model weights based on small subsets of data, batching is important as it enables memory-efficient training and helps stabilise the optimisation process).
- Convolutions (This is important as it is the primary operation in Convolutional Neural Networks, where small filters slide over the image to extract local features)
- Data Augmentation (This is important as it helps artificially increase the size of the training data, which is useful for training models in a data-efficient way—achieving better performance with limited original data).


### Dataset

We will be using sample images from the _MoNuSeg_ dataset provided by [Kumar et al, 2018](https://ieeexplore.ieee.org/document/8880654). The data was publicly made available [here](https://monuseg.grand-challenge.org/) by the authors of the publication.
This dataset shows Hematoxylin and Eosin (H&E) Stained Images showing nuclei in different shapes.
