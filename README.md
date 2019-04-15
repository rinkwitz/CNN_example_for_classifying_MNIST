# CNN example for classifying MNIST

This project is intended for educational purposes. It demonstrates the effectiveness of a simple CNN architecture classifying the MNIST data set. The neural network is trained solely on the CPU in reasonable time (~5 min).

## Prerequisites

The software makes use of the following libraries for training and visualization.

* Tensorflow 1.11.0
* Tensorboard 1.11.0
* Keras 2.2.4
* Matplotlib 3.0.0

You can install the dependencies using:

```
pip3 install tensorflow==1.11.0 tensorboard==1.11.0 keras==2.2.4 matplotlib==3.0.0
```

## Visualizations of Data and Preprocessing

The script produces several figures. It displays the first 9 training examples and their labels.

<p align="center">
<img src="/img/Figure_0.png" alt="examples and labels" width="750">
</p>

The MNIST data set consists of 28 x 28 grayscale images. Another way of visualizing the images would be to make a 3D plot with coordinates (x, y, z). This way we could use (x, y) to describe the position of a pixel in the image and the z-coordinate to show the 8-bit grayscale vale of that pixel. The next figure shows the first 4 training example and their labels.

<p align="center">
<img src="/img/Figure_1.png" alt="examples and labels in 3D" width="750">
</p>

In order for gradient descent to converge as fast as possible, we need to standardize our data. For this we subtract the mean and divide by the standard deviation feature-wise:

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?x_{\text{std}}=\frac{x-\mu_{\text{feat}}}{\sigma_{\text{feat}}}" title="equation 01" />
</p>

Here are the first 4 training examples after standardization. 

<p align="center">
<img src="/img/Figure_2.png" alt="examples and labels in 3D after standardization" width="750">
</p>

## Visualizations of Training and Validation

The CNN is trained over 10 epochs using the Adam optimizer and categorical cross entropy as loss.

CNN architecture:
* 8 nodes using 3 x 3 convolutions, activation: ReLu
* Max Pooling using a pool size of 2 x 2
* 8 nodes using 3 x 3 convolutions, activation: ReLu
* Max Pooling using a pool size of 2 x 2
* Flatten
* Dense layer with 128 nodes, activation: ReLu
* Dense layer with 128 nodes, activation: ReLu
* Dense layer with 128 nodes, activation: ReLu
* Dense layer with 10 nodes, activation: Softmax

The following figure show the accuracy and loss on training and validation set over the training process. 

* Training set:

<p align="center">
<img src="/img/acc.png" alt="training accuracy" width="750">
</p>

<p align="center">
<img src="/img/loss.png" alt="training loss" width="750">
</p>

* Validation set:

<p align="center">
<img src="/img/val_acc.png" alt="validation accuracy" width="750">
</p>

<p align="center">
<img src="/img/val_loss.png" alt="validation loss" width="750">
</p>

The figures were created using Tensorboard as training monitoring software. To start Tensorboard in your localhost use the following in terminal.

```
tensorboard --logdir=logs/
```

## Results

Here are some random predictions over the testing set. In the title you can see the predicted label and the probability with which the CNN has assigned the label.

<p align="center">
<img src="/img/Figure_3.png" alt="prediction" width="750">
</p>

## Authors

* [Philip Rinkwitz](https://github.com/rinkwitz)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Useful Links:
* [Visualization of 2D CNN](http://scs.ryerson.ca/~aharley/vis/conv/flat.html)
* [Blog entry on using Tensorboard](https://fizzylogic.nl/2017/05/08/monitor-progress-of-your-keras-based-neural-network-using-tensorboard/)

## Acknowledgements:

The formulas of this README were create using:
* [Codecogs online Latex editor](https://www.codecogs.com/latex/eqneditor.php)



