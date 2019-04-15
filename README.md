# CNN example for classifying MNIST

This project is intended for educational purposes. It demonstrates the effectiveness of a simple CNN architecture classfying the MNIST dataset. The neural netwok is trained solely on the CPU in reasonable time (~5 min).

## Prerequisites

The software makes use of the following libraries for training and visualization:

* Tensorflow 1.11.0
* Tensorboard 1.11.0
* Keras 2.2.4
* Matplotlib 3.0.0

You can install the dependencies using:

```
pip3 install tensorflow==1.11.0 tensorboard==1.11.0 keras==2.2.4 matplotlib==3.0.0
```

## Visualizations of data and preprocessing

The script produces several figures. It displays the first 9 training examples and their labels.

<p align="center">
<img src="/img/Figure_0.png" alt="examples and labels" width="500">
</p>

The MNIST dataset consists of 28x28 greyscale images. Another way of visualizing the images would be to make a 3D plot with coordinates (x, y, z). This way we could use (x, y) to describe the position of a pixel in the image and the z-coordinate to show the 8-bit greyscale vale of that pixel. The next figure shows the first 4 training example and their labels.

<p align="center">
<img src="/img/Figure_1.png" alt="examples and labels in 3D" width="500">
</p>

In order for gradient descent to converge as fast as possible, we need to standardize our data. For this we substract the mean and divide by the standard deviation feature-wise:

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?x_{\text{std}}=\frac{x-\mu_{\text{feat}}}{\sigma_{\text{feat}}}" title="equation 01" />
</p>


## Authors

* [Philip Rinkwitz](https://github.com/rinkwitz)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Useful Links:
* [Visualization of 2D CNN](http://scs.ryerson.ca/~aharley/vis/conv/flat.html)
* [Blog entry on using Tensorboard](https://fizzylogic.nl/2017/05/08/monitor-progress-of-your-keras-based-neural-network-using-tensorboard/)

