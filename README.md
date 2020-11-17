# Denoising Autoencoder

![results](https://github.com/priyavrat-misra/denoising-autoencoder/blob/master/images/Denoised.png?raw=true)
_<sup>__Figure.__ `1st row: Original image`; `2nd row: Noised image`; `3rd & 4th row: 2 different denoising autoencoder's outputs`;</sup>_

## How it works
> Just like a traditional autoencoder, even here there are two parts i.e., an `encoder` and a `decoder`.<br>
> Given an input set of noisy image data and a target set which is non-noisy, the encoder can learn to filter-out important information from the noisy image and the decoder can learn to produce a non-noisy reconstruction!<br>
> After the autoencoder is trained, it should be able to de-noise the new test data _(few such test results can be seen in the figure above)_

<br>

## About this project
> This project uses the [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) for [training the autoencoders](https://github.com/priyavrat-misra/denoising-autoencoder/blob/master/train.ipynb "train.ipynb").
> But before starting the training procedure, random noise is added to the input data. This way the autoencoder will have noisy images as input and original and clean images as target outputs.
>
> Following [two autoencoder architectures](https://github.com/priyavrat-misra/denoising-autoencoder/blob/master/networks.py "networks.py") were trained:
> - One with `Nearest Neighbor Interpolation with Convolution layers` in its decoder and,
> - the other with `Transpose Convolution layers` in its decoder.