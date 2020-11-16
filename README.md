# Deep Dreaming

## Overview
This is an implementation of deep dreaming in Python using PyTorch. The model used in a pre-trained VGG19.

The basic idea is to make an input image into a variable that has gradients associated with it. This image is then passed through the network until the desired layer is reached. The loss is then computer as the L2 norm of the desired layer and this is backpropagated to the input image. The computed gradients are then used to perform gradient ascent on the input image by the formula <img src="https://render.githubusercontent.com/render/math?math=X = X + lr * \frac{d}{dX}f(X)">  where X is the input image as a tensor.

## Examples

![Dog Dream1](outputs/bliss_20.jpg)

![Dog Dream2](outputs/road_20.jpg)

![Dog Dream3](outputs/starry_night_20.jpg)
