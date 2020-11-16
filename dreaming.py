import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib as mpl
from PIL import Image, ImageFilter, ImageEnhance
from tqdm import tqdm
import os


if torch.cuda.is_available: device = "cuda:0" 
else: device = "cpu"

print(f"Running on device {device}")

# DREAM PARAMS
LAYER_ID = 34
LEARNING_RATE = 0.0005
ITERATIONS = 5
OCTAVES = 6
OCTAVE_SCALE = 1.3

# ADDITIONAL PROCESSING PARAMS
INITIAL_DOWNSCALE=3
BLUR = False
BLUR_RADIUS = 1
BLEND = False
BLEND_ALPHA = 0.5
SHARPEN = True
SHARPNESS = 0.3

# use the VGG16 model
model = torchvision.models.vgg19(pretrained=True).to(device)

# get a list of layers in the model and print them out
layers = list(model.features.modules())[1:]

for idx, layer in enumerate(layers):
    print("Layer:", idx, layer)

# R G B Means
transformMean = [0.485, 0.456, 0.406]
# R G B Standard Deviations
transformStd = [0.229, 0.224, 0.225]
# Normalisation is done as per the standard normalisation used for pretrained pytorch models
transformNormalise = transforms.Normalize(mean=transformMean,std=transformStd)
# preprocessing transforms image array to tensor and normalises it
transformPreprocess = transforms.Compose([transforms.ToTensor(),
                                          transformNormalise])

# the means for each channel as a tensor
mean_tensor = torch.Tensor(transformMean) 
# standard deviations for each channel as a tensor     
std_tensor = torch.Tensor(transformStd)

# helper to convert tensor to image (reversion the normalisation)
def toImage(tnsr):
    return tnsr * std_tensor + mean_tensor


def dream_once(model, image, layer, iterations, lr):
    '''toImage
    Dream using the supplied model and maximise values at the specified layer using gradient ascent

    @params
        model: a pytorch model
        image: a PIL image
        layer: the layer whos activations we wish to maximise
        iterations: the numer of iterations of gradient ascent to perform
        lr: learning rate for gradient ascent
    
    @returns
        A numpy array of type uint8 that is the dreamed image
    '''
    # transform the input image
    transform = transformPreprocess(image).unsqueeze(0).to(device)
    # turn the input image in a tensor variable with gradients
    img_w_grad = torch.autograd.Variable(transform, requires_grad=True)
    # zero the model
    model.zero_grad()
    # iterate
    for _ in tqdm(range(iterations)):
        # set out to image with gradients
        out = img_w_grad
        # iterate over layers basically a forward pass through the network
        for i in range(layer):
            out = layers[i](out)
        
            # calcuate the L2 norm of the matrix and use it as loss
            loss = torch.linalg.norm(out)
            # calculate loss/dx for weights in input image
            loss.backward(retain_graph=True)
            # gradient ascent on the values in the input image
            img_w_grad.data = img_w_grad.data + lr * img_w_grad.grad.data
    
    # covert from tensor to numpy array
    img = img_w_grad.data.squeeze()
    img = torch.transpose(img, 0, 1)
    img = torch.transpose(img, 1, 2)
    img = np.clip(toImage(img.cpu()), 0, 1)
    return np.uint8(img*250)

def octave_dream(model, image, num_octaves, octave_scale, layer, iterations, lr):
    '''
    Dream at multiple octaves i.e. dream the image at increasing sizes so that features are maximised at multiple granularities

    @params
        model: A pytorch model
        image: A PIL image
        num_octaves: The number of octaves to dream with
        octave_scale: The scale for each octave. A scale > 1 will increase image size and a scale < 1 will decrease
        layer: The layer in the model to maximise
        iterations: The number of iterations for dreaming at each octave
        lr: The learning rate
    
    @returns
        A PIL image 
    '''
    # initial width and height
    width, height = image.size
    original = image
    for octave in range(num_octaves):
        print(f"Processing octave {octave+1} of {num_octaves} | Image resolution is now {width}x{height}")
        # do a dream at this octave
        image_arr = dream_once(model, image, layer, iterations, lr)
        # covert to pil image
        image = Image.fromarray(image_arr, 'RGB')
        # step image size by octave_scale
        height = int(height*octave_scale)
        width = int(width*octave_scale)
        # resize image
        image = image.resize((width, height), resample=Image.LANCZOS)

        if BLEND:
            image = Image.blend(image, original.resize((width, height), resample=Image.LANCZOS), BLEND_ALPHA)
        if BLUR:
            image = image.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))
        if SHARPEN:
            sharpener = ImageEnhance.Sharpness(image)
            image = sharpener.enhance(SHARPNESS)
    
    return image

# do some dreams
for image in os.listdir(os.path.join("inputs")):
    print(f"---- Dreaming {image} ----")
    img_name = image.split(".")[0]
    image = Image.open(os.path.join("inputs", image))
    width, height = image.size
    height = int(height/INITIAL_DOWNSCALE)
    width = int(width/INITIAL_DOWNSCALE)
    image = image.resize((width, height), resample=Image.LANCZOS)
    deep_dreamed_image = octave_dream(model, image, OCTAVES, OCTAVE_SCALE, LAYER_ID, ITERATIONS, LEARNING_RATE)
    deep_dreamed_image.save(os.path.join("outputs", f"{img_name}.jpg"))

