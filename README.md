## This repo contains work for the paper 'Efficient CNN-LSTM based Image Captioning using Neural Network Compression'

## Team Members
1. Harshit Rampal
2. Aman Mohanty

## Abstract
Modern Neural Networks are eminent in achieving state of the art performance on tasks under Computer Vision, Natural Language Processing and related verticals. However, they are notorious for their voracious memory and compute appetite which further obstructs their deployment on resource limited edge devices. In order to achieve edge deployment, researchers have developed pruning and quantization algorithms to compress such networks without compromising their efficacy. Such compression algorithms are broadly experimented on standalone CNN and RNN architectures while in this work, we present an unconventional end to end compression pipeline of a CNN-LSTM based Image Captioning model. The model is trained using VGG16 or ResNet50 as an encoder and an LSTM decoder on the flickr8k dataset. We then examine the effects of different compression architectures on the model and design a compression architecture that achieves a 71.3% reduction in model size, 73.1% reduction in inference time and a 7.7% increase in BLEU score as compared to its uncompressed counterpart.

## Project Dependencies
tensorflow, tensorflow-model-optimization, keras, nltk, numpy, pickle, PIL, requests, BytesIO

## Dataset
For this project, we train our model on the [flickr8k dataset](https://www.kaggle.com/adityajn105/flickr8k)

## Results
Performance comparison between baseline model and the quantized encoder quantized decoder model. The baseline image captioning model has VGG16 encoder and LSTM decoder and is trained on the flickr8k dataset.

Models | BLEU Score | Model Size(MB) | Inference time for 2000 samples(mins)
-------|------------|----------------|--------------------------------------
Baseline-Baseline | 0.527 | 578.4 | 5.68
Quantized-Quantized | 0.568 | 155.39 | 1.63


## Generated Captions
<img src="https://s3.amazonaws.com/cdn-origin-etr.akc.org/wp-content/uploads/2018/06/05231748/belgian-malinois-running-through-field.jpg" width="250">  

**Baseline**: two dogs are playing together in the grass

**Best**: dog is running through the grass

<img src="https://static01.nyt.com/images/2020/09/25/sports/25soccer-nationalWEB1/merlin_177451008_91c7b66d-3c8a-4963-896e-54280f374b6d-articleLarge.jpg?quality=75&auto=webp&disable=upscale" width="250">

**Baseline**: two men are playing soccer on the grass

**Best**: two men are playing soccer


