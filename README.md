# Jax&Flax ResNet18 Classification Experiment

## Motivation
Constructing deep learning architectures with many layers is expensive as a lot of calculations, such as backpropagation, are needed. One of the solutions is to use Julia, which uses Just-In-Time. However, it does not have a debugging tool because it is not mature enough. Jax and Flax are neural networks for Python, and they use Just-In-Time compilation while training. Thus, Jax and Flax boost training time rapidly. This project reveals Jax and Flax potential with a cat&dog dataset and points out issues.

## 1.Requirements
- Python ==> 3.8
- Flax ==> 0.5.3
- Jax ==> 0.3.14
- Optax ==> 0.1.3
- Torch
- Numpy

Binary Images:
https://drive.google.com/drive/folders/1RoBdT1k3JI4QMNXOkAx46DxAy-fxMiyO?usp=sharing

Sample Image of Cats and Dogs: 
<img src="./0/cat.1.png" alt="demo cat" title="demo cat">
<img src="./1/dog.1.png" alt="demo dog" title="demo dog">


## 2.Demo
The video shows the actual execution with Jax and Flax.

https://user-images.githubusercontent.com/52090852/182778858-e51c2f0c-5f53-4ffb-85e0-b63ed5130408.mp4


## 3.Results
