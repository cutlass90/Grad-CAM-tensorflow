# grad-cam.tensorflow
Gradient class activation maps are a visualization technique for deep learning networks.
This is my implementation of Grad-CAM in tensorflow.
I rewrite Grad-CAM funcion through it is a part of VGG graph. CAM may be obtained directly by launching graph.

The original paper: https://arxiv.org/pdf/1610.02391v1.pdf

Download the VGG16 weights from https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz and put it to project directory.

## Usage

```sh
python3 main.py --input laska.png --output laska_save.png
```

## Results

| Input | Output |
| ------|-----:|
| ![Original image][inp] | ![Original image + Visualization][out] |

[inp]: https://github.com/cutlass90/Grad-CAM-tensorflow/blob/master/laska.png
[out]: https://github.com/cutlass90/Grad-CAM-tensorflow/blob/master/laska_save.png

## Acknowledgement
All code was taken from https://github.com/Ankush96/grad-cam.tensorflow.git and modified a little.



