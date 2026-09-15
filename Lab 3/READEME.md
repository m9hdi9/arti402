# ARTI 402 - Deep Learning

## Lab 3: CNN Architecture (The Four Building Blocks)

This repository contains the implementation and analysis for **Lab 3** in the ARTI 402 course.

### Overview
The lab covers building the four core components of a Convolutional Neural Network (CNN) from scratch using NumPy:
1. **Filtering (Convolution)**: Feature and edge extraction (`convolve2d`, `conv_layer`).
2. **Max Pooling**: Downsampling and translation invariance (`max_pool2d`, `maxpool_layer`).
3. **Flatten**: Reshaping 3D feature maps into 1D vectors (`flatten_layer`).
4. **Fully Connected (Dense)**: Softmax classification head (`Layer_Dense`).

### Key Findings
- **Translation Invariance**: Demonstrated that a raw-pixel dense model overfits and fails on shifted test data (~33.3% accuracy), whereas a CNN feature extraction pipeline achieves 100% test accuracy.
- **Pooling Effect**: Larger pooling tile sizes increase tolerance to spatial shifts by discarding redundant coordinate details.

### Files
- `arti402_Lab3_2240007563.ipynb`: Completed notebook with all code implementations and outputs.
- `arti402_figures.py`: Visualization helpers.
- `Zebra_image.jpeg`: Sample image used for architecture illustration.
