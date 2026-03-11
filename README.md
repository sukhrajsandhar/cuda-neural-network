# CUDA Neural Network from Scratch

A fully-connected neural network implemented entirely in CUDA C++ without any 
machine learning frameworks. Every operation — forward pass, backpropagation, 
and optimization — runs as a custom GPU kernel on the RTX 5070 Ti.

## Results

| Epoch | Loss   | Train Accuracy | Test Accuracy |
|-------|--------|----------------|---------------|
| 1     | 1.4067 | 64.6%          | —             |
| 10    | 0.2852 | 91.6%          | —             |
| 30    | 0.1548 | 95.4%          | —             |
| 60    | 0.0952 | 97.3%          | **95.97%**    |

> GPU: NVIDIA RTX 5070 Ti | CUDA 13.2 | Dataset: MNIST (60,000 training images)

## Architecture
```
Input (784) → Hidden1 (512) → Hidden2 (256) → Output (10)
```

- **Activation:** ReLU on hidden layers, Softmax on output
- **Loss:** Cross-entropy
- **Optimizer:** SGD with momentum (0.9) and learning rate decay
- **Batch size:** 128
- **Epochs:** 60

## What Makes This Unique

Most ML projects use PyTorch or TensorFlow which hide all the complexity.
This project implements every operation manually as CUDA kernels:

| Operation | Implementation |
|---|---|
| Matrix multiply | `matmulKernel` — each thread computes one output cell |
| ReLU | `reluKernel` — parallel activation across all neurons |
| Softmax | `softmaxKernel` — numerically stable with max subtraction |
| Backpropagation | Manual chain rule across all 3 layers |
| Momentum SGD | `momentumKernel` — velocity accumulation on GPU |
| Bias gradients | `sumColumnsKernel` — parallel reduction across batch |

## How It Works

### Forward Pass
```
input → matmul(W1) → addBias(b1) → ReLU → 
        matmul(W2) → addBias(b2) → ReLU →
        matmul(W3) → addBias(b3) → Softmax → probabilities
```

### Backward Pass
```
dOutput = softmax - labels          (cross entropy + softmax derivative)
dW3     = h2ᵀ × dOutput            (weight gradient)
dH2     = dOutput × W3ᵀ * ReLU'    (propagate error back)
dW2     = h1ᵀ × dH2
dH1     = dH2 × W2ᵀ * ReLU'
dW1     = inputᵀ × dH1
```

### Weight Update (Momentum SGD)
```
velocity = 0.9 * velocity + gradient
weight   = weight - lr * velocity
```

## Project Structure
```
cuda-neural-network/
├── src/
│   ├── neural_net.cu    # Main training loop
│   ├── matrix.cuh       # All CUDA kernels
│   └── mnist.cuh        # MNIST data loader
├── data/                # MNIST binary files (not tracked)
└── README.md
```

## Build & Run

**Requirements:**
- NVIDIA GPU (CUDA capable)
- CUDA Toolkit 12+
- Visual Studio Build Tools 2022

**Download MNIST data:**
```bash
python -c "
import urllib.request, gzip, shutil, os
os.makedirs('data', exist_ok=True)
files = {
    'data/train-images.idx3-ubyte': 'https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz',
    'data/train-labels.idx1-ubyte': 'https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz',
    'data/test-images.idx3-ubyte':  'https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz',
    'data/test-labels.idx1-ubyte':  'https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz',
}
for dest, url in files.items():
    gz = dest + '.gz'
    urllib.request.urlretrieve(url, gz)
    with gzip.open(gz, 'rb') as f_in, open(dest, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(gz)
"
```

**Build and run:**
```bash
cd src
nvcc neural_net.cu -o neural_net.exe
neural_net.exe
```