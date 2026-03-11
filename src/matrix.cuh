#pragma once
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matmulKernel(float* A, float* B, float* C,
                              int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rowsA && col < colsB) {
        float sum = 0.0f;
        for (int k = 0; k < colsA; k++)
            sum += A[row * colsA + k] * B[k * colsB + col];
        C[row * colsB + col] = sum;
    }
}

__global__ void addBiasKernel(float* output, float* bias, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols)
        output[row * cols + col] += bias[col];
}

__global__ void reluKernel(float* data, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        data[i] = fmaxf(0.0f, data[i]);
}

__global__ void reluDerivKernel(float* grad, float* activated, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        grad[i] *= (activated[i] > 0.0f) ? 1.0f : 0.0f;
}

__global__ void softmaxKernel(float* input, float* output, int batchSize, int numClasses) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batchSize) return;
    float* in  = input  + b * numClasses;
    float* out = output + b * numClasses;
    float maxVal = in[0];
    for (int i = 1; i < numClasses; i++)
        maxVal = fmaxf(maxVal, in[i]);
    float sum = 0.0f;
    for (int i = 0; i < numClasses; i++) {
        out[i] = expf(in[i] - maxVal);
        sum += out[i];
    }
    for (int i = 0; i < numClasses; i++)
        out[i] /= sum;
}

// Momentum SGD: velocity = momentum*velocity + grad
//               weight   = weight - lr*velocity
__global__ void momentumKernel(float* weights, float* grads, float* velocity,
                                float lr, float momentum, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        velocity[i] = momentum * velocity[i] + grads[i];
        weights[i] -= lr * velocity[i];
    }
}

__global__ void subtractKernel(float* A, float* B, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        A[i] -= B[i];
}

__global__ void transposeKernel(float* input, float* output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols)
        output[col * rows + row] = input[row * cols + col];
}

__global__ void sumColumnsKernel(float* input, float* output, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;
    float sum = 0.0f;
    for (int row = 0; row < rows; row++)
        sum += input[row * cols + col];
    output[col] = sum;
}

__global__ void sgdKernel(float* weights, float* grads, float lr, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        weights[i] -= lr * grads[i];
}