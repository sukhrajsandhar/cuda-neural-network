#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "matrix.cuh"
#include "mnist.cuh"

#define INPUT_SIZE    784
#define HIDDEN1_SIZE  512
#define HIDDEN2_SIZE  256
#define OUTPUT_SIZE   10
#define BATCH_SIZE    128
#define EPOCHS        60
#define LEARNING_RATE 0.01f
#define MOMENTUM      0.9f

float* gpuAlloc(int size) {
    float* ptr;
    cudaMalloc(&ptr, size * sizeof(float));
    cudaMemset(ptr, 0, size * sizeof(float));
    return ptr;
}

void initWeights(float* d_weights, int size, float scale) {
    float* h = new float[size];
    for (int i = 0; i < size; i++)
        h[i] = ((float)rand() / RAND_MAX * 2 - 1) * scale;
    cudaMemcpy(d_weights, h, size * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h;
}

float computeLossAndAccuracy(float* h_output, float* h_labels,
                              int batchSize, float& accuracy) {
    float loss = 0.0f;
    int correct = 0;
    for (int b = 0; b < batchSize; b++) {
        float* out = h_output + b * OUTPUT_SIZE;
        float* lbl = h_labels + b * OUTPUT_SIZE;
        int predicted = 0, actual = 0;
        for (int i = 1; i < OUTPUT_SIZE; i++) {
            if (out[i] > out[predicted]) predicted = i;
            if (lbl[i] > lbl[actual])   actual = i;
        }
        if (predicted == actual) correct++;
        loss -= logf(fmaxf(out[actual], 1e-7f));
    }
    accuracy = (float)correct / batchSize * 100.0f;
    return loss / batchSize;
}

int main() {
    srand(42);
    printf("=== CUDA Neural Network from Scratch ===\n");
    printf("Architecture: %d -> %d -> %d -> %d\n\n",
           INPUT_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, OUTPUT_SIZE);

    int trainCount, testCount, imageSize, labelCount;
    float* h_trainImages = loadImages("../data/train-images.idx3-ubyte", &trainCount, &imageSize);
    float* h_trainLabels = loadLabels("../data/train-labels.idx1-ubyte", &labelCount);
    float* h_testImages  = loadImages("../data/test-images.idx3-ubyte",  &testCount,  &imageSize);
    float* h_testLabels  = loadLabels("../data/test-labels.idx1-ubyte",  &testCount);
    printf("\n");

    // Weights
    float* d_W1 = gpuAlloc(INPUT_SIZE   * HIDDEN1_SIZE);
    float* d_W2 = gpuAlloc(HIDDEN1_SIZE * HIDDEN2_SIZE);
    float* d_W3 = gpuAlloc(HIDDEN2_SIZE * OUTPUT_SIZE);
    float* d_b1 = gpuAlloc(HIDDEN1_SIZE);
    float* d_b2 = gpuAlloc(HIDDEN2_SIZE);
    float* d_b3 = gpuAlloc(OUTPUT_SIZE);

    initWeights(d_W1, INPUT_SIZE   * HIDDEN1_SIZE, sqrtf(2.0f / INPUT_SIZE));
    initWeights(d_W2, HIDDEN1_SIZE * HIDDEN2_SIZE, sqrtf(2.0f / HIDDEN1_SIZE));
    initWeights(d_W3, HIDDEN2_SIZE * OUTPUT_SIZE,  sqrtf(2.0f / HIDDEN2_SIZE));

    // Momentum velocities
    float* d_vW1 = gpuAlloc(INPUT_SIZE   * HIDDEN1_SIZE);
    float* d_vW2 = gpuAlloc(HIDDEN1_SIZE * HIDDEN2_SIZE);
    float* d_vW3 = gpuAlloc(HIDDEN2_SIZE * OUTPUT_SIZE);
    float* d_vb1 = gpuAlloc(HIDDEN1_SIZE);
    float* d_vb2 = gpuAlloc(HIDDEN2_SIZE);
    float* d_vb3 = gpuAlloc(OUTPUT_SIZE);

    // Activations
    float* d_input   = gpuAlloc(BATCH_SIZE * INPUT_SIZE);
    float* d_h1      = gpuAlloc(BATCH_SIZE * HIDDEN1_SIZE);
    float* d_h2      = gpuAlloc(BATCH_SIZE * HIDDEN2_SIZE);
    float* d_output  = gpuAlloc(BATCH_SIZE * OUTPUT_SIZE);
    float* d_softmax = gpuAlloc(BATCH_SIZE * OUTPUT_SIZE);

    // Gradients
    float* d_dOutput = gpuAlloc(BATCH_SIZE * OUTPUT_SIZE);
    float* d_dH2     = gpuAlloc(BATCH_SIZE * HIDDEN2_SIZE);
    float* d_dH1     = gpuAlloc(BATCH_SIZE * HIDDEN1_SIZE);
    float* d_dW1     = gpuAlloc(INPUT_SIZE   * HIDDEN1_SIZE);
    float* d_dW2     = gpuAlloc(HIDDEN1_SIZE * HIDDEN2_SIZE);
    float* d_dW3     = gpuAlloc(HIDDEN2_SIZE * OUTPUT_SIZE);
    float* d_db1     = gpuAlloc(HIDDEN1_SIZE);
    float* d_db2     = gpuAlloc(HIDDEN2_SIZE);
    float* d_db3     = gpuAlloc(OUTPUT_SIZE);

    // Transposes
    float* d_inputT = gpuAlloc(BATCH_SIZE * INPUT_SIZE);
    float* d_h1T    = gpuAlloc(BATCH_SIZE * HIDDEN1_SIZE);
    float* d_h2T    = gpuAlloc(BATCH_SIZE * HIDDEN2_SIZE);
    float* d_W2T    = gpuAlloc(HIDDEN1_SIZE * HIDDEN2_SIZE);
    float* d_W3T    = gpuAlloc(HIDDEN2_SIZE * OUTPUT_SIZE);

    float* h_softmax = new float[BATCH_SIZE * OUTPUT_SIZE];
    float* h_labels  = new float[BATCH_SIZE * OUTPUT_SIZE];

    int numBatches = trainCount / BATCH_SIZE;
    printf("Training: %d images | Batch: %d | Batches/epoch: %d\n\n",
           trainCount, BATCH_SIZE, numBatches);

    dim3 block(16, 16);

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float epochLoss = 0.0f;
        float epochAcc  = 0.0f;
        float lr = LEARNING_RATE / (1.0f + 0.005f * epoch);

        for (int batch = 0; batch < numBatches; batch++) {
            int offset = batch * BATCH_SIZE;

            cudaMemcpy(d_input,
                       h_trainImages + offset * INPUT_SIZE,
                       BATCH_SIZE * INPUT_SIZE * sizeof(float),
                       cudaMemcpyHostToDevice);

            // ======== FORWARD PASS ========

            // Layer 1: input -> h1
            matmulKernel<<<dim3((HIDDEN1_SIZE+15)/16,(BATCH_SIZE+15)/16), block>>>(
                d_input, d_W1, d_h1, BATCH_SIZE, INPUT_SIZE, HIDDEN1_SIZE);
            addBiasKernel<<<dim3((BATCH_SIZE+15)/16,(HIDDEN1_SIZE+15)/16), block>>>(
                d_h1, d_b1, BATCH_SIZE, HIDDEN1_SIZE);
            reluKernel<<<(BATCH_SIZE*HIDDEN1_SIZE+255)/256, 256>>>(
                d_h1, BATCH_SIZE * HIDDEN1_SIZE);

            // Layer 2: h1 -> h2
            matmulKernel<<<dim3((HIDDEN2_SIZE+15)/16,(BATCH_SIZE+15)/16), block>>>(
                d_h1, d_W2, d_h2, BATCH_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE);
            addBiasKernel<<<dim3((BATCH_SIZE+15)/16,(HIDDEN2_SIZE+15)/16), block>>>(
                d_h2, d_b2, BATCH_SIZE, HIDDEN2_SIZE);
            reluKernel<<<(BATCH_SIZE*HIDDEN2_SIZE+255)/256, 256>>>(
                d_h2, BATCH_SIZE * HIDDEN2_SIZE);

            // Layer 3: h2 -> output
            matmulKernel<<<dim3((OUTPUT_SIZE+15)/16,(BATCH_SIZE+15)/16), block>>>(
                d_h2, d_W3, d_output, BATCH_SIZE, HIDDEN2_SIZE, OUTPUT_SIZE);
            addBiasKernel<<<dim3((BATCH_SIZE+15)/16,(OUTPUT_SIZE+15)/16), block>>>(
                d_output, d_b3, BATCH_SIZE, OUTPUT_SIZE);
            softmaxKernel<<<(BATCH_SIZE+255)/256, 256>>>(
                d_output, d_softmax, BATCH_SIZE, OUTPUT_SIZE);

            // Loss
            cudaMemcpy(h_softmax, d_softmax,
                       BATCH_SIZE * OUTPUT_SIZE * sizeof(float),
                       cudaMemcpyDeviceToHost);
            memcpy(h_labels, h_trainLabels + offset * OUTPUT_SIZE,
                   BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
            float batchAcc;
            epochLoss += computeLossAndAccuracy(h_softmax, h_labels, BATCH_SIZE, batchAcc);
            epochAcc  += batchAcc;

            // ======== BACKWARD PASS ========

            // dOutput = softmax - labels
            float* d_labels;
            cudaMalloc(&d_labels, BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
            cudaMemcpy(d_labels, h_labels,
                       BATCH_SIZE * OUTPUT_SIZE * sizeof(float),
                       cudaMemcpyHostToDevice);
            cudaMemcpy(d_dOutput, d_softmax,
                       BATCH_SIZE * OUTPUT_SIZE * sizeof(float),
                       cudaMemcpyDeviceToDevice);
            subtractKernel<<<(BATCH_SIZE*OUTPUT_SIZE+255)/256, 256>>>(
                d_dOutput, d_labels, BATCH_SIZE * OUTPUT_SIZE);
            cudaFree(d_labels);

            // dW3 = h2T x dOutput
            transposeKernel<<<dim3((BATCH_SIZE+15)/16,(HIDDEN2_SIZE+15)/16), block>>>(
                d_h2, d_h2T, BATCH_SIZE, HIDDEN2_SIZE);
            cudaMemset(d_dW3, 0, HIDDEN2_SIZE * OUTPUT_SIZE * sizeof(float));
            matmulKernel<<<dim3((OUTPUT_SIZE+15)/16,(HIDDEN2_SIZE+15)/16), block>>>(
                d_h2T, d_dOutput, d_dW3, HIDDEN2_SIZE, BATCH_SIZE, OUTPUT_SIZE);

            // dH2 = dOutput x W3T
            transposeKernel<<<dim3((OUTPUT_SIZE+15)/16,(HIDDEN2_SIZE+15)/16), block>>>(
                d_W3, d_W3T, HIDDEN2_SIZE, OUTPUT_SIZE);
            matmulKernel<<<dim3((HIDDEN2_SIZE+15)/16,(BATCH_SIZE+15)/16), block>>>(
                d_dOutput, d_W3T, d_dH2, BATCH_SIZE, OUTPUT_SIZE, HIDDEN2_SIZE);
            reluDerivKernel<<<(BATCH_SIZE*HIDDEN2_SIZE+255)/256, 256>>>(
                d_dH2, d_h2, BATCH_SIZE * HIDDEN2_SIZE);

            // dW2 = h1T x dH2
            transposeKernel<<<dim3((BATCH_SIZE+15)/16,(HIDDEN1_SIZE+15)/16), block>>>(
                d_h1, d_h1T, BATCH_SIZE, HIDDEN1_SIZE);
            cudaMemset(d_dW2, 0, HIDDEN1_SIZE * HIDDEN2_SIZE * sizeof(float));
            matmulKernel<<<dim3((HIDDEN2_SIZE+15)/16,(HIDDEN1_SIZE+15)/16), block>>>(
                d_h1T, d_dH2, d_dW2, HIDDEN1_SIZE, BATCH_SIZE, HIDDEN2_SIZE);

            // dH1 = dH2 x W2T
            transposeKernel<<<dim3((HIDDEN2_SIZE+15)/16,(HIDDEN1_SIZE+15)/16), block>>>(
                d_W2, d_W2T, HIDDEN1_SIZE, HIDDEN2_SIZE);
            matmulKernel<<<dim3((HIDDEN1_SIZE+15)/16,(BATCH_SIZE+15)/16), block>>>(
                d_dH2, d_W2T, d_dH1, BATCH_SIZE, HIDDEN2_SIZE, HIDDEN1_SIZE);
            reluDerivKernel<<<(BATCH_SIZE*HIDDEN1_SIZE+255)/256, 256>>>(
                d_dH1, d_h1, BATCH_SIZE * HIDDEN1_SIZE);

            // dW1 = inputT x dH1
            transposeKernel<<<dim3((BATCH_SIZE+15)/16,(INPUT_SIZE+15)/16), block>>>(
                d_input, d_inputT, BATCH_SIZE, INPUT_SIZE);
            cudaMemset(d_dW1, 0, INPUT_SIZE * HIDDEN1_SIZE * sizeof(float));
            matmulKernel<<<dim3((HIDDEN1_SIZE+15)/16,(INPUT_SIZE+15)/16), block>>>(
                d_inputT, d_dH1, d_dW1, INPUT_SIZE, BATCH_SIZE, HIDDEN1_SIZE);

            // ======== UPDATE WITH MOMENTUM ========
            float scaledLR = lr / BATCH_SIZE;

            momentumKernel<<<(INPUT_SIZE*HIDDEN1_SIZE+255)/256, 256>>>(
                d_W1, d_dW1, d_vW1, scaledLR, MOMENTUM, INPUT_SIZE*HIDDEN1_SIZE);
            momentumKernel<<<(HIDDEN1_SIZE*HIDDEN2_SIZE+255)/256, 256>>>(
                d_W2, d_dW2, d_vW2, scaledLR, MOMENTUM, HIDDEN1_SIZE*HIDDEN2_SIZE);
            momentumKernel<<<(HIDDEN2_SIZE*OUTPUT_SIZE+255)/256, 256>>>(
                d_W3, d_dW3, d_vW3, scaledLR, MOMENTUM, HIDDEN2_SIZE*OUTPUT_SIZE);

            // Bias gradients
            sumColumnsKernel<<<(OUTPUT_SIZE+255)/256,  256>>>(d_dOutput, d_db3, BATCH_SIZE, OUTPUT_SIZE);
            sumColumnsKernel<<<(HIDDEN2_SIZE+255)/256, 256>>>(d_dH2,    d_db2, BATCH_SIZE, HIDDEN2_SIZE);
            sumColumnsKernel<<<(HIDDEN1_SIZE+255)/256, 256>>>(d_dH1,    d_db1, BATCH_SIZE, HIDDEN1_SIZE);

            momentumKernel<<<(OUTPUT_SIZE+255)/256,  256>>>(d_b3, d_db3, d_vb3, scaledLR, MOMENTUM, OUTPUT_SIZE);
            momentumKernel<<<(HIDDEN2_SIZE+255)/256, 256>>>(d_b2, d_db2, d_vb2, scaledLR, MOMENTUM, HIDDEN2_SIZE);
            momentumKernel<<<(HIDDEN1_SIZE+255)/256, 256>>>(d_b1, d_db1, d_vb1, scaledLR, MOMENTUM, HIDDEN1_SIZE);

        } // end batch

        printf("Epoch %2d/%d | LR: %.5f | Loss: %.4f | Accuracy: %.1f%%\n",
               epoch+1, EPOCHS, lr, epochLoss/numBatches, epochAcc/numBatches);

    } // end epoch

    // ======== TEST ========
    printf("\n=== Testing ===\n");
    float testAcc = 0.0f;
    int testBatches = testCount / BATCH_SIZE;

    for (int batch = 0; batch < testBatches; batch++) {
        int offset = batch * BATCH_SIZE;
        cudaMemcpy(d_input,
                   h_testImages + offset * INPUT_SIZE,
                   BATCH_SIZE * INPUT_SIZE * sizeof(float),
                   cudaMemcpyHostToDevice);

        matmulKernel<<<dim3((HIDDEN1_SIZE+15)/16,(BATCH_SIZE+15)/16), block>>>(
            d_input, d_W1, d_h1, BATCH_SIZE, INPUT_SIZE, HIDDEN1_SIZE);
        addBiasKernel<<<dim3((BATCH_SIZE+15)/16,(HIDDEN1_SIZE+15)/16), block>>>(
            d_h1, d_b1, BATCH_SIZE, HIDDEN1_SIZE);
        reluKernel<<<(BATCH_SIZE*HIDDEN1_SIZE+255)/256, 256>>>(
            d_h1, BATCH_SIZE * HIDDEN1_SIZE);

        matmulKernel<<<dim3((HIDDEN2_SIZE+15)/16,(BATCH_SIZE+15)/16), block>>>(
            d_h1, d_W2, d_h2, BATCH_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE);
        addBiasKernel<<<dim3((BATCH_SIZE+15)/16,(HIDDEN2_SIZE+15)/16), block>>>(
            d_h2, d_b2, BATCH_SIZE, HIDDEN2_SIZE);
        reluKernel<<<(BATCH_SIZE*HIDDEN2_SIZE+255)/256, 256>>>(
            d_h2, BATCH_SIZE * HIDDEN2_SIZE);

        matmulKernel<<<dim3((OUTPUT_SIZE+15)/16,(BATCH_SIZE+15)/16), block>>>(
            d_h2, d_W3, d_output, BATCH_SIZE, HIDDEN2_SIZE, OUTPUT_SIZE);
        addBiasKernel<<<dim3((BATCH_SIZE+15)/16,(OUTPUT_SIZE+15)/16), block>>>(
            d_output, d_b3, BATCH_SIZE, OUTPUT_SIZE);
        softmaxKernel<<<(BATCH_SIZE+255)/256, 256>>>(
            d_output, d_softmax, BATCH_SIZE, OUTPUT_SIZE);

        cudaMemcpy(h_softmax, d_softmax,
                   BATCH_SIZE * OUTPUT_SIZE * sizeof(float),
                   cudaMemcpyDeviceToHost);
        float acc;
        computeLossAndAccuracy(h_softmax,
                               h_testLabels + offset * OUTPUT_SIZE,
                               BATCH_SIZE, acc);
        testAcc += acc;
    }

    printf("Test Accuracy: %.2f%%\n", testAcc / testBatches);

    // Cleanup
    cudaFree(d_W1); cudaFree(d_W2); cudaFree(d_W3);
    cudaFree(d_b1); cudaFree(d_b2); cudaFree(d_b3);
    cudaFree(d_vW1); cudaFree(d_vW2); cudaFree(d_vW3);
    cudaFree(d_vb1); cudaFree(d_vb2); cudaFree(d_vb3);
    cudaFree(d_input); cudaFree(d_h1); cudaFree(d_h2);
    cudaFree(d_output); cudaFree(d_softmax);
    cudaFree(d_dOutput); cudaFree(d_dH1); cudaFree(d_dH2);
    cudaFree(d_dW1); cudaFree(d_dW2); cudaFree(d_dW3);
    cudaFree(d_db1); cudaFree(d_db2); cudaFree(d_db3);
    cudaFree(d_inputT); cudaFree(d_h1T); cudaFree(d_h2T);
    cudaFree(d_W2T); cudaFree(d_W3T);
    delete[] h_trainImages; delete[] h_trainLabels;
    delete[] h_testImages;  delete[] h_testLabels;
    delete[] h_softmax; delete[] h_labels;

    return 0;
}

