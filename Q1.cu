#include "com2039.hpp"
#include <string>
#include <cstdlib>

size_t loadSamples(std::string fileName, float** addressToSamples) {
    std::string line;
    std::ifstream inputFile(fileName);

    // Count the number of lines to allocate
    // the right amount of memory
    size_t numLines = 0;
    while (std::getline(inputFile, line)) {
        numLines++;
    }

    // Allocate memory on CPU
    // Notice that we dereference addressToSamples
    *addressToSamples = (float*)malloc(numLines * sizeof(float));

    // Return to the beginning of the file
    // and fill in the
    inputFile.clear();
    inputFile.seekg(0);
    for (int j = 0; j < numLines; j++) {
        std::getline(inputFile, line);
        (*addressToSamples)[j] = std::stof(line);
    }

    inputFile.close();
    return numLines;
}

// Find maximum
__global__ void maxReduceKernel(float* d_in, int lenArray, float* d_out) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gridSize = blockDim.x * gridDim.x;

    float maxVal = -FLT_MAX;
    while (i < lenArray) {
        maxVal = fmaxf(maxVal, d_in[i]);
        i += gridSize;
    }

    sdata[tid] = maxVal;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (tid == 0) {
        d_out[blockIdx.x] = sdata[0];
    }
}

float findMaxValue(float* samples_h, size_t numSamples) {
    float* samples_d;
    cudaMalloc((void**)&samples_d, numSamples * sizeof(float));
    cudaMemcpy(samples_d, samples_h, numSamples * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (numSamples + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate memory for block-wise maximums
    float* maxValues_d;
    cudaMalloc((void**)&maxValues_d, blocksPerGrid * sizeof(float));

    // Launch kernel for reduction
    maxReduceKernel << <blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float) >> > (samples_d, numSamples, maxValues_d);

    // Launch another kernel to find the maximum among block-wise maximums
    int numBlocksRemaining = blocksPerGrid;
    while (numBlocksRemaining > 1) {
        int threads = (numBlocksRemaining + 1) / 2;
        maxReduceKernel << <threads, threadsPerBlock, threadsPerBlock * sizeof(float) >> > (maxValues_d, numBlocksRemaining, maxValues_d);
        numBlocksRemaining = (numBlocksRemaining + threadsPerBlock - 1) / threadsPerBlock;
    }

    // Copy the result back to host
    float result;
    cudaMemcpy(&result, maxValues_d, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(samples_d);
    cudaFree(maxValues_d);

    return result;
}


// Find minimum
__global__ void minReduceKernel(float* d_in, int lenArray, float* d_out) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gridSize = blockDim.x * gridDim.x;

    float minVal = FLT_MAX;
    while (i < lenArray) {
        minVal = fminf(minVal, d_in[i]);
        i += gridSize;
    }

    sdata[tid] = minVal;


    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (tid == 0) {
        d_out[blockIdx.x] = sdata[0];
    }
}

float findMinValue(float* samples_h, size_t numSamples) {
    float* samples_d;
    cudaMalloc((void**)&samples_d, numSamples * sizeof(float));
    cudaMemcpy(samples_d, samples_h, numSamples * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (numSamples + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate memory for block-wise minimums
    float* minValues_d;
    cudaMalloc((void**)&minValues_d, blocksPerGrid * sizeof(float));

    // Launch kernel for reduction
    minReduceKernel << <blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float) >> > (samples_d, numSamples, minValues_d);

    // Launch another kernel to find the minimum among block-wise minimums
    int numBlocksRemaining = blocksPerGrid;
    while (numBlocksRemaining > 1) {
        int threads = (numBlocksRemaining + 1) / 2;
        minReduceKernel << <threads, threadsPerBlock, threadsPerBlock * sizeof(float) >> > (minValues_d, numBlocksRemaining, minValues_d);
        numBlocksRemaining = (numBlocksRemaining + threadsPerBlock - 1) / threadsPerBlock;
    }

    // Copy the result back to host
    float result;
    cudaMemcpy(&result, minValues_d, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(samples_d);
    cudaFree(minValues_d);

    return result;
}

#define BIN_COUNT 256

// Create histogram
__global__ void histogramKernel256(float* d_in, unsigned int* hist, size_t lenArray, float minValue, float maxValue) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int numThreads = blockDim.x * gridDim.x;

    // Initialize histogram bins to zero
    for (int i = tid; i < BIN_COUNT; i += numThreads) {
        hist[i] = 0;
    }

    __syncthreads();

    // Compute histogram
    for (int i = tid; i < lenArray; i += numThreads) {
        float value = d_in[i];
        int bin = (value - minValue) / (maxValue - minValue) * BIN_COUNT;
        atomicAdd(&hist[bin], 1);
    }
}

/// histogram
void histogram256(float* samples_h, size_t numSamples, unsigned int** hist_h, float minValue, float maxValue) {
    // Allocate device memory for samples and histogram
    float* samples_d;
    cudaMalloc((void**)&samples_d, numSamples * sizeof(float));
    cudaMemcpy(samples_d, samples_h, numSamples * sizeof(float), cudaMemcpyHostToDevice);

    unsigned int* hist_d;
    cudaMalloc((void**)&hist_d, BIN_COUNT * sizeof(unsigned int));

    // Set up kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (numSamples + threadsPerBlock - 1) / threadsPerBlock;

    // Launch histogram kernel
    histogramKernel256 << <blocksPerGrid, threadsPerBlock >> > (samples_d, hist_d, numSamples, minValue, maxValue);

    // Copy histogram from device to host
    *hist_h = (unsigned int*)malloc(BIN_COUNT * sizeof(unsigned int));
    cudaMemcpy(*hist_h, hist_d, BIN_COUNT * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(samples_d);
    cudaFree(hist_d);
}
