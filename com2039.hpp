#ifndef COM2039_HPP_
#define COM2039_HPP_

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cfloat>
//#include <filesystem>

#include "cuda_runtime.h"

using namespace std;

const size_t BLOCK_SIZE = 1024;
const size_t NUM_BINS = 256;

// Find Maximum
__global__ void maxReduceKernel(float* d_in, int lenArray);
float findMaxValue(float* samples_h, size_t numSamples);

// Find Minimum
__global__ void minReduceKernel(float* d_in, int lenArray);
float findMinValue(float* samples_h, size_t numSamples);

// Histogram
__global__ void histogramKernel256(float* d_in, unsigned int* hist, size_t lenArray, float minValue, float maxValue);
void histogram256(float* samples_h, size_t numSamples, unsigned int** hist_h, float minValue, float maxValue);

// Load files
size_t loadSamples(string fileName, float** addressToSamples);

// Check if file exists
inline bool exists(char* name) {
    ifstream f(name);
    return f.good();
}

#endif /* COM2039_HPP_ */
