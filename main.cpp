#include "com2039.hpp"

using namespace std;


int main(int argc, char* argv[]) {
	string datasetFileName = "";

	if (argc == 2) {
		datasetFileName = argv[1];
	}
	else if (exists("data_points.txt")) {
		datasetFileName = "data_points.txt";
	}
	else if (exists("/vol/teaching/CSEE/COM2039/data_points.txt")) {
		datasetFileName = "/vol/teaching/CSEE/COM2039/data_points.txt";
	}
	else {
		cout << "Download the data file from https://csee.pages.surrey.ac.uk/com2039/data/data_points.txt and put it in the current directory.";
		return 1;
	}

	// Load dataset from file
	float* samples_h;
	size_t numSamples = loadSamples(datasetFileName, &samples_h);
	std::cout << "length of vector " << numSamples << std::endl;

	// Find maximum and minimum values
	float maxValue = findMaxValue(samples_h, numSamples);
	std::cout << "GPU Max: " << std::fixed << std::setprecision(6) << maxValue << std::endl;

	float minValue = findMinValue(samples_h, numSamples);
	std::cout << "GPU Min: " << std::fixed << std::setprecision(6) << minValue << std::endl;

	// Find histogram
	unsigned int* hist_h = NULL;

	histogram256(samples_h, numSamples, &hist_h, minValue, maxValue);
	unsigned long int counter = 0;
	for (int j = 0; j < NUM_BINS; j++) {
		std::cout << "Bin[" << j << "]: " << hist_h[j] << std::endl;
		counter += hist_h[j];
	}
	std::cout << "Total number of elements in histogram: " << counter << std::endl;

	// free memory
	free(samples_h);


	return 0;
}