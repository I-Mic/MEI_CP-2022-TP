#ifndef HEADER_FILE
#define HEADER_FILE

#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <cuda_runtime.h>

cudaEvent_t start, stop;

using namespace std;

void checkCUDAError (const char *msg) {
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) {
		cerr << "Cuda error: " << msg << ", " << cudaGetErrorString( err) << endl;
		exit(-1);
	}
}

void startKernelTime (void) {
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
}

void stopKernelTime (void) {
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cout << endl << "Basic profiling: " << milliseconds << " ms have elapsed for the kernel execution" << endl << endl;
}

#endif