
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <algorithm>

/*
TODOs

Wrong result when size != 2^n
Cannot handle (wrong result) when size is huge
*/

using namespace std;

static void HandleError(cudaError_t err,
	const char *file,
	int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		system("pause");
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__global__ void GenerateHistogramAndPredicate(int *input, int *currentBit, int *numBits, int *bitHistogram, int *predicate, int *size)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= (*size))
	{
		return;
	}

	int bit = (input[id] >> (*currentBit)) & ((1 << *numBits) - 1);

	atomicAdd(&(bitHistogram[bit * gridDim.x + blockIdx.x]), 1);
	predicate[bit * (*size) + id] = 1;
}

__global__ void PrefixSum(int *input, int *output, int *size, int *totalBits)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= (*size))
	{
		return;
	}

	int bit = 0;

	for (bit = 0; bit < (*totalBits); bit++)
	{
		int current_value = input[bit * (*size) + id];
		int current_cdf = input[bit * (*size) + id];

		for (unsigned int interval = 1; interval < blockDim.x; interval <<= 1)
		{
			if (threadIdx.x >= interval)
			{
				current_cdf += input[bit * (*size) + id - interval];
			}
			__syncthreads();

			input[bit * (*size) + id] = current_cdf;
			__syncthreads();
		}

		output[bit * (*size) + id] = input[bit * (*size) + id] - current_value;
		__syncthreads();
	}
}

__global__ void ReOrder(int *input, int *output, int *bitScan, int *relativePos, int *currentBit, int *numBits, int *size)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= (*size))
	{
		return;
	}

	int bit = (input[id] >> (*currentBit)) & ((1 << *numBits) - 1);

	output[relativePos[bit * (*size) + id] + bitScan[bit * gridDim.x + blockIdx.x]] = input[id];
}

int pow(int a, int b)
{
	int result = 1;
	for (int i = 0; i < b; i++)
	{
		result *= a;
	}

	return result;
}

const int arraySize = 262144, gridSize = 1024;
const int gridCount = arraySize / gridSize;
int input[arraySize] = { 0 };
int output[arraySize] = { 0 };

int main()
{
	const int totalBits = 22, numBits = 2;
	const int numBitsPow2 = pow(2, numBits);

	int sizeBitScan = numBitsPow2 * gridCount;
	int one = 1;

	int tmp_bitHistogram[32] = { 0 };

	printf("Data generation...\n===============\n| Range: 0 ~ %d\n| Size:  %d\n| GSize: %d\n===============\n\n", pow(2, totalBits), arraySize, gridSize);
	//init data
	srand(time(0));
	for (int i = 0; i < arraySize; i++)
	{
		input[i] = rand() % (pow(2, totalBits) - 1);
	}

	printf("Sending data to GPU...\n");

	//Input:		arraySize				the input array
	//Output:		arraySize				result
	//currentBit:	1						current bit pos
	//bitLenth:		1						current bit lenth (numBits)
	//bitHistogram:	2^numBits				count of items with value i at current bit
	//bitScan:		2^numBits				prefix sum of bitHistogram
	//predicate:	arraySize * 2^numBits	T/F if item value equals to i at current bit
	//relativePos:	arraySize * 2^numBits	prefix sum of predicate
	//size:			1						arraySize
	int *d_Input = 0, *d_Output = 0, *d_bitHistogram = 0, *d_bitScan = 0,
		*d_predicate = 0, *d_relativePos = 0, *d_currentBit = 0, *d_bitLenth = 0, *d_size = 0,
		*d_sizeBitScan = 0, *d_one = 0, *d_bitLenthPow2 = 0;

	// Choose which GPU to run on, change this on a multi-GPU system.
	HANDLE_ERROR(cudaSetDevice(0));

	// Allocate GPU buffers
	HANDLE_ERROR(cudaMalloc((void**)&d_Output, arraySize * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&d_Input, arraySize * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&d_bitHistogram, gridCount * numBitsPow2 * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&d_bitScan, gridCount * numBitsPow2 * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&d_predicate, arraySize * numBitsPow2 * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&d_relativePos, arraySize * numBitsPow2 * sizeof(int)));

	HANDLE_ERROR(cudaMalloc((void**)&d_currentBit, sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&d_bitLenth, sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&d_size, sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&d_sizeBitScan, sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&d_one, sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&d_bitLenthPow2, sizeof(int)));

	// Copy input vectors from host memory to GPU buffers.
	HANDLE_ERROR(cudaMemcpy(d_Input, input, arraySize * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_bitLenth, &numBits, sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_size, &arraySize, sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_sizeBitScan, &sizeBitScan, sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_bitLenthPow2, &numBitsPow2, sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_one, &one, sizeof(int), cudaMemcpyHostToDevice));

	printf("GPU Sort Started!\n");

	std::clock_t start;
	start = std::clock();

	//Do the sort
	for (int i = 0; i < totalBits; i += numBits)
	{
		//update current bit
		HANDLE_ERROR(cudaMemcpy(d_currentBit, &i, sizeof(int), cudaMemcpyHostToDevice));

		//clear buffers
		HANDLE_ERROR(cudaMemset(d_bitHistogram, 0, gridCount * numBitsPow2 * sizeof(int)));
		HANDLE_ERROR(cudaMemset(d_bitScan, 0, gridCount * numBitsPow2 * sizeof(int)));
		HANDLE_ERROR(cudaMemset(d_predicate, 0, numBitsPow2 * arraySize * sizeof(int)));
		HANDLE_ERROR(cudaMemset(d_relativePos, 0, numBitsPow2 * arraySize * sizeof(int)));

		////check results
		//HANDLE_ERROR(cudaMemcpy(output, d_Input, arraySize * sizeof(int), cudaMemcpyDeviceToHost));
		//printf("Input:\t");
		//for (int i = 0; i < arraySize; i++)
		//{
		//	printf("%d ", output[i]);
		//}
		//printf("\n");

		/////////////////

		GenerateHistogramAndPredicate <<< gridCount, gridSize >>> (d_Input, d_currentBit, d_bitLenth, d_bitHistogram, d_predicate, d_size);

		////check results
		//HANDLE_ERROR(cudaDeviceSynchronize());
		//HANDLE_ERROR(cudaMemcpy(tmp_bitHistogram, d_bitHistogram, numBitsPow2 * gridCount * sizeof(int), cudaMemcpyDeviceToHost));
		//printf("Bit  %d:\t", i);
		//for (int j = 0; j < gridCount; j++)
		//{
		//	for (int k = 0; k < numBitsPow2; k++)
		//	{
		//		printf("%d ", tmp_bitHistogram[j * numBitsPow2 + k]);
		//	}
		//	printf("| ");
		//}
		//printf("\n");

		/////////////////

		PrefixSum <<< 1, numBitsPow2 * gridCount >>> (d_bitHistogram, d_bitScan, d_sizeBitScan, d_one);

		////check results
		//HANDLE_ERROR(cudaDeviceSynchronize());
		//HANDLE_ERROR(cudaMemcpy(tmp_bitHistogram, d_bitScan, numBitsPow2 * gridCount * sizeof(int), cudaMemcpyDeviceToHost));
		//printf("Scan %d:\t", i);
		//for (int j = 0; j < gridCount; j++)
		//{
		//	for (int k = 0; k < numBitsPow2; k++)
		//	{
		//		printf("%d ", tmp_bitHistogram[j * numBitsPow2 + k]);
		//	}
		//	printf("| ");
		//}
		//printf("\n");

		/////////////////

		////check results
		//HANDLE_ERROR(cudaMemcpy(tmp_bitHistogram, d_predicate, numBitsPow2 * arraySize * sizeof(int), cudaMemcpyDeviceToHost));
		//printf("Pred %d:\t", i);
		//for (int j = 0; j < numBitsPow2; j++)
		//{
		//	for (int k = 0; k < arraySize; k++)
		//	{
		//		printf("%d ", tmp_bitHistogram[j * arraySize + k]);
		//	}
		//	printf("| ");
		//}
		//printf("\n");

		/////////////////

		PrefixSum <<< gridCount, gridSize >>> (d_predicate, d_relativePos, d_size, d_bitLenthPow2);

		////check results
		//HANDLE_ERROR(cudaDeviceSynchronize());
		//HANDLE_ERROR(cudaMemcpy(tmp_bitHistogram, d_relativePos, numBitsPow2 * arraySize * sizeof(int), cudaMemcpyDeviceToHost));
		//printf("RPos %d:\t", i);
		//for (int j = 0; j < numBitsPow2; j++)
		//{
		//	for (int k = 0; k < arraySize; k++)
		//	{
		//		printf("%d ", tmp_bitHistogram[j * arraySize + k]);
		//	}
		//	printf("| ");
		//}
		//printf("\n");

		/////////////////

		ReOrder <<< gridCount, gridSize >>> (d_Input, d_Output, d_bitScan, d_relativePos, d_currentBit, d_bitLenth, d_size);

		////check results
		//HANDLE_ERROR(cudaDeviceSynchronize());
		//HANDLE_ERROR(cudaMemcpy(output, d_Output, arraySize * sizeof(int), cudaMemcpyDeviceToHost));
		//printf("Output:\t");
		//for (int i = 0; i < arraySize; i++)
		//{
		//	printf("%d ", output[i]);
		//}
		//printf("\n");
		HANDLE_ERROR(cudaDeviceSynchronize());

		/////////////////
		//Swap input and output for next iter
		int* tmp = d_Input;
		d_Input = d_Output;
		d_Output = tmp;
		
		//printf("\n*-*-*-*-*-*-*\n");
	}
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	HANDLE_ERROR(cudaDeviceSynchronize());

	double duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	printf("\nGPU Sort Finished! time cost (ms): %.3lf\n\n", duration * 1000.0);

	printf("Collecting results...\n");
	HANDLE_ERROR(cudaMemcpy(output, d_Input, arraySize * sizeof(int), cudaMemcpyDeviceToHost));

	printf("Checking results...\n\n");
	bool validate = true;
	for (int i = 1; i < arraySize; i++)
	{
		if (output[i - 1] > output[i])
		{
			validate = false;
		}
	}

	if (validate)
	{
		printf("Correct!\n");
	}
	else
	{
		printf("Wrong...!\n");
	}

	printf("\n==*==*==*==\nCPU Sort Started!\n");
	start = std::clock();

	std::sort(input, input + arraySize);

	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	printf("\nCPU Sort Finished! time cost (ms): %.3lf\n\n", duration * 1000.0);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
	HANDLE_ERROR(cudaDeviceReset());

	cudaFree(d_Input);
	cudaFree(d_Output);
	cudaFree(d_bitHistogram);
	cudaFree(d_bitScan);
	cudaFree(d_predicate);
	cudaFree(d_relativePos);
	cudaFree(d_currentBit);
	cudaFree(d_bitLenth);
	cudaFree(d_size);

	system("pause");

    return 0;
}
