#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "DS_timer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DO_CPU
#define DATA_TYPE int

// Matrix size
#define SIZE_M (512*2)
#define SIZE_N (512*4)
#define SIZE_K (512*2)

#define BLOCK_SIZE 16

template<class T> void allocNinitMem(T** p, long long size, double* memUsage = NULL);
bool compareMatrix(DATA_TYPE* _A, DATA_TYPE* _B, int _size);

/******************************************************************
* Complete this kernels
******************************************************************/
__global__ void MatMul(DATA_TYPE* matA, DATA_TYPE* matB, DATA_TYPE* matC, int m, int n, int k)
{
	// Case 1: 블록을 하나만 사용하는 경우 (행렬 C 크기 < 블록 최대 크기 1024)
	//int row = threadIdx.x;
	//int col = threadIdx.y;
	//int index = row * n + col;

	//matC[index] = 0;
	//for (int offset = 0; offset < k; offset++)
	//	matC[index] += matA[row * k + offset] * matB[col + n * offset];

	// Case 2: 여러 개의 블록을 사용하는 경우 (행렬 C 크기 > 블록 최대 크기 1024)
	int row = (blockDim.x * blockIdx.x) + threadIdx.x;
	int col = (blockDim.y * blockIdx.y) + threadIdx.y;
	int index = row * n + col;

	if (row >= m || col >= n) return;

	matC[index] = 0;
	for (int offset = 0; offset < k; offset++)
		matC[index] += matA[row * k + offset] * matB[col + n * offset];
}


int main(int argc, char* argv[])
{
	DS_timer timer(10);
	timer.setTimerName(0, (char*)"CPU code");
	timer.setTimerName(1, (char*)"Kernel");
	timer.setTimerName(2, (char*)"[Data transter] host->device");
	timer.setTimerName(3, (char*)"[Data transfer] device->host");
	timer.setTimerName(4, (char*)"GPU total");

	// set matrix size
	int m, n, k;
	if (argc < 3) {
		m = SIZE_M;
		n = SIZE_N;
		k = SIZE_K;
	}
	else {
		m = atoi(argv[1]);
		n = atoi(argv[2]);
		k = atoi(argv[3]);
	}

	printf("Size : A = (%d by %d), B = (%d by %d), C = (%d by %d)\n", m, k, k, n, m, n);

	int sizeA = m * k;
	int sizeB = k * n;
	int sizeC = m * n;

	// Make matrix
	DATA_TYPE* A = NULL, * B = NULL;
	allocNinitMem<DATA_TYPE>(&A, sizeA);
	allocNinitMem<DATA_TYPE>(&B, sizeB);

	DATA_TYPE* Ccpu = NULL, * Cgpu = NULL;
	allocNinitMem<DATA_TYPE>(&Ccpu, sizeC);
	allocNinitMem<DATA_TYPE>(&Cgpu, sizeC);

	// generate input matrices
	for (int i = 0; i < sizeA; i++) A[i] = ((rand() % 10) + ((rand() % 100) / 100.0));
	for (int i = 0; i < sizeB; i++) B[i] = ((rand() % 10) + ((rand() % 100) / 100.0));

	// CPU algorithm
	timer.onTimer(0);
	for (int row = 0; row < m; row++) {
		for (int col = 0; col < n; col++) {
			int cIndex = row * n + col;
			Ccpu[cIndex] = 0;
			for (int i = 0; i < k; i++)
				Ccpu[cIndex] += (A[row * k + i] * B[i * n + col]);
		}
	}
	printf("CPU finished!\n");
	timer.offTimer(0);

	timer.onTimer(4);
	/******************************************************************
	* Write your codes for GPU algorithm from here
	******************************************************************/
	DATA_TYPE* dA, * dB, * dC;

	// 1. Allocate device memory for dA, dB, dC
	// Hint: cudaMalloc, cudaMemset
	cudaMalloc(&dA, sizeA * sizeof(int));
	cudaMemset(dA, 0, sizeA * sizeof(int));
	cudaMalloc(&dB, sizeB * sizeof(int));
	cudaMemset(dB, 0, sizeB * sizeof(int));
	cudaMalloc(&dC, sizeC * sizeof(int));
	cudaMemset(dC, 0, sizeC * sizeof(int));

	timer.onTimer(2);

	// 2. Send(Copy) the input matrices to GPU (A -> dB, B -> dB)
	// Hint: cudaMemcpy
	cudaMemcpy(dA, A, sizeA * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, sizeB * sizeof(int), cudaMemcpyHostToDevice);
	timer.offTimer(2);

	// 3. Set the thread layout
	// 
	// dim3 gridDim(?, ?);
	// dim3 blockDim(?, ?);
	dim3 gridDim(ceil((float)m / BLOCK_SIZE), ceil((float)n / BLOCK_SIZE));
	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

	printf("Grid(%d, %d), Block(%d, %d)\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);

	timer.onTimer(1);

	// 4. Kernel call
	//MatMul <<< gridDim, blockDim >>> (dA, dB, dC, m, n, k);
	MatMul<<<gridDim, blockDim>>>(dA, dB, dC, m, n, k);

	cudaDeviceSynchronize(); // this is synchronization for mearusing the kernel processing time
	timer.offTimer(1);

	timer.onTimer(3);

	//5. Get(copy) the result from GPU to host memory (dC -> Cgpu)
	// Hint: cudaMemcpy
	cudaMemcpy(Cgpu, dC, sizeC * sizeof(int), cudaMemcpyDeviceToHost);
	timer.offTimer(3);

	// 6. Release device memory space (dA, dB, dC)
	// Hint: cudaFree
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);

	/******************************************************************
	******************************************************************/
	timer.offTimer(4);

	compareMatrix(Ccpu, Cgpu, sizeC);
	timer.printTimer(1);

	delete A;
	delete B;
	delete Ccpu;
	delete Cgpu;

	return 0;
}


// Utility functions
bool compareMatrix(DATA_TYPE* _A, DATA_TYPE* _B, int _size)
{
	bool isMatched = true;
	for (int i = 0; i < _size; i++) {
		if (_A[i] != _B[i]) {
			printf("[%d] not matched! (%f, %f)\n", i, _A[i], _B[i]);
			getchar();
			isMatched = false;
		}
	}
	if (isMatched)
		printf("Results are matched!\n");
	else
		printf("Results are not matched!!!!!!!!!!!\n");

	return isMatched;
}

template<class T>
void allocNinitMem(T** p, long long size, double* memUsage) {
	*p = new T[size];
	memset(*p, 0, sizeof(T) * size);

	if (memUsage != NULL) {
		*memUsage += sizeof(T) * size;
	}
}