#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// The size of the vector
#define NUM_DATA 1030

// Simple vector sum kernel (Max vector size : 1024)
__global__ void vecAdd(int* _a, int* _b, int* _c) {
    int tID = threadIdx.x;
    _c[tID] = _a[tID] + _b[tID];
}

int main(void)
{
    int* a, * b, * c, * h_c;	// Vectors on the host
    int* d_a, * d_b, * d_c;	// Vectors on the device

    int memSize = sizeof(int) * NUM_DATA;
    printf("%d elements, memSize = %d bytes\n", NUM_DATA, memSize);

    // Memory allocation on the host-side
    a = new int[NUM_DATA]; memset(a, 0, memSize);
    b = new int[NUM_DATA]; memset(b, 0, memSize);
    c = new int[NUM_DATA]; memset(c, 0, memSize);
    h_c = new int[NUM_DATA]; memset(h_c, 0, memSize);

    // Data generation
    for (int i = 0; i < NUM_DATA; i++) {
        a[i] = rand() % 10;
        b[i] = rand() % 10;
    }

    // Vector sum on host (for performance comparision)
    for (int i = 0; i < NUM_DATA; i++)
        h_c[i] = a[i] + b[i];

    //****************************************//
    //******* Write your code - start ********//

    // 1. Memory allocation on the device-side (d_a, d_b, d_c)
    cudaMalloc(&d_a, memSize); cudaMemset(d_a, 0, memSize);
    cudaMalloc(&d_b, memSize); cudaMemset(d_b, 0, memSize);
    cudaMalloc(&d_c, memSize); cudaMemset(d_c, 0, memSize);

    // 2. Data copy : Host (a, b) -> Device (d_a, d_b)
    cudaMemcpy(d_a, a, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, memSize, cudaMemcpyHostToDevice);

    // 3. Kernel call
    // 블록의 최대 x 크기는 1024
    // 벡터의 크기가 1024보다 크다면 여러 개의 블록이 필요함
    // 그리드의 형태를 ceil(NUM_DATA/1024)로 설정해 필요한 데이터 수에 맞게
    // 블록이 만들어질 수 있게 함
    vecAdd<<<ceil(NUM_DATA/1024), 1024>>>(d_a, d_b, d_c);

    // 4. Copy results : Device (d_c) -> Host (c)
    cudaMemcpy(c, d_c, memSize, cudaMemcpyDeviceToHost);

    // 5. Release device memory (d_a, d_b, d_c)
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    //******** Write your code - end *********//
    //****************************************//

    // Check results
    bool result = true;
    for (int i = 0; i < NUM_DATA; i++) {
        if (h_c[i] != c[i]) {
            printf("[%d] The result is not matched! (%d, %d)\n"
                , i, h_c[i], c[i]);
            result = false;
        }
    }

    if (result)
        printf("GPU works well!\n");

    // Release host memory
    delete[] a; delete[] b; delete[] c;

    return 0;
}