﻿#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

void checkDeviceMemory(void) {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    printf("Device memory (free/total) = %lld/%lld bytes\n", free, total);
}

int main(void) {
    int* dDataPtr;
    cudaError_t errorCode;

    // 디바이스 메모리 할당
    checkDeviceMemory();
    errorCode = cudaMalloc(&dDataPtr, sizeof(int) * 1024 * 1024);
    printf("cudaMalloc - %s\n", cudaGetErrorName(errorCode));
    checkDeviceMemory();

    // 메모리 초기화
    errorCode = cudaMemset(dDataPtr, 0, sizeof(int) * 1024 * 1024);
    printf("cudaMemset - %s\n", cudaGetErrorName(errorCode));

    errorCode = cudaFree(dDataPtr);
    printf("cudaFree - %s\n", cudaGetErrorName(errorCode));
    checkDeviceMemory();
}