#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void printData(int* dDataPtr) {
    printf("%d", dDataPtr[threadIdx.x]);
}

__global__ void setData(int* dDataPtr) {
    dDataPtr[threadIdx.x] = 2;
}

int main(void) {
    int data[10] = { 0, };
    for (int i = 0; i < 10; i++)
	data[i] = 1;

    int* dDataPtr;
    cudaMalloc(&dDataPtr, sizeof(int) * 10);
    cudaMemset(dDataPtr, 0, sizeof(int) * 10);

    printf("Data in device: ");
    printData <<<1, 10 >>>(dDataPtr);
    
    // 호스트와 디바이스는 서로 별개의 메모리를 가지므로 복사가 필요함
    // cudaMemcpy로 값을 복사할 수 있음
    // c의 memcpy와 비슷하나 마지막에 kind를 잘 설정해줘야 함
    // cudaMemcpyHostToHost: 호스트(src) -> 호스트(dst)
    // cudaMemcpyHostToDevice: 호스트 -> 디바이스
    // cudaMemcpyDeviceToHost: 디바이스 -> 호스트
    // cudaMemcpyDeviceToDevice: 디바이스 -> 디바이스
    // cudaMemcpyDefault: unified virtual addressing 지원 시스템에서 자동 결정 (성능 떨어짐)

    // 호스트 메모리에서 디바이스 메모리로 값 복사
    cudaMemcpy(dDataPtr, data, sizeof(int) * 10, cudaMemcpyHostToDevice);
    printf("\nHost -> Device: ");
    printData <<<1, 10 >>>(dDataPtr);

    setData<<<1,10>>>(dDataPtr);

    // 디바이스 메모리에서 호스트 메모리로 값 복사
    cudaMemcpy(data, dDataPtr, sizeof(int) * 10, cudaMemcpyDeviceToHost);
    printf("\nDevice -> Host: ");
    for (int i = 0; i < 10; i++)
	printf("%d", data[i]);

    cudaFree(dDataPtr);
}