#include "cuda_runtime.h"
#include "device_launch_parameters.h"

int main(void) {
    // cudaMalloc 함수로 디바이스 메모리를 할당받을 수 있음
    // void** ptr: 할당된 디바이스 메모리의 시작 주소를 담을 포인터 변수
    // size_t size: 할당할 공간의 크기(바이트 단위)

    // cudaMalloc 함수는 cudaError_t 열거형 리턴
    // API 호출 성공하면 cudaSuccess(=0) 리턴
    // 실패하면 다른 에러 코드 리턴

    // CUDA 프로그램에서는 디바이스 메모리 영역 사용하는 변수 구분하기 위해
    // 변수 이름 앞에 d 접두사 붙이는 것이 일반적
    int* dDataPtr;
    // 디바이스 메모리에 int 32개 담을 공간 할당
    cudaMalloc(&dDataPtr, sizeof(int) * 32);

    // cudaMalloc으로 메모리 할당 받으면 쓰레기 값이 그대로 남아있음
    // cudaMemset 함수로 디바이스 메모리 공간을 특정 값으로 초기화 가능
    // void* ptr: 할당받은 디바이스 메모리 시작 주소
    // int value: 각 바이트의 초기화 값
    // size_t size: 초기화할 공간의 크기(바이트 단위)
    cudaMemset(dDataPtr, 0, sizeof(int) * 32); // 0으로 초기화

    // 할당받은 디바이스 메모리 모두 사용했으면 해제해야 함
    // cudaFree 함수로 디바이스 메모리 해제 가능
    // void* ptr: 해제할 디바이스 메모리 시작 주소

    // 리턴 값으로 cudaError_t 열거형 리턴
    cudaFree(dDataPtr);
}