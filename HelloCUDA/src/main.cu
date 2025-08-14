// cuda_runtime.h.와 device_launch_parameters.h는
// CUDA 런타임 API를 사용해서 프로그램 작성하기 위한 중요한 정의들이 있음
// 항상 추가해준다고 생각하면 됨

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

// __global__은 호스트(CPU)에서 호출하고, 
// 디바이스(GPU)에서 실행되는 함수임을 지정하는 키워드
// 함수의 호출자와 실행 공간을 지정하는 키워드는 세 가지가 있음
//   키워드   | 함수의 호출자 | 실행 공간
//  __host__        host          host
// __device__      device        device
// __global__      global        device

// 아무 것도 적지 않으면 기본 값은 __host__
// 만약 호스트와 디바이스 모두에서 사용하고 싶으면 __host__ __device__같이
// 두 개를 나란히 적어주면 됨

// __global__로 지정된 함수는 호스트에서 호출하고 디바이스에서 실행되는 함수
// CUDA에서는 이를 커널이라 함
__global__ void helloCUDA(void) {
    printf("Hello CUDA from GPU!\n");
}

int main(void)
{
    // CPU(호스트)에서 실행되는 호스트 코드
    // CPU가 GPU의 디바이스 코드(커널)을 실행함

    printf("Hello GPU from CPU!\n");

    // 커널 호출 시에는 연산을 수행할 CUDA 스레드의 수를 지정해야 함
    // CUDA 스레드 수는 <<<>>> 문법을 통해 지정함 (실행 구성 문법)
    // CUDA 스레드는 특정 단위로 그룹을 이루고, 그룹들은 계층적으로 구성
    // 아래 코드는 10개의 CUDA 스레드에게 helloCUDA 커널을 수행하라는 의미
    helloCUDA<<<1, 10 >>>();
    return 0;
}