// 블록 ID
#define BID_X blockIdx.x
#define BID_Y blockIdx.y
#define BID_Z blockIdx.z

// 스레드 ID
#define TID_X threadIdx.x
#define TID_Y threadIdx.y
#define TID_Z threadIdx.z

// 그리드 Dimension
#define GDIM_X gridDim.x
#define GDIM_Y gridDim.y
#define GDIM_Z gridDim.z

// 블록 Dimension
#define BDIM_X blockDim.x
#define BDIM_Y blockDim.y
#define BDIM_Z blockDim.z

// 한 블록 내 스레드 인덱스
#define TID_IN_BLOCK (TID_Z * (BDIM_X * BDIM_Y) + (TID_Y * BDIM_X) + TID_X)
#define NUM_THREAD_IN_BLOCK (BDIM_X * BDIM_Y * BDIM_Z)

// 1차원 그리드에서의 스레드 인덱스
#define GRID_1D_TID ((BID_X * NUM_THREAD_IN_BLOCK) + NUM_THREAD_IN_BLOCK)
// 2차원 그리드에서의 스레드 인덱스
#define GRID_2D_TID (BID_Y * (GDIM_X * NUM_THREAD_IN_BLOCK) + GRID_1D_TTD)
// 3차원 그리드(글로벌)에서의 스레드 인덱스
#define GLOBAL_TID (BID_Z * (GDIM_X * GDIM_Y * NUM_THREAD_IN_BLOCK) + GRID_2D_TID)