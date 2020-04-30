// doing 1024 * 1024 element's reducing
// 1024 blocks with 1024 threads  ->  1 block with 1024 threads -> result

#include <iostream>

/*
 * This kernel uses global memory, which can be optimized
 */
__global__
void global_reduce_kernel(int* g_in, int* g_out)
{
    int global_t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int local_t_idx = threadIdx.x;

    // do reduction in global mem
    // s >>= 1  is s = s >> 1  is  s /=2
    for(unsigned int s = blockDim.x/2; s > 0; s >>= 1)
    {
        if(local_t_idx < s)  // take the first half
        {
            g_in[global_t_idx] += g_in[global_t_idx + s];  // s here is the buffer, draw the diagram will help to understand
        }
        __syncthreads();   
    }

    // assgin only the first element, since it is the result of summation
    if(local_t_idx == 0)
    {
        g_out[blockIdx.x] = g_in[global_t_idx];
    }
}

/*
 * This kernel uses shared memory
 */
 __global__
 void share_reduce_kernel(int* g_in, int* g_out)
 {
    extern __shared__ int block_memory[]; // local memory share among the same block
    int global_t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int local_t_idx = threadIdx.x;

    // copy from g_in to shared
    block_memory[local_t_idx] = g_in[global_t_idx];
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s = blockDim.x/2; s > 0; s >>= 1)
    {
        if(local_t_idx < s)  // take the first half
        {
            block_memory[local_t_idx] += block_memory[local_t_idx + s];  // s here is the buffer, draw the diagram will help to understand
        }
        __syncthreads();   
    }

    // assgin only the first element, since it is the result of summation
    if(local_t_idx == 0)
    {
        g_out[blockIdx.x] = block_memory[0];
    }
 }

// logger
void trace(const int* array, const size_t size)
{
    for(size_t i(0); i < size; ++i)
    {
        std::cout << array[i] << "\t";
        
        // formatting
        if(i%8 == 0)
        {
            std::cout << std::endl;
        }
    }
}

int main()
{
    // declare on host memeory
    int h_array_in[1024*1024];
    int h_array_out[1024];
    int* d_array_in;  // 1024 * 1024
    int* d_array_out;  // 1024

    // cudaevent
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // dummy-values for host arrary
    for(size_t i = 0; i < 1024*1024; ++i)
    {
        h_array_in[i] = static_cast<int>(i);
    }

    // allocate device memory
    cudaMalloc(&d_array_in, 1024*1024*sizeof(int));
    cudaMemset(&d_array_in, 0, 1024*1024*sizeof(int));
    cudaMalloc(&d_array_out, 1024*sizeof(int));
    cudaMemset(&d_array_out, 0, 1024*sizeof(int));

    // copy the host_array into device
    cudaMemcpy(d_array_in, h_array_in, 1024*1024*sizeof(int), cudaMemcpyHostToDevice);

    // launch kernel
    dim3 gridSize(1024, 1, 1);
    dim3 blockSize(1024, 1, 1);
    cudaEventRecord(start);
    share_reduce_kernel<<<gridSize, blockSize, 1024*sizeof(int) /*allocated share memory size*/>>>(d_array_in, d_array_out); 
    //global_reduce_kernel<<<gridSize, blockSize>>>(d_array_in, d_array_out)
    cudaEventRecord(stop);

    // copy back the result from device to host
    cudaMemcpy(h_array_out, d_array_out, 1024*sizeof(int), cudaMemcpyDeviceToHost);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Elapsed time = " << milliseconds << std::endl;
    trace(h_array_out, 1024);

    return 0;
}