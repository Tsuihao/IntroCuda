#include <iostream>

// __gloabl__ c construct called "decleration specifier" let the cuda knows this is kernel code
// Caution: the input arguements of the kernel code needs to be allocated on the GPU (d_ naming convension helps to debug)
__global__ void square(float* d_out, float* d_in)
{
    int idx = threadIdx.x;   // here depends on the <<<block, threadPerBlock>>>,  build-in variable: threadIdx
    float f = d_in[idx];
    d_out[idx] = f * f;
}

__global__ void cubic(float* d_out, float* d_in)
{
    int idx = threadIdx.x; 
    float f = d_in[idx];
    d_out[idx] = f * f * f;
}

int main()
{
    const int ARRAY_SIZE = 64;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

    // generate the input array on the host
    float h_in[ARRAY_SIZE];
    for(int i(0); i < ARRAY_SIZE; ++i)
    {
        h_in[i] = float(i);
    }

    // decalre the output array on the host
    float h_out_square[ARRAY_SIZE];
    float h_out_cubic[ARRAY_SIZE];

    // decalre GPU memory pointers
    float* d_in;
    float* d_out_square;
    float* d_out_cubic;

    // allocate GPU memory
    cudaMalloc((void**) &d_in, ARRAY_BYTES);
    cudaMalloc((void**) &d_out_square, ARRAY_BYTES);
    cudaMalloc((void**) &d_out_cubic, ARRAY_BYTES);

    // transfer the array to the GPU
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

    // lauch the kernel
    square<<<1, ARRAY_SIZE>>>(d_out_square, d_in);
    cubic<<<1, ARRAY_SIZE>>>(d_out_cubic, d_in);

    // copy back the result array to the CPU
    cudaMemcpy(h_out_square, d_out_square, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_cubic, d_out_cubic, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    // print out the result
    std::cout<< "Square result :" << std::endl;
    for(int i(0); i < ARRAY_SIZE; ++i)
    {
        std::cout << h_out_square[i] << "\t";
    }

    std::cout << std::endl;
    
    std::cout<< "Cubic result :" << std::endl;
    for(int i(0); i < ARRAY_SIZE; ++i)
    {
        std::cout << h_out_cubic[i] << "\t";
    }

    // free GPU memory allocation
    cudaFree(d_in);
    cudaFree(d_out_square);
    cudaFree(d_out_cubic);

    return 0;

}