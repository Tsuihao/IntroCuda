/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"

__global__ void reduce_max_kernel(const float* const d_in, float* d_out)
{
	extern __shared__ float block_memory[]; // local memory share among the same block

	int local_t_idx = threadIdx.x;
    int global_t_idx = threadIdx.x + blockIdx.x * blockDim.x;

    // copy from global memory to shared memory  -> 2020 Ampere can by-pass directly to shared memory
	block_memory[local_t_idx] = d_in[global_t_idx];
	__syncthreads();
   
   // do reduction in shared mem
   for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
   {
      if(local_t_idx < s) // take the first half
      { 
        block_memory[local_t_idx] = max(block_memory[local_t_idx], block_memory[local_t_idx + s]);	
	  }
	  __syncthreads();
	}
	
    // assgin only the first element (since it is the result) to the blockId
	if(local_t_idx == 0)
		d_out[blockIdx.x] = block_memory[local_t_idx];
}

__global__ void reduce_min_kernel(const float* const d_in, float* d_out)
{
	extern __shared__ float block_memory[];

	int local_t_idx = threadIdx.x;
    int global_t_idx = threadIdx.x + blockIdx.x * blockDim.x;

	block_memory[local_t_idx] = d_in[global_t_idx];
	__syncthreads();
   
   for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
   {
      if(local_t_idx < s)
      { 
          block_memory[local_t_idx] = min(block_memory[local_t_idx], block_memory[local_t_idx + s]); // the only diff compare to reduce_max_kernel
	  }
		__syncthreads();
	}

	if(local_t_idx == 0)
		d_out[blockIdx.x] = block_memory[local_t_idx];
}

void reduce(const float* const d_in, float& min_logLum, float& max_logLum, const size_t numRows, const size_t numCols)
{

	const int blockSize = numCols;  // This code will break if the width of the image is over than 1024 (max of SM)
   	const int gridSize  = numRows;

   /*
    * -------------------------------------------------------------------------------------------------------
    * |     numCols    |    numCols    | ......   
    * -------------------------------------------------------------------------------------------------------
    */
   
	// declare device memory pointers
	float * d_intermediate, *d_max, *d_min;
		
	// allocate device memory
	checkCudaErrors(cudaMalloc((void **) &d_intermediate, gridSize*sizeof(float)));  // the intermediate result is each Block -> gridSize
	checkCudaErrors(cudaMalloc((void **) &d_max, sizeof(float)));
	checkCudaErrors(cudaMalloc((void **) &d_min, sizeof(float)));

	// Find maximum (perform 2 times of reduce)
	// Reduce 1- find the maximum in each block
	reduce_max_kernel<<<gridSize, blockSize, blockSize*sizeof(float)>>>(d_in, d_intermediate);
	// Reduce 2- find the global maximum
	reduce_max_kernel<<<1, gridSize, gridSize*sizeof(float)>>>(d_intermediate, d_max);
    // Clean up the d_iintermediate
    checkCudaErrors(cudaMemset(d_intermediate, 0, gridSize*sizeof(float)));
   
	// Find minimum (perform 2 times of reduce)
	// Reduce 1- find the minimum in each block
	reduce_min_kernel<<<gridSize, blockSize, blockSize*sizeof(float)>>>(d_in, d_intermediate);
	// Reduce 2- find the global maximum
	reduce_min_kernel<<<1, gridSize, gridSize*sizeof(float)>>>(d_intermediate, d_min);

	// Copy max/min result back to Host
	checkCudaErrors(cudaMemcpy(&max_logLum, d_max, sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&min_logLum, d_min, sizeof(float), cudaMemcpyDeviceToHost));

	// free device memory
	checkCudaErrors(cudaFree(d_intermediate));
	checkCudaErrors(cudaFree(d_max));
	checkCudaErrors(cudaFree(d_min));

	return;	
}


__global__ void histogram(const float* const d_in, unsigned int * const d_out, const float logLumRange, const int min_logLum, const int numBins)
{
	int g_t_idx = blockIdx.x * blockDim.x + threadIdx.x;
	float value = d_in[g_t_idx];
	int bin_idx = (value - min_logLum) / logLumRange * numBins;

	atomicAdd(&(d_out[bin_idx]), 1);
}


__global__ void SCAN_HS(const unsigned int * const d_in, unsigned int * const d_out)
{
	/*	Hillis Steele Scan
		for d := 1 to log2n do
			forall k in parallel do
		 		if k ≥ 2^d then
					x[out][k] := x[in][k − 2^d-1] + x[in][k]
		 		else
					x[out][k] := x[in][k]
		 	swap(in,out) 
		This version can handle arrays only as large as can be processed by a single thread block running 
		on one multiprocessor of a GPU
	*/
	extern __shared__ unsigned int temp[];

	int tid = threadIdx.x;
	int pout = 0, pin = 1;

	// exclusicve scan
	temp[tid] =  tid > 0? d_in[tid-1] : 0;
	// make sure all data in this block are loaded into shared shared memory
	__syncthreads();
	
	for(unsigned int stride = 1; stride < blockDim.x; stride <<= 1){
		// swap double buffer indices
		pout = 1 - pout;
		pin  = 1 - pout;

		if(tid >= stride)
			temp[pout*blockDim.x+tid] = temp[pin*blockDim.x+tid] + temp[pin*blockDim.x+tid - stride];
		else
			temp[pout*blockDim.x+tid] = temp[pin*blockDim.x+tid];
		// make sure all operations at one stage are done!
		__syncthreads();
	}

	d_out[tid] = temp[pout*blockDim.x + tid];	
}


__global__ void SCAN_BL(const unsigned int * const d_in, unsigned int * const d_out, const int nums)
{
	/* Blelloch Scan : Up-Sweep(reduce) + Down-Sweep
		Up-Sweep:
		for d := 0 to log2n - 1 do
			for k from 0 to n – 1 by 2^(d+1) in parallel do
				x[k + 2^(d + 1) - 1] := x[k + 2^d - 1] + x [k + 2^(d+1) - 1] 

		Down-Sweep:
		x[n - 1] := 0
		for d := log2n down to 0 do
			for k from 0 to n – 1 by 2^(d+1) in parallel do
				t := x[k + 2^d- 1]
				x[k + 2^d - 1] := x [k + 2^(d+1) - 1]
				x[k + 2^(d+1) - 1] := t + x [k + 2^(d+1) - 1] 
	*/
	extern __shared__ unsigned int temp[];

	int tid = threadIdx.x;

	// exclusicve scan
	temp[2*tid] = d_in[2*tid];
	
	if(2*tid+1 < nums)
		temp[2*tid+1] = d_in[2*tid+1];
	else
		temp[2*tid+1] = 0;

	// make sure all data in this block are loaded into shared memory
	__syncthreads();
	
	int stride = 1;
	// reduce step
	for(unsigned int d = blockDim.x; d > 0; d >>= 1){
		if(tid < d){	
			int idx1 = (2*tid+1)*stride - 1;
			int idx2 = (2*tid+2)*stride - 1;
			temp[idx2] += temp[idx1];
		}
		stride *= 2;
		// make sure all operations at one stage are done!
		__syncthreads();
	}

	// Downsweep Step
	// set identity value
	if(tid == 0)
		temp[nums-1] = 0;
	for(unsigned int d = 1; d < nums; d <<= 1){
		stride >>= 1;
		// make sure all operations at one stage are done!
		__syncthreads();
		if( tid < d){
			int idx1 = (2*tid+1)*stride - 1;
			int idx2 = (2*tid+2)*stride - 1;
			unsigned int tmp  = temp[idx1];
			temp[idx1] = temp[idx2];
			temp[idx2] += tmp;
		}		
	}
	// make sure all operations at the last  stage are done!
	__syncthreads();
	d_out[2*tid] = temp[2*tid];
	if(2*tid+1 < nums)
		d_out[2*tid+1] = temp[2*tid+1];
}

// Scan algorithm from Course : Hetergeneous Parallel Programming
__global__ void SCAN_HPP(const unsigned int * const d_in, unsigned int * const d_out, const int nums)
{

	extern __shared__ unsigned int temp[];

	int tid = threadIdx.x;

	// exclusicve scan
	if(tid == 0){
		temp[2*tid] = 0;
		temp[2*tid+1] = d_in[2*tid];	
	}
	else{
		temp[2*tid] = d_in[2*tid-1];
		if(2*tid+1 < nums)
			temp[2*tid+1] = d_in[2*tid];
		else
			temp[2*tid+1] = 0;
	}
	// make sure all data in this block are loaded into shared shared memory
	__syncthreads();
	
	// Reduction Phase
	for(unsigned int stride = 1; stride <= blockDim.x; stride <<= 1){
		// first update all idx == 2n-1, then 4n-1, then 8n-1 ...  
		// finaly 2(blockDim.x/2) * n - 1(only 1 value will be updated partial[blockDim.x-1])
		int idx = (tid+1)*stride*2 - 1;
		if( idx  < 2*blockDim.x)
			temp[idx] += temp[idx-stride];
		// make sure all operations at one stage are done!
		__syncthreads();
	}
	// Example:
	// After reduction phase , position at 0, 1, 3, 7, ... has their final values (blockDim.x == 8)
	// then we update values reversely.
	// first use position 3's value to update position 5(stride == 2 == blockDim.x/4, idx == 3 == (0+1)*2*2-1, only 1 thread do calculation)
	// then use position 1 to update postion 2 , position 3 to update position 4, position 5 to update position 6
	//			(stride == 1 == blockDim.x/8, idx == (0+1)*1*2-1=1,(1+1)*1*2-1=3, (2+1)*1*2-1=5, 3 threads do calculation)

	// Post Reduction Reverse Phase
	for(unsigned int stride = blockDim.x/2; stride > 0; stride >>= 1){
		// first update all idx == 2(blockDim.x/4) * n - 1 + blockDim.x/4, 
		// then 2(blockDim.x/8)n-1+blockDim.x/8, then 2(blockDim.x/16)n-1 + blockDim.x/16...  
		// finaly 2 * n - 1
		int idx = (tid+1)*stride*2 - 1;
		if( idx + stride  < 2*blockDim.x)
			temp[idx + stride] += temp[idx];
		// make sure all operations at one stage are done!
		__syncthreads();
	}

	// exclusive scan

	d_out[2*tid] = temp[2*tid];
	if(2*tid+1 < nums)
		d_out[2*tid+1] = temp[2*tid+1];
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

	
	// Step 1 : find minimum and maximum value
	reduce(d_logLuminance, min_logLum, max_logLum, numRows, numCols);

	// Step 2: find the range 
	float logLumRange = max_logLum - min_logLum;

	// Step 3 : generate a histogram of all the values
	
	// declare bin number on device memory
	unsigned int  *d_bins;
	// allocate device memory
	checkCudaErrors(cudaMalloc((void **) &d_bins, numBins*sizeof(unsigned int)));
	checkCudaErrors(cudaMemset(d_bins, 0, numBins*sizeof(unsigned int)));
	histogram<<<numRows, numCols>>>(d_logLuminance, d_bins, logLumRange, min_logLum, numBins);
	
	// Step 4 : SCAN to compute the cumulative distribution
	//SCAN_HS<<<1, numBins, numBins*sizeof(unsigned int)>>>(d_bins, d_cdf);
	//SCAN_HPP<<<1, ceil(numBins/2), numBins*sizeof(unsigned int)>>>(d_bins, d_cdf, numBins);
	SCAN_BL<<<1, ceil(numBins/2), numBins*sizeof(unsigned int)>>>(d_bins, d_cdf, numBins);
	
	// free GPU memory allocation
	checkCudaErrors(cudaFree(d_bins));
}