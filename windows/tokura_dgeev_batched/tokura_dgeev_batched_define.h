#ifndef __TOKURA_DGEEV_BATCHED_STRUCT_DEFINE__
#define __TOKURA_DGEEV_BATCHED_STRUCT_DEFINE__


typedef struct 
{
	//1 GPU conputation environments
	double** A_device;//matrix array in device memory
	double** A_tmp_device;//matrix array in device memory for matrix rearrangement
	double** wr_device;//real part of eigenvelue array in device memory
	double** wr_tmp_device;//real part of eigenvelue array in device memoryfor matrix rearrangement
	double** wi_device;//imaginary  part of eigenvelue array in device memory
	double** wi_tmp_device;//imaginary  part of eigenvelue array in device memoryfor matrix rearrangement
	cudaStream_t *stream;//CUDA stream for data transfer hiding
	
	char** flags;//flag array
	//flags[i]==0:eigenvalues of i-th matrix are computed correctly
	//flags[i]==1:eigenvalues of i-th matrix are NOT computed correctly

	int WARP_SIZE;
	
}tokuraInternalhandle_t;


#endif