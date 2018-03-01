#ifndef __TOKURABLAS_HANDLER__
#define __TOKURABLAS_HANDLER__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"tokura_dgeev_batched_define.h"
#include"tokura_dgeev_batched_const.h"
void tokuraCreate(tokuraInternalhandle_t** tokurahandle)
{
	*tokurahandle = (tokuraInternalhandle_t*)malloc(sizeof(tokuraInternalhandle_t));
	cudaDeviceProp dev;
	cudaGetDeviceProperties(&dev, 0);
	(*tokurahandle)->WARP_SIZE = dev.warpSize;

}
void tokuraDestroy(tokuraInternalhandle_t* tokurahandle)
{
	free(tokurahandle);
}
void tokuraMemorymalloc(tokuraInternalhandle_t* tokurahandle, int n, int batchCount)
{
	int i;

	tokurahandle->A_device = (double**)malloc(sizeof(double*)*NUMBER_OF_STREAMS);
	tokurahandle->A_tmp_device = (double**)malloc(sizeof(double*)*NUMBER_OF_STREAMS);
	tokurahandle->wr_device = (double**)malloc(sizeof(double*)*NUMBER_OF_STREAMS);
	tokurahandle->wr_tmp_device = (double**)malloc(sizeof(double*)*NUMBER_OF_STREAMS);
	tokurahandle->wi_device = (double**)malloc(sizeof(double*)*NUMBER_OF_STREAMS);
	tokurahandle->wi_tmp_device = (double**)malloc(sizeof(double*)*NUMBER_OF_STREAMS);
	tokurahandle->stream = (cudaStream_t*)malloc(sizeof(cudaStream_t)*NUMBER_OF_STREAMS);
	tokurahandle->flags = (char**)malloc(sizeof(char*)*NUMBER_OF_STREAMS);


	for (i = 0; i < NUMBER_OF_STREAMS; i++)
	{
		cudaMalloc((void**)&tokurahandle->A_device[i], sizeof(double)*n*n*NUMBER_OF_COMPUTED_MATRICES_PER_STREAM);
		cudaMalloc((void**)&tokurahandle->A_tmp_device[i], sizeof(double)*n*n*NUMBER_OF_COMPUTED_MATRICES_PER_STREAM);
		cudaMalloc((void**)&tokurahandle->wr_device[i], sizeof(double)*n*NUMBER_OF_COMPUTED_MATRICES_PER_STREAM);
		cudaMalloc((void**)&tokurahandle->wi_device[i], sizeof(double)*n*NUMBER_OF_COMPUTED_MATRICES_PER_STREAM);
		cudaMalloc((void**)&tokurahandle->wr_tmp_device[i], sizeof(double)*n*NUMBER_OF_COMPUTED_MATRICES_PER_STREAM);
		cudaMalloc((void**)&tokurahandle->wi_tmp_device[i], sizeof(double)*n*NUMBER_OF_COMPUTED_MATRICES_PER_STREAM);
		cudaMalloc((void**)&tokurahandle->flags[i], sizeof(char)*NUMBER_OF_COMPUTED_MATRICES_PER_STREAM);

		cudaStreamCreate(tokurahandle->stream + i);


	}
}
void tokuraMemoryfree(tokuraInternalhandle_t* tokurahandle, int n, int batchCount)
{
	int i;

	for (i = 0; i < NUMBER_OF_STREAMS; i++)
	{
		cudaStreamSynchronize(tokurahandle->stream[i]);

	}
	for (i = 0; i < NUMBER_OF_STREAMS; i++)
	{
		cudaFree(tokurahandle->A_device[i]);
		cudaFree(tokurahandle->A_tmp_device[i]);
		cudaFree(tokurahandle->wr_device[i]);
		cudaFree(tokurahandle->wi_device[i]);
		cudaFree(tokurahandle->wr_tmp_device[i]);
		cudaFree(tokurahandle->wi_tmp_device[i]);
		cudaStreamDestroy(tokurahandle->stream[i]);
	}

	free(tokurahandle->A_device);
	free(tokurahandle->A_tmp_device);
	free(tokurahandle->wr_device);
	free(tokurahandle->wr_tmp_device);
	free(tokurahandle->wi_device);
	free(tokurahandle->wi_tmp_device);
	free(tokurahandle->stream);

}
#endif