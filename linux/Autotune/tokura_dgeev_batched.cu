#include<stdio.h>
#include<stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include"tokura_blas_define.h"
#include"tokura_blas_const.h"
//#include"tokura_tuned_thread_parameters.h"


int tokura_dgeev_batched_MWB(tokuraInternalhandle_t* tokurahandle, int n, double** A, double** wr, double** wi, int batchCount);
int tokura_dgeev_batched_SWBMWB(tokuraInternalhandle_t* tokurahandle, int n, double** A, double** wr, double** wi, int batchCount);

int tokura_dgeev_batched_MWB_tune(tokuraInternalhandle_t* tokurahandle, int n, double** A, double** wr, double** wi, int batchCount, float* time);
int tokura_dgeev_batched_SWB_tune(tokuraInternalhandle_t* tokurahandle, int n, double** A, double** wr, double** wi, int batchCount,float* time);


void tokuraCreate(tokuraInternalhandle_t** tokurahandle)
{
	*tokurahandle = (tokuraInternalhandle_t*)malloc(sizeof(tokuraInternalhandle_t));
	cudaDeviceProp dev;
	cudaGetDeviceProperties(&dev, 0);
	(*tokurahandle)->WARP_SIZE = dev.warpSize;
	(*tokurahandle)->sharedsize = dev.sharedMemPerBlock;

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
	tokurahandle->wi_tmp_device =(double**)malloc(sizeof(double*)*NUMBER_OF_STREAMS);
	tokurahandle->stream = (cudaStream_t*)malloc(sizeof(cudaStream_t)*NUMBER_OF_STREAMS);

	for (i = 0; i < NUMBER_OF_STREAMS; i++)
	{
		cudaMalloc((void**)&tokurahandle->A_device[i], sizeof(double)*n*n*NUMBER_OF_COMPUTED_MATRICES_PER_STREAM);
		cudaMalloc((void**)&tokurahandle->A_tmp_device[i], sizeof(double)*n*n*NUMBER_OF_COMPUTED_MATRICES_PER_STREAM);
		cudaMalloc((void**)&tokurahandle->wr_device[i], sizeof(double)*n*NUMBER_OF_COMPUTED_MATRICES_PER_STREAM);
		cudaMalloc((void**)&tokurahandle->wi_device[i], sizeof(double)*n*NUMBER_OF_COMPUTED_MATRICES_PER_STREAM);
		cudaMalloc((void**)&tokurahandle->wr_tmp_device[i], sizeof(double)*n*NUMBER_OF_COMPUTED_MATRICES_PER_STREAM);
		cudaMalloc((void**)&tokurahandle->wi_tmp_device[i], sizeof(double)*n*NUMBER_OF_COMPUTED_MATRICES_PER_STREAM);

		cudaStreamCreate(tokurahandle->stream + i);
	}

}

void tokuraMemoryfree(tokuraInternalhandle_t* tokurahandle, int n, int batchCount)
{
	int i;


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


int tokura_dgeev_batched_MWBtune(int n, double** A, double** wr, double** wi, int batchCount, float* time)
{
	if (!(n > 0 && n <= 32))
	{
		//入力行列サイズチェック
		return -1;
	}
	if (A == NULL)
	{
		//入力行列のメモリ確保チェック
		return -2;
	}
	if (wr == NULL)
	{
		//出力固有値実部のメモリ確保チェック
		return -3;
	}
	if (wi == NULL)
	{
		//出力固有値実部のメモリ確保チェック
		return -4;
	}


	tokuraInternalhandle_t* tokurahandle;
	int threadnum;
	tokuraCreate(&tokurahandle);
	tokuraMemorymalloc(tokurahandle, n, batchCount);
	threadnum =tokura_dgeev_batched_MWB_tune(tokurahandle, n, A, wr, wi, batchCount,time);
	tokuraMemoryfree(tokurahandle, n, batchCount);


	tokuraDestroy(tokurahandle);
	return threadnum;//success

}


int tokura_dgeev_batched_SWBtune(int n, double** A, double** wr, double** wi, int batchCount, float* time)
{
	if (!(n > 0 && n <= 32))
	{
		//入力行列サイズチェック
		return -1;
	}
	if (A == NULL)
	{
		//入力行列のメモリ確保チェック
		return -2;
	}
	if (wr == NULL)
	{
		//出力固有値実部のメモリ確保チェック
		return -3;
	}
	if (wi == NULL)
	{
		//出力固有値実部のメモリ確保チェック
		return -4;
	}


	tokuraInternalhandle_t* tokurahandle;
	int threadnum;
	tokuraCreate(&tokurahandle);
	tokuraMemorymalloc(tokurahandle, n, batchCount);
	threadnum = tokura_dgeev_batched_SWB_tune(tokurahandle, n, A, wr, wi, batchCount, time);
	tokuraMemoryfree(tokurahandle, n, batchCount);


	tokuraDestroy(tokurahandle);
	return threadnum;//success

}