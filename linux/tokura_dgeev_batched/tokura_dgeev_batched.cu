#ifndef __TOKURADGEEV_BATCHED__
#define __TOKURADGEEV_BATCHED__

#include<stdio.h>
#include<stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "tokura_dgeev_batched_define.h"
#include "tokura_dgeev_batched_const.h"
#include "tokura_dgeev_batched_tuned_thread_parameters.h"

#include "tokura_dgeev_batched_MWB.cuh"
#include "tokura_dgeev_batched_SWBMWB.cuh"
#include "tokura_dgeev_batched_handler.cuh"


#ifdef WIN64
__declspec(dllexport)int tokura_dgeev_batched(int n, double** A, double** wr, double** wi, int batchCount, char* flags);
__declspec(dllexport)void tokura_malloc(double **A, int size, int  batchCount);
__declspec(dllexport)void tokura_flags_malloc(char** flags, int batchCount);
__declspec(dllexport)void tokura_free(double** A);
__declspec(dllexport)void tokura_flags_free(char* flags);
__declspec(dllexport)void tokura_cudaHostRegister(double** mat, int size);
__declspec(dllexport)void tokura_cudaHostUnregister(double* mat);
#endif
#ifdef __unix__
int tokura_dgeev_batched(int n, double** A, double** wr, double** wi, int batchCount,char* flags);
void tokura_malloc(double **A, int size, int  batchCount);
void tokura_flags_malloc(char** flags, int batchCount);
void tokura_free(double** A);
void tokura_flags_free(char* flags);
void tokura_cudaHostRegister(double** mat, int size);
void tokura_cudaHostUnregister(double* mat);

#endif
//For MATLAB
void tokura_cudaHostRegister(double** mat, int size)
{
	cudaHostRegister(*mat, sizeof(double)*size, cudaHostAllocDefault);
}
//For MATLAB
void tokura_cudaHostUnregister(double* mat)
{
	cudaHostUnregister(mat);
}
//Host memory allocation for matrices or eigenvalues
void tokura_malloc(double **A, int size, int  batchCount)
{
	int i;
	double** A_pointer = A;
	double *tmp_pointer;
	cudaMallocHost(&tmp_pointer, size*batchCount * sizeof(double));
	for (i = 0; i < batchCount; i++)
	{
		//	cudaMallocHost(&A[i], size* sizeof(double));

		A[i] = &tmp_pointer[i*size];
	}
}
//Host memory allocation for flags
void tokura_flags_malloc(char** flags, int batchCount)
{
	cudaMallocHost(flags, batchCount * sizeof(char));

}
//Host memory free for matrices or eigenvalues
void tokura_free(double** A)
{

	cudaFreeHost(A[0]);

}

//Host memory free for flags
void tokura_flags_free(char* flags)
{
	cudaFreeHost(flags);
}

//Helper function
//This function switch thread assginents 
//Switching is depended on matrix size.
int tokura_dgeev_batched(int n, double** A, double** wr, double** wi,int batchCount,char* flags)
{
	if (!(n > 0 && n<= MAX_MATRIX_SIZE))
	{
		return -1;
	}
	if (A == NULL)
	{
		return -2;
	}
	if (wr == NULL)
	{
		return -3;
	}
	if (wi == NULL)
	{
		return -4;
	}
	if (!(batchCount > 0))
	{
		return -5;
	}
	if (flags == NULL)
	{
		return -6;
	}


	tokuraInternalhandle_t* tokurahandle;
	tokuraCreate(&tokurahandle);
	tokuraMemorymalloc(tokurahandle, n, batchCount);

	int i;
	

	if (n < TOKURA_SWITCH_ALGORITHM_MATRIXSIZE)
	{
		tokura_dgeev_batched_MWB(tokurahandle, n, A, wr, wi, batchCount, flags);
	}
	else
	{
		tokura_dgeev_batched_SWBMWB(tokurahandle, n, A, wr, wi, batchCount, flags);
	}

	tokuraMemoryfree(tokurahandle, n, batchCount);

	tokuraDestroy(tokurahandle);


	return 0;
}
#endif