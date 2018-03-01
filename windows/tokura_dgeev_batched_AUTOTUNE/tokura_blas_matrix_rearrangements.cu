#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"tokura_blas_define.h"
#include"tokura_blas_const.h"

//This kernel rearrenge matrix arrangement from Matrix wise to Elemental wise.
__global__ void tokura_matrixrearrangement_MWtoEW(const int n, const double* __restrict__ a, double *a2, const int batchCount_per_stream)
{
	const int tmp_mat_pad = blockIdx.x * 32;
	const int shared_pad = 32 + 1;
	const int n_n = n*n;

	if (!(tmp_mat_pad < batchCount_per_stream))
	{
		return;
	}


	const int remain_mat = (tmp_mat_pad + 32) < batchCount_per_stream ? 32 : (batchCount_per_stream - tmp_mat_pad);


	int i, j, k;
	int remain = 0;
	int tmp_remain;

	__shared__ double tmp_mat_shared[32 * shared_pad];


	while ((remain) < n_n)
	{
		tmp_remain = ((remain + 32) < n_n) ? 32 : (n_n - remain);

		if ((remain + (threadIdx.x&(32 - 1))) < n_n)
		{
			for (k = 0 + (threadIdx.x >> 5); k < remain_mat; k += (blockDim.x >> 5))
			{
				tmp_mat_shared[(threadIdx.x&(32 - 1)) + k*shared_pad] = a[remain + (threadIdx.x&(32 - 1)) + (tmp_mat_pad + k)*n_n];
			}
		}
		__syncthreads();

		if ((threadIdx.x&(32 - 1)) < remain_mat)
		{
			for (k = 0 + (threadIdx.x >> 5); k < tmp_remain; k += (blockDim.x >> 5)/*k++*/)
			{
				j = (remain + k) / n;
				i = (remain + k) % n;
			

				a2[(/*(remain + k)*/j + i*n)*batchCount_per_stream + (tmp_mat_pad + (threadIdx.x&(32 - 1)))] = tmp_mat_shared[k + ((threadIdx.x&(32 - 1)))*shared_pad];
			}
		}

		remain += 32;
		__syncthreads();

	}
}


//This kernel rearrenge eigenvalues arrangement from Elemental  wise to Matrix wise.
__global__ void tokura_eigenvaluesrearrangement_EWtoMW(const int n, const  int mat_num, const double* __restrict__ comp_real, const double* __restrict__ comp_imag, double *comp_real_out, double *comp_imag_out)
{
	const int tmp_mat_pad = blockIdx.x * 32;
	const int shared_pad = 32 + 1;
	const int n_n = n*n;

	if (!(tmp_mat_pad < mat_num))
	{
		return;
	}


	const int remain_mat = (tmp_mat_pad + 32) < mat_num ? 32 : (mat_num - tmp_mat_pad);


	int i, j, k;
	int remain = 0;
	int tmp_remain;

	extern __shared__ double shared_d_eig_MWB[];
	double *real_s = &shared_d_eig_MWB[0];
	double *imag_s = &shared_d_eig_MWB[n*shared_pad];


	if (threadIdx.x < remain_mat)
	{
		for (i = threadIdx.y; i < n; i += blockDim.y)
		{
			real_s[threadIdx.x + i*shared_pad] = comp_real[i*mat_num + tmp_mat_pad + threadIdx.x];
			imag_s[threadIdx.x + i*shared_pad] = comp_imag[i*mat_num + tmp_mat_pad + threadIdx.x];
		}

	}
	__syncthreads();

	if (threadIdx.x < n)
	{
		for (i = threadIdx.y; i < remain_mat; i += blockDim.y)
		{
			comp_real_out[threadIdx.x + (tmp_mat_pad + i)*n] = real_s[i + threadIdx.x*shared_pad];
			comp_imag_out[threadIdx.x + (tmp_mat_pad + i)*n] = imag_s[i + threadIdx.x*shared_pad];
		}
	}


}




//This kernel rearrenge matrix arrangement from Matrix wise to Row wise.
__global__ void tokura_matrixrearrangement_RWtoEW
(
	const int n, 
	const  int mat_num,
	const double* __restrict__ a, 
	double *a2
)
{
	const int tmp_mat_pad = blockIdx.x * 32;
	const int shared_pad = 32 + 1;
	const int n_n = n*n;



	const int remain_mat = (tmp_mat_pad + 32) <= mat_num ? 32 : (mat_num - tmp_mat_pad);


	int i, j, k;
	int remain = 0;
	int tmp_remain;


	__shared__ double tmp_mat_shared[32 * shared_pad];


	if (threadIdx.x < n)
	{
		for (i = threadIdx.y; i <remain_mat; i += blockDim.y)
		{
			tmp_mat_shared[i + threadIdx.x * shared_pad] = a[(tmp_mat_pad + i)*n + threadIdx.x + (blockIdx.y)*mat_num*n];
		}
	}

	__syncthreads();

	if (threadIdx.x< remain_mat)
	{
		for (i = threadIdx.y; i < n; i += blockDim.y)
		{

			a2[(blockIdx.y + i*n)*mat_num + tmp_mat_pad + threadIdx.x] = tmp_mat_shared[threadIdx.x + i * shared_pad];

		}
	}
	__syncthreads();
}


//This kernel rearrenge matrix arrangement from Matrix wise to Row wise.
__global__ void tokura_matrixrearrangement_MWtoRW
(
	int n,
	double *input,
	double *output,
	int mat_num
)
{

	const int shared_pad = 32 + 1;
	__shared__ double tmp_mat_shared[32 * shared_pad];

	int i;
	if (threadIdx.x < n)
	{
		for (i = threadIdx.y; i < n; i += blockDim.y)
		{
			tmp_mat_shared[threadIdx.x*shared_pad + i] = input[((i)*n + (threadIdx.x)) + blockIdx.x*n*n];
		}
	}

	__syncthreads();

	if (threadIdx.x < n)
	{
		for (i = threadIdx.y; i < n; i += blockDim.y)
		{
			output[threadIdx.x + i*mat_num*n + blockIdx.x*n] = tmp_mat_shared[threadIdx.x + i*shared_pad];
		}
	}
}

