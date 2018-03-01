#ifndef __TOKURABLAS_DGEHRD_BATCHED_MWB_KERNEL__
#define __TOKURABLAS_DGEHRD_BATCHED_MWB_KERNEL__
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"tokura_dgeev_batched_define.h"
#include"tokura_dgeev_batched_const.h"


//DGEHRD reduces a DOUBLE PRECISION general matrix A 
//to upper Hessenberg form H by an orthogonal similarity transformation
__global__ void tokura_dgehrd_batched_MWB
(
	const int n,
	const  int mat_num,
	double*  a,
	const int HOUSE_KERNEL_THREAD_JOIN_NUM_VAR
)
{
	const int matrix_id = threadIdx.x + threadIdx.z*blockDim.x;
	const int warpid = threadIdx.y;
	const int mat_pad = (blockIdx.x*blockDim.x*blockDim.z) + matrix_id;
	const int MATRIX_NUM_PER_BLOCK = blockDim.x*blockDim.z;

	int i, j, k;
	int flag;
	double tmp;
	double tmp_sum;


	double tau;

	double vex_x_norm;
	char house_pm;


	extern __shared__ double shared_dynamic[];

	double *house = &shared_dynamic[matrix_id];
	double *sum_tmp_shared = &shared_dynamic[(n - 1) * MATRIX_NUM_PER_BLOCK];



	if (!((mat_pad < mat_num)))
	{
		return;
	}

	for (i = 10; i > 0; i--)
	{
		if ((HOUSE_KERNEL_THREAD_JOIN_NUM_VAR >> i) == 1)
		{
			break;
		}
	}

	if ((1 << (i)) == HOUSE_KERNEL_THREAD_JOIN_NUM_VAR)
	{
		i = i - 1;
	}
	if (i < 0)
	{
		i = 0;
	}
	if (HOUSE_KERNEL_THREAD_JOIN_NUM_VAR == 1)
	{
		i = 0;
	}



	const int REDUCTION_JOIN_THREADS_NUM = 1 << (i);


	for (k = 1; k < n - 1; k++)
	{
		tmp_sum = 0.0;
		i = k + 1 + warpid;
		while (i < n)
		{
			tmp = a[((k - 1) + i*n)*mat_num + mat_pad];
			house[ ((i - 1)*MATRIX_NUM_PER_BLOCK)] = tmp;
			tmp_sum += tmp*tmp;
			i += HOUSE_KERNEL_THREAD_JOIN_NUM_VAR;
		}


		sum_tmp_shared[threadIdx.x + threadIdx.z*blockDim.x + threadIdx.y*blockDim.x*blockDim.z] = tmp_sum;


		__syncthreads();



		tmp_sum = 0.0;
		i = REDUCTION_JOIN_THREADS_NUM;
		while (i != 0)
		{

			if (warpid < i && ((warpid + i) < HOUSE_KERNEL_THREAD_JOIN_NUM_VAR))
			{
				sum_tmp_shared[(warpid)*MATRIX_NUM_PER_BLOCK + matrix_id] += sum_tmp_shared[(warpid + i)*MATRIX_NUM_PER_BLOCK + matrix_id];
			}
			__syncthreads();
			i = i >> 1;
		}

		tmp_sum = sum_tmp_shared[matrix_id];

		if (tmp_sum == 0.0)
		{
			continue;
		}


		if (warpid == (HOUSE_KERNEL_THREAD_JOIN_NUM_VAR - 1)/*0*/)
		{
			tau = a[((k - 1) + (k)*n)*mat_num + mat_pad];

			tmp = tau*tau + tmp_sum;
			tmp = sqrt(tmp);
			vex_x_norm = tmp;
			house_pm = (char)(tau > 0.0);
			house[/*matrix_id +*/ ((k - 1) *MATRIX_NUM_PER_BLOCK)] = tau + (house_pm ? tmp : -tmp);

		}
		__syncthreads();

		tmp = house[ ((k - 1) *MATRIX_NUM_PER_BLOCK)] * house[ ((k - 1)*MATRIX_NUM_PER_BLOCK)] + tmp_sum;

		tmp_sum = tmp;

		flag = (tmp_sum != 0.0);
		if (flag)
		{
			tau = 2.0 / tmp;
		}
		if (flag)
		{
			//matrix multication from left at similarity transformation
			if (warpid == (HOUSE_KERNEL_THREAD_JOIN_NUM_VAR - 1))
			{
				i = k + 1 + warpid;
				j = k - 1;
				while (i < n)
				{
					a[(j + i*n)*mat_num + mat_pad] = 0.0;
					i += 1;
				}

				a[((k - 1) + (k)*n)*mat_num + mat_pad] = (house_pm ? -vex_x_norm : vex_x_norm);
			}

			{

				j = k - 1 + 1 + warpid;
				while (j < n)
				{

					tmp = house[((k - 1) *MATRIX_NUM_PER_BLOCK)] * a[(j + k*n)*mat_num + mat_pad];

					for (i = k + 1; i < n; i++)
					{
						tmp += house[((i - 1)*MATRIX_NUM_PER_BLOCK)] * a[(j + i*n)*mat_num + mat_pad];
					}
					tmp *= tau;
					for (i = n - 1; i >= k; i--)
					{
						a[(j + i*n)*mat_num + mat_pad] -= house[((i - 1) *MATRIX_NUM_PER_BLOCK)] * tmp;
					}
					j += HOUSE_KERNEL_THREAD_JOIN_NUM_VAR;
				}
			}
		}
		__syncthreads();

		if (flag)
		{
			//matrix multication from right at similarity transformation
			i = 0 + warpid;
			while (i < n)
			{
				tmp = house[((k - 1) *MATRIX_NUM_PER_BLOCK)] * a[(k + i*n)*mat_num + mat_pad];
				for (j = k + 1; j < n; j++)
				{
					tmp += house[((j - 1) *MATRIX_NUM_PER_BLOCK)] * a[(j + i*n)*mat_num + mat_pad];
				}
				tmp *= tau;
				for (j = n - 1; j >= k; j--)
				{
					a[(j + i*n)*mat_num + mat_pad] -= house[/*matrix_id +*/ ((j - 1) *MATRIX_NUM_PER_BLOCK)] * tmp;
				}
				i += HOUSE_KERNEL_THREAD_JOIN_NUM_VAR;
			}
		}


		__syncthreads();

	}

}

#endif