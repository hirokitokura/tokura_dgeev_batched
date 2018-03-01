#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"tokura_blas_define.h"
#include"tokura_blas_const.h"

#include"tokura_blas.h"
#include"tokura_blas_functions.h"


//DGEHRD reduces a DOUBLE PRECISION general matrix A 
//to upper Hessenberg form H by an orthogonal similarity transformation
__global__ void tokura_dgehrd_batched_SWB
(
	const int n_o,
	const  int mat_num,
	double*  a_o,
	const int hessen_join_num,
	const int THREADS_PER_MATRIX,
	const int MATRIX_PER_BLOCK
)
{

	const int thin = (threadIdx.x%THREADS_PER_MATRIX);
	const int mat_pad_index = (threadIdx.x / THREADS_PER_MATRIX);
	const int mat_pad = (threadIdx.x / THREADS_PER_MATRIX)*n_o;
	const int mat_pad_g = (blockIdx.x *  MATRIX_PER_BLOCK) *(n_o);
	const int n = (n_o*MATRIX_PER_BLOCK);
	const int REDUCTION_JOIN_THREADS_NUM = hessen_join_num;

	int i, j, k;
	double tmp;
	double tmp_sum;
	double tau;
	double vex_x_norm;


	if (!(((blockIdx.x *MATRIX_PER_BLOCK) + mat_pad_index) < mat_num))
	{
		return;
	}


	volatile extern __shared__ double SHARED_EXTERN[];
	volatile double *a = &SHARED_EXTERN[0];
	volatile double *house_tmp = &SHARED_EXTERN[n_o*n];
	volatile double *house = &house_tmp[mat_pad];
	volatile double *sum_tmp_shared = &SHARED_EXTERN[n_o*n + n_o*MATRIX_PER_BLOCK];


	const int ALIVE_THREADS_NUM = (mat_num - (blockIdx.x *MATRIX_PER_BLOCK)) > MATRIX_PER_BLOCK ? MATRIX_PER_BLOCK*THREADS_PER_MATRIX : (mat_num - (blockIdx.x *MATRIX_PER_BLOCK))*THREADS_PER_MATRIX;
	const int ALIVE_MATRIX_NUM = (mat_num - (blockIdx.x *MATRIX_PER_BLOCK)) > MATRIX_PER_BLOCK ? MATRIX_PER_BLOCK : (mat_num - (blockIdx.x *MATRIX_PER_BLOCK));


	{
		int count;
		int mat_pad_index;
		for (j = 0; j < n_o; j++)
		{
			for (count = threadIdx.x; count < n_o*ALIVE_MATRIX_NUM; count += ALIVE_THREADS_NUM)
			{
				i = count%n_o;
				mat_pad_index = count / n_o;
				a[j*n + i + mat_pad_index*n_o] = a_o[j*n_o*mat_num + i + mat_pad_index*n_o + mat_pad_g];
			}
		}
	}
	__syncwarp();


	for (k = 1; k < n_o - 1; k++)
	{
		tmp_sum = 0.0;

		i = k + 1 + thin;

		while (i < n_o)
		{
			tmp = a[(k - 1)*n + i + mat_pad];
			house[i] = tmp;
			tmp_sum += tmp*tmp;
			i += THREADS_PER_MATRIX;
		}
		__syncwarp();



		sum_tmp_shared[thin + mat_pad_index*THREADS_PER_MATRIX] = tmp_sum;
		__syncwarp();

		if (thin == 0)
		{
			house[k] = a[((k - 1)*n + (k)) + mat_pad];
		}
		__syncwarp();


		tmp_sum = 0.0;
		i = REDUCTION_JOIN_THREADS_NUM;
		while (i != 0)
		{
			if ((thin < i) && (thin + i < (THREADS_PER_MATRIX)))
			{
				sum_tmp_shared[(thin)+mat_pad_index*THREADS_PER_MATRIX] += sum_tmp_shared[(thin + i) + mat_pad_index*THREADS_PER_MATRIX];
			}
			__syncwarp();

			i = i >> 1;
		}
		__syncwarp();


		tmp_sum = sum_tmp_shared[0 + mat_pad_index*THREADS_PER_MATRIX];
		__syncwarp();

		int FLAG = (tmp_sum == 0.0);
		if (!FLAG)
		{
			if (thin == 0)
			{
				tmp = house[k] * house[k] + tmp_sum;
				tmp = sqrt(tmp);
				vex_x_norm = tmp;
				house[k] += (house[k] > 0.0) ? tmp : -tmp;
			}
		}
		__syncwarp();




		if (!FLAG)
		{
			tmp = house[k] * house[k] + tmp_sum;
		}
		__syncwarp();

		if (!FLAG)
		{
			tmp_sum = tmp;
			if (tmp_sum != 0.0)
			{
				tau = 2.0 / tmp;
			}
		}
		__syncwarp();



		if (!FLAG)
		{
			if (thin == 0)
			{
				a[((k - 1)*n + k) + mat_pad] = (house[k] > 0.0) ? -vex_x_norm : vex_x_norm;
				for (i = k + 1; i < n_o; i++)
				{
					a[((k - 1)*n + i) + mat_pad] = 0.0;
				}
			}
		}
		__syncwarp();

		if (!FLAG)
		{
			//matrix multication from left at similarity transformation

			j = k - 1 + 1 + thin;
			while (j < n_o)
			{
				tmp = 0.0;
				for (i = k; i < n_o; i++)
				{
					tmp += house[i] * a[(j*n + i) + mat_pad];
				}
				tmp *= tau;
				for (i = k; i < n_o; i++)
				{
					a[(j*n + i) + mat_pad] -= house[i] * tmp;
				}
				j += THREADS_PER_MATRIX;


			}
		}
		__syncwarp();

		if (!FLAG)
		{
			//matrix multication from right at similarity transformation

			i = 0 + thin;
			while (i < n_o)
			{
				tmp = 0.0;

				for (j = k; j < n_o; j++)
				{
					tmp += house[j] * a[(j*n + i) + mat_pad];
				}
				tmp *= tau;
				for (j = k; j < n_o; j++)
				{
					a[(j*n + i) + mat_pad] -= house[j] * tmp;
				}
				i += THREADS_PER_MATRIX;
			}
		}
	}
	__syncwarp();



	//__syncwarp();
	{
		int count;
		int mat_pad_index;
		for (j = 0; j < n_o; j++)
		{
			for (count = threadIdx.x; count < n_o*ALIVE_MATRIX_NUM; count += ALIVE_THREADS_NUM)
			{
				i = count%n_o;
				mat_pad_index = count / n_o;


				a_o[j*n_o*mat_num + i + mat_pad_index*n_o + mat_pad_g] = a[j*n + i + mat_pad_index*n_o];
			}
		}
	}

}