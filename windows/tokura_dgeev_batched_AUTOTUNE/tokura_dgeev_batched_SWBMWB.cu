
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"tokura_blas_define.h"
#include"tokura_blas_const.h"

#include"tokura_blas.h"
#include"tokura_blas_functions.h"
#include<float.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


int tokura_dgeev_batched_SWB_tune(tokuraInternalhandle_t* tokurahandle, int n, double** A, double** wr, double** wi, int batchCount, float* time)
{
	int i;
	int batchCount_per_stream;
	const int WARPSIZE = tokurahandle->WARP_SIZE;

	int shift;

	double *comp_real_h, *comp_imag_h;


	int tmp_thred_per_matrix;

	int hessen_join_num;
	int matrix_num_per_block;
	hessen_join_num = get_hessenbergreduction_SWBthreads_num(n);


	int doubleqr_join_num = get_doubleshiftQR_MWBthreads_num(n);


	int optimal_hrd_thread = 0;

	int matrix_index = 0;
	int stream_id = 0;



	i = 0;
	int thread_count;
	cudaEvent_t start, stop;
	float elapsed_time_ms = 0.0f;
	float hrd_local_time = FLT_MAX;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	for (thread_count = 1; thread_count <= n; thread_count++)
	{
		tmp_thred_per_matrix = thread_count;
		matrix_num_per_block = WARPSIZE / tmp_thred_per_matrix;

		if (
			tokurahandle->sharedsize
			<
			(sizeof(double)*n*n*matrix_num_per_block + sizeof(double)*n*matrix_num_per_block + sizeof(double)*tmp_thred_per_matrix*matrix_num_per_block)
			)
		{
			continue;
		}
		hessen_join_num = tmp_thred_per_matrix;
		for (i = 10; i > 0; i--)
		{
			if ((hessen_join_num >> i) == 1)
			{
				break;
			}
		}

		if ((1 << (i)) == hessen_join_num)
		{
			i = i - 1;
		}
		if (i < 0)
		{
			i = 0;
		}
		if (hessen_join_num == 1)
		{
			i = 0;
		}
		hessen_join_num = 1 << i;
		stream_id = 0;
		batchCount_per_stream = batchCount;

		for (int transferedmatrixid = 0; transferedmatrixid < batchCount_per_stream; transferedmatrixid++)
		{
			cudaMemcpyAsync(
				&tokurahandle->A_tmp_device[stream_id][(matrix_index + transferedmatrixid)*n*n],
				A[matrix_index + transferedmatrixid],
				sizeof(double)*n*n,
				cudaMemcpyHostToDevice,
				0
			);
		}
		cudaThreadSynchronize();
		cudaEventRecord(start, 0);
		dim3 grid, block;
		int temp = WARPSIZE / n;
		block = dim3(WARPSIZE, 4);
		tokura_matrixrearrangement_MWtoRW
			<< <batchCount_per_stream, block, 0 >> >
			(
				n,
				tokurahandle->A_tmp_device[stream_id],
				tokurahandle->A_device[stream_id],
				batchCount_per_stream
				);
		cudaThreadSynchronize();

		grid = dim3((batchCount_per_stream + matrix_num_per_block - 1) / matrix_num_per_block);
		block = dim3(tmp_thred_per_matrix*matrix_num_per_block);
		tokura_dgehrd_batched_SWB
			<< <
			grid,
			block,
			sizeof(double)*n*n*matrix_num_per_block 
			+ sizeof(double)*n*matrix_num_per_block 
			+ sizeof(double)*tmp_thred_per_matrix*matrix_num_per_block
			>> >
			(
				n,
				batchCount_per_stream,
				tokurahandle->A_device[stream_id],
				hessen_join_num,
				tmp_thred_per_matrix,
				matrix_num_per_block
				);
		cudaThreadSynchronize();

		dim3 block_RWtoEW = dim3((batchCount_per_stream + WARPSIZE - 1) / WARPSIZE, n);
		dim3 thread_RWtoEW = dim3(WARPSIZE, 1);
		tokura_matrixrearrangement_RWtoEW
			<< <
			block_RWtoEW,
			thread_RWtoEW,
			0
			>> >
			(
				n,
				batchCount_per_stream,
				tokurahandle->A_device[stream_id],
				tokurahandle->A_tmp_device[stream_id]
				);


		cudaThreadSynchronize();

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&elapsed_time_ms, start, stop);
		if (hrd_local_time > elapsed_time_ms)
		{
			hrd_local_time = elapsed_time_ms;
			optimal_hrd_thread = thread_count;
		}



	}
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	time[n] = hrd_local_time;
	return optimal_hrd_thread;

}