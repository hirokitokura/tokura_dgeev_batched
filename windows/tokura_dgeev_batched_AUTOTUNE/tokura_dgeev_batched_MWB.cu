
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

#include<float.h>



int tokura_dgeev_batched_MWB_tune(tokuraInternalhandle_t* tokurahandle, int n, double** A, double** wr, double** wi, int batchCount, float* time)
{
	int i, j;
	int matrix_index;
	int batchCount_per_stream;
	int WARPSIZE = tokurahandle->WARP_SIZE;

	int hessen_join_num;
	int doubleqr_join_num;
	int stream_id;


	hessen_join_num = get_hessenbergreduction_MWBthreads_num(n);
	doubleqr_join_num = get_doubleshiftQR_MWBthreads_num(n);
	cudaEvent_t start, stop;
	float elapsed_time_ms = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	matrix_index = 0;
	float hrd_local_time = FLT_MAX;
	float qr_local_time=FLT_MAX;
	i = 0;
	int optimal_hrd_thread=0;
	int optimal_qr_thread=0;
	int thread_count;
	for(thread_count=1;thread_count<=n;thread_count++)
	{
	
		hessen_join_num = thread_count;
		doubleqr_join_num = thread_count;
			stream_id = 0;

			batchCount_per_stream= batchCount;

			
				for (int transferedmatrixid = 0; transferedmatrixid < batchCount_per_stream; transferedmatrixid++)
				{
					cudaMemcpyAsync
					(
						&tokurahandle->A_tmp_device[stream_id][(matrix_index + transferedmatrixid)*n*n],
						A[matrix_index + transferedmatrixid],
						sizeof(double)*n*n,
						cudaMemcpyHostToDevice,
						tokurahandle->stream[stream_id]

					);

				}
				cudaThreadSynchronize();

				cudaEventRecord(start, 0);

				tokura_matrixrearrangement_MWtoEW
					<< <
					(batchCount_per_stream + WARPSIZE - 1) / WARPSIZE,
					TRANSPOSE_CUDA_THREADS_MULTI,
					0
					>> >
					(
						n,
						tokurahandle->A_tmp_device[stream_id],
						tokurahandle->A_device[stream_id],
						batchCount_per_stream
						);

				dim3 thread_hessen(WARPSIZE, hessen_join_num);
				dim3 block_hessen((batchCount_per_stream + (WARPSIZE)-1) / (WARPSIZE));
				tokura_dgehrd_batched_MWB << <block_hessen, thread_hessen, sizeof(double)*(n - 1) * WARPSIZE + sizeof(double)*(hessen_join_num)* WARPSIZE >> >
					(
						n,
						batchCount_per_stream,
						tokurahandle->A_device[stream_id],
						hessen_join_num
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

				cudaEventRecord(start, 0);
				dim3 thread_qr(WARPSIZE, doubleqr_join_num);
				dim3 block_qr((batchCount_per_stream + (WARPSIZE)-1) / (WARPSIZE));
				tokura_dhseqr_batched_MWB << <block_qr, thread_qr, sizeof(int)*(n + 1) * (WARPSIZE)+sizeof(int)* (WARPSIZE)>> >
					(
						n,
						batchCount_per_stream,
						tokurahandle->A_device[stream_id],
						doubleqr_join_num,
						tokurahandle->wr_device[stream_id],
						tokurahandle->wi_device[stream_id]
						);
				cudaThreadSynchronize();
				
				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);

				cudaEventElapsedTime(&elapsed_time_ms, start, stop);
				if (qr_local_time > elapsed_time_ms)
				{
					qr_local_time = elapsed_time_ms;
					optimal_qr_thread = thread_count;
				}
	}
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	time[n] = hrd_local_time;
	//time[n+32] = qr_local_time;
	return (optimal_qr_thread << 16) + optimal_hrd_thread;
}

