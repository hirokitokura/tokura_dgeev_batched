#ifndef __TOKURABLAS_DGEEV_BATCHED_MWB__
#define __TOKURABLAS_DGEEV_BATCHED_MWB__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"tokura_dgeev_batched_define.h"
#include"tokura_dgeev_batched_const.h"


#include"tokura_dgeev_batched_functions.h"
#include"tokura_dgeev_batched_tuned_thread_parameters.h"



#include"tokura_dgehrd_batched_MWB.cuh"
#include"tokura_dhseqr_batched_MWB.cuh"
#include"tokura_dgeev_batched_matrix_rearrangements.cuh"
#include"tokura_dgeev_batched_get_tuned_thread_parameters.cuh"
//Hessenberg reduction: MWB
//Double shift QR sweep: MWB
int tokura_dgeev_batched_MWB
(
	tokuraInternalhandle_t* tokurahandle, 
	int n, 
	double** A,
	double** wr, 
	double** wi,
	int batchCount,
	char* flags
)
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
	

	matrix_index = 0;

	i = 0;
	while (matrix_index < batchCount)
	{
		for (i = 0; i < NUMBER_OF_STREAMS; i++)
		{

			stream_id = i%NUMBER_OF_STREAMS;

			batchCount_per_stream = (matrix_index + NUMBER_OF_COMPUTED_MATRICES_PER_STREAM) > batchCount ? batchCount - matrix_index : NUMBER_OF_COMPUTED_MATRICES_PER_STREAM;
			
			if (batchCount_per_stream > 0)
			{
				/*for (int transferedmatrixid = 0; transferedmatrixid < batchCount_per_stream; transferedmatrixid++)
				{
					cudaMemcpyAsync
					(
						&tokurahandle->A_tmp_device[stream_id][transferedmatrixid*n*n],
						A[matrix_index+ transferedmatrixid],
						sizeof(double)*n*n,
						cudaMemcpyHostToDevice,
						tokurahandle->stream[stream_id]
					);

				}*/
				cudaMemcpyAsync
				(
					tokurahandle->A_tmp_device[stream_id],
					A[matrix_index],
					sizeof(double)*n*n*batchCount_per_stream,
					cudaMemcpyHostToDevice,
					tokurahandle->stream[stream_id]
				);
				tokura_matrixrearrangement_MWtoEW 
					<< < 
					(batchCount_per_stream + WARPSIZE - 1) / WARPSIZE,
					TRANSPOSE_CUDA_THREADS_MULTI,
					0,
					tokurahandle->stream[stream_id] >> >
					(
						n,
						tokurahandle->A_tmp_device[stream_id],
						tokurahandle->A_device[stream_id],
						batchCount_per_stream
						);

				dim3 thread_hessen(WARPSIZE, hessen_join_num);
				dim3 block_hessen((batchCount_per_stream + (WARPSIZE)-1) / (WARPSIZE));
				tokura_dgehrd_batched_MWB << <block_hessen, thread_hessen, sizeof(double)*(n - 1) * WARPSIZE + sizeof(double)*(hessen_join_num)* WARPSIZE/*0*/, tokurahandle->stream[stream_id] >> >
					(
						n,
						batchCount_per_stream,
						tokurahandle->A_device[stream_id],
						hessen_join_num
						);


				dim3 thread_qr(WARPSIZE, doubleqr_join_num);
				dim3 block_qr((batchCount_per_stream + (WARPSIZE)-1) / (WARPSIZE));
				tokura_dhseqr_batched_MWB << <block_qr, thread_qr, sizeof(int)*(n + 1) * (WARPSIZE)+sizeof(int)* (WARPSIZE), tokurahandle->stream[stream_id] >> >
					(
						n,
						batchCount_per_stream,
						tokurahandle->A_device[stream_id],
						doubleqr_join_num,
						tokurahandle->wr_device[stream_id],
						tokurahandle->wi_device[stream_id],
						tokurahandle->flags[stream_id]

						);

				dim3 thread_trans_eig(WARPSIZE, 8);
				dim3 block_trans_eig((batchCount_per_stream + (WARPSIZE)-1) / (WARPSIZE));
				tokura_eigenvaluesrearrangement_EWtoMW << < block_trans_eig, thread_trans_eig, sizeof(double)*n*(WARPSIZE + 1) * 2, tokurahandle->stream[stream_id] >> >
					(
						n,
						batchCount_per_stream,
						tokurahandle->wr_device[stream_id],
						tokurahandle->wi_device[stream_id],
						tokurahandle->wr_tmp_device[stream_id],
						tokurahandle->wi_tmp_device[stream_id]
						);
				


				/*for (int transferedmatrixid = 0; transferedmatrixid < batchCount_per_stream; transferedmatrixid++)
				{
					cudaMemcpyAsync(wr[matrix_index+ transferedmatrixid], tokurahandle->wr_tmp_device[stream_id], sizeof(double)*n, cudaMemcpyDeviceToHost, tokurahandle->stream[stream_id]);
					cudaMemcpyAsync(wi[matrix_index+ transferedmatrixid], tokurahandle->wi_tmp_device[stream_id], sizeof(double)*n, cudaMemcpyDeviceToHost, tokurahandle->stream[stream_id]);
					cudaMemcpyAsync(&flags[matrix_index+ transferedmatrixid], tokurahandle->flags[stream_id], sizeof(char), cudaMemcpyDeviceToHost, tokurahandle->stream[stream_id]);
				}*/

				cudaMemcpyAsync(wr[matrix_index],tokurahandle->wr_tmp_device[stream_id], sizeof(double)*n*batchCount_per_stream, cudaMemcpyDeviceToHost, tokurahandle->stream[stream_id]);
				cudaMemcpyAsync(wi[matrix_index], tokurahandle->wi_tmp_device[stream_id], sizeof(double)*n*batchCount_per_stream, cudaMemcpyDeviceToHost, tokurahandle->stream[stream_id]);
				cudaMemcpyAsync(&flags[matrix_index ], tokurahandle->flags[stream_id], sizeof(char)*batchCount_per_stream, cudaMemcpyDeviceToHost, tokurahandle->stream[stream_id]);
			}

			matrix_index += NUMBER_OF_COMPUTED_MATRICES_PER_STREAM;
		}
	}
	
	int matrix_size = n;
	matrix_index = 0;
	


	return 0;
}

#endif