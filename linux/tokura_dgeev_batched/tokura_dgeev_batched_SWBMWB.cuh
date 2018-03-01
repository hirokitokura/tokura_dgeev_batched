#ifndef __TOKURABLAS_DGEEV_BATCHED_SWB__
#define __TOKURABLAS_DGEEV_BATCHED_SWB__
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"tokura_dgeev_batched_define.h"
#include"tokura_dgeev_batched_const.h"


#include"tokura_dgeev_batched_functions.h"
#include"tokura_dgeev_batched_tuned_thread_parameters.h"

#include"tokura_dgehrd_batched_SWB.cuh"
#include"tokura_dhseqr_batched_MWB.cuh"
#include"tokura_dgeev_batched_matrix_rearrangements.cuh"
#include"tokura_dgeev_batched_get_tuned_thread_parameters.cuh"



//Hessenberg reduction: SWB
//Double shift QR sweep: MWB
int tokura_dgeev_batched_SWBMWB
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
	int i;
	int batchCount_per_stream;
	const int WARPSIZE = tokurahandle->WARP_SIZE;

	int shift;

	double *comp_real_h, *comp_imag_h;

	int tmp_thred_per_matrix =get_hessenbergreduction_SWBthreads_num(n);



	int hessen_join_num ;
	int matrix_num_per_block;
	hessen_join_num = tmp_thred_per_matrix;
	matrix_num_per_block = WARPSIZE / tmp_thred_per_matrix;
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

	int doubleqr_join_num = get_doubleshiftQR_MWBthreads_num(n);
	


	int matrix_index = 0;
	int stream_id=0;
	batchCount_per_stream = NUMBER_OF_COMPUTED_MATRICES_PER_STREAM;
	dim3 block_RWtoEW = dim3((batchCount_per_stream + WARPSIZE - 1) / WARPSIZE, n);
	dim3 thread_RWtoEW = dim3(WARPSIZE, 1);
	
	i = 0;
	while (matrix_index < batchCount)
	{

		for (i = 0; i < NUMBER_OF_STREAMS; i++)
		{
			stream_id = i%NUMBER_OF_STREAMS;
			batchCount_per_stream = (matrix_index + NUMBER_OF_COMPUTED_MATRICES_PER_STREAM) > batchCount ? batchCount - matrix_index : NUMBER_OF_COMPUTED_MATRICES_PER_STREAM;


		//	cudaMemsetAsync(tokurahandle->wr_tmp_device[stream_id], 0,sizeof(double)*n*NUMBER_OF_COMPUTED_MATRICES_PER_STREAM, tokurahandle->stream[stream_id]);
			if (batchCount_per_stream > 0)
			{
				/*for (int transferedmatrixid = 0; transferedmatrixid < batchCount_per_stream; transferedmatrixid++)
				{
					cudaMemcpyAsync(
						&tokurahandle->A_tmp_device[stream_id][( transferedmatrixid)*n*n],
						A[matrix_index + transferedmatrixid],
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
				
				dim3 grid, block;
				int temp = WARPSIZE / n;
				//grid = dim3((comp_mat_num + (temp)-1) / (temp));
				block = dim3(WARPSIZE, 4);
				tokura_matrixrearrangement_MWtoRW
					<< <batchCount_per_stream, block, 0, tokurahandle->stream[stream_id] >> >
					(
						n,
						tokurahandle->A_tmp_device[stream_id],
						tokurahandle->A_device[stream_id],
						batchCount_per_stream
						);
				
				
				grid = dim3((batchCount_per_stream + matrix_num_per_block - 1) / matrix_num_per_block);
				block = dim3(tmp_thred_per_matrix*matrix_num_per_block);

				tokura_dgehrd_batched_SWB
					<< <
					grid,
					block,
					sizeof(double)*n*n*matrix_num_per_block + sizeof(double)*n*matrix_num_per_block + sizeof(double)*tmp_thred_per_matrix*matrix_num_per_block,
					tokurahandle->stream[stream_id]
					>> >
					(
						n,
						batchCount_per_stream,
						tokurahandle->A_device[stream_id],
						hessen_join_num,
						tmp_thred_per_matrix,
						matrix_num_per_block
						);
				

				dim3 block_RWtoEW = dim3((batchCount_per_stream + WARPSIZE - 1) / WARPSIZE, n);
				dim3 thread_RWtoEW = dim3(WARPSIZE, 1);
				tokura_matrixrearrangement_RWtoEW
					<< <
					block_RWtoEW,
					thread_RWtoEW,
					0,
					tokurahandle->stream[stream_id]
					>> >
					(
						n,
						batchCount_per_stream,
						tokurahandle->A_device[stream_id],
						tokurahandle->A_tmp_device[stream_id]
						);

				




				dim3 thread_qr(WARPSIZE, doubleqr_join_num);
				dim3 block_qr((batchCount_per_stream + (WARPSIZE)-1) / (WARPSIZE));
				tokura_dhseqr_batched_MWB
					<< <
					block_qr,
					thread_qr,
					sizeof(int)*(n + 1) * (WARPSIZE)+sizeof(int)* (WARPSIZE),
					tokurahandle->stream[stream_id] >> >
					(
						n,
						batchCount_per_stream,
						tokurahandle->A_tmp_device[stream_id],
						doubleqr_join_num,
						tokurahandle->wr_device[i],
						tokurahandle->wi_device[i],
						tokurahandle->flags[stream_id]
						);

			
				dim3 thread_trans_eig(WARPSIZE, 8);
				dim3 block_trans_eig((batchCount_per_stream + (WARPSIZE)-1) / (WARPSIZE));
				tokura_eigenvaluesrearrangement_EWtoMW 
					<< < 
					block_trans_eig,
					thread_trans_eig, 
					sizeof(double)*n*(WARPSIZE + 1) * 2,
					tokurahandle->stream[stream_id] 
					>> >
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
				cudaMemcpyAsync(wr[matrix_index], tokurahandle->wr_tmp_device[stream_id], sizeof(double)*n*batchCount_per_stream, cudaMemcpyDeviceToHost, tokurahandle->stream[stream_id]);
				cudaMemcpyAsync(wi[matrix_index], tokurahandle->wi_tmp_device[stream_id], sizeof(double)*n*batchCount_per_stream, cudaMemcpyDeviceToHost, tokurahandle->stream[stream_id]);
				cudaMemcpyAsync(&flags[matrix_index], tokurahandle->flags[stream_id], sizeof(char)*batchCount_per_stream, cudaMemcpyDeviceToHost, tokurahandle->stream[stream_id]);

			}

			matrix_index += NUMBER_OF_COMPUTED_MATRICES_PER_STREAM;
		}


	}
	
	return 0;
}
#endif