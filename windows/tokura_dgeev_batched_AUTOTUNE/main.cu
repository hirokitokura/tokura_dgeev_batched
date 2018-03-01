
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include"tokura_blas_define.h"
//#include"const.h"

#include<omp.h>

#include"MT.h"

#include"tokura_blas.h"
#define MAX_MATRIX_SIZE 32

#define USE_MATRIX_NUM_FOR_TUNE (32768>>2)

void set_mat(double **A, int n, int batchCount)
{
	int i = 0;
	int j, k;
	double tmp;
	FILE* fp;


	init_genrand(100);

	for (k = 0; k < batchCount; k++)
	{
		for (i = 0; i <n; i++)
		{
			for (j = 0; j <n; j++)
			{
				A[k][j*n + i] = genrand_res53();

			}
		}
	}

	return;
}
void parameter_write(float time[2][MAX_MATRIX_SIZE + 1], int threadnum[2][MAX_MATRIX_SIZE * 2])
{
	int switch_size = MAX_MATRIX_SIZE + 1;
	int n;
	for (n = 1; n <= MAX_MATRIX_SIZE; n++)
	{
		if (time[0][n] > time[1][n])
		{
			switch_size = n + 1;
			break;
		}
	}
	printf("SWITCH %d\n", switch_size);

	char define_name[1024];
	int i;
	FILE* fp;
	fp = fopen("../tokura_dgeev_batched/tokura_dgeev_batched_tuned_thread_parameters.h", "w");
	if (fp == NULL)
	{
		printf("FILE OPEN ERROR\n");
		exit(-1);
	}

	fprintf(fp, "#ifndef __TOKURABLAS_TUNED_PARAMETERS__\n");
	fprintf(fp, "#define __TOKURABLAS_TUNED_PARAMETERS__\n");
	fprintf(fp, "#define NUMBER_OF_COMPUTED_MATRICES_PER_STREAM %d\n", USE_MATRIX_NUM_FOR_TUNE);
	fprintf(fp, "#define TOKURA_SWITCH_ALGORITHM_MATRIXSIZE %d\n", switch_size);


	for (i = 1; i <= MAX_MATRIX_SIZE; i++)
	{
		sprintf(define_name, "#define TOKURA_MWB_HRD_%d %d\n", i, threadnum[0][i]);
		fprintf(fp, "%s", define_name);
	}
	for (i = 1; i <= MAX_MATRIX_SIZE; i++)
	{
		sprintf(define_name, "#define TOKURA_MWB_DOUBLESHIFTQR_%d %d\n", i, threadnum[0][i + MAX_MATRIX_SIZE]);
		fprintf(fp, "%s", define_name);
	}
	for (i = 1; i <= MAX_MATRIX_SIZE; i++)
	{
		sprintf(define_name, "#define TOKURA_SWB_HRD_%d %d\n", i, threadnum[1][i]);
		fprintf(fp, "%s", define_name);
	}
	fprintf(fp, "#endif\n");

	fclose(fp);

}


int main(void)
{
	int i;
	int n;//n: 行列サイズ
	int batchCount = USE_MATRIX_NUM_FOR_TUNE;
		;//batchCount: 行列数
	int info;

	float time[2][MAX_MATRIX_SIZE+1];

	int threadnum[2][MAX_MATRIX_SIZE*2];
	//get MWB executin time

	int start = 1;
	int end=32;
	for (n = start; n <= end; n += 1)
	{

		double **A;//入力配列
		double **wr;
		double **wi;
		A = (double**)malloc(sizeof(double*)*batchCount);
		wr = (double**)malloc(sizeof(double*)*batchCount);
		wi = (double**)malloc(sizeof(double*)*batchCount);
		
		
		for (i = 0; i < batchCount; i++)
		{
			A[i] = (double*)malloc(sizeof(double)*n*n);
			wr[i] = (double*)malloc(sizeof(double)*n);
			wi[i] = (double*)malloc(sizeof(double)*n);
		}
		set_mat(A, n, batchCount);
		for (i = 0; i < batchCount; i++)
		{
			int j;
			for (j = 0; j < n; j++)
			{
				wr[i][j] = 0.0;
				wi[i][j] = 0.0;
			}

		}


		int threads=tokura_dgeev_batched_MWBtune(n, A, wr, wi, batchCount,time[0]);
		threadnum[0][n] = threads & 0xffff;
		threadnum[0][n+ MAX_MATRIX_SIZE] = (threads >>16)& 0xffff;

		for (i = 0; i < batchCount; i++)
		{
			free(A[i]);
			free(wr[i]);
			free(wi[i]);
		}
		free(A);
		free(wr);
		free(wi);

		printf("MWB:%d end\n", n);
	}
	//get SWB executin time
	for (n = start; n <= end; n += 1)
	{

		double **A;//入力配列
		double **wr;
		double **wi;
		A = (double**)malloc(sizeof(double*)*batchCount);
		wr = (double**)malloc(sizeof(double*)*batchCount);
		wi = (double**)malloc(sizeof(double*)*batchCount);


		for (i = 0; i < batchCount; i++)
		{
			A[i] = (double*)malloc(sizeof(double)*n*n);
			wr[i] = (double*)malloc(sizeof(double)*n);
			wi[i] = (double*)malloc(sizeof(double)*n);
		}
		set_mat(A, n, batchCount);
		for (i = 0; i < batchCount; i++)
		{
			int j;
			for (j = 0; j < n; j++)
			{
				wr[i][j] = 0.0;
				wi[i][j] = 0.0;
			}

		}

		threadnum[1][n] = tokura_dgeev_batched_SWBtune(n, A, wr, wi, batchCount, time[1]);

		for (i = 0; i < batchCount; i++)
		{
			free(A[i]);
			free(wr[i]);
			free(wi[i]);
		}
		free(A);
		free(wr);
		free(wi);

		printf("SWB:%d end\n", n);

	}

	for (n = start; n <= end; n += 1)
	{
		printf("%d:%lf %lf[ms]\n", n, time[0][n], time[1][n]);

	}

	

	parameter_write(time, threadnum);
	return 0;
}
