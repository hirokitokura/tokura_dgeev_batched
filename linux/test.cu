#include<stdio.h>
#include<stdlib.h>
#include<float.h>

#include"MT.h"

#include"tokura_blas.h"

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



int main(void)
{
	int i;
	int n;//n: matrix size
	int batchCount = 500000;//the number of matrices
	int info;
	//info ==0; successful exit
	//info < 0; if INFO = -i, the i-th argument had an illegal value.

	double **A;//matrix array
	double **wr;//real part of eigenvalues
	double **wi;//imaginary part of eigenvalues
	char* flags;
	//flags[i]==0: eigenvalues of i-th matrix are computed corectly
	//flags[i]!=0: eigenvalues of i-th matrix are NOT computed corectly

	for (n = 1; n <= 32; n += 1)
	{
		A = (double**)malloc(sizeof(double*)*batchCount);
		wr = (double**)malloc(sizeof(double*)*batchCount);
		wi = (double**)malloc(sizeof(double*)*batchCount);



		int elementnum_per_matrix;
		elementnum_per_matrix = n*n;
		tokura_malloc(A, elementnum_per_matrix, batchCount);
		elementnum_per_matrix = n;
		tokura_malloc(wr, elementnum_per_matrix, batchCount);
		tokura_malloc(wi, elementnum_per_matrix, batchCount);
		tokura_flags_malloc(&flags, batchCount);

		set_mat(A, n, batchCount);

		tokura_dgeev_batched(n, A, wr, wi, batchCount, flags);
		int matrix_index;

		matrix_index = 0;
		for (matrix_index = 0; matrix_index < batchCount; matrix_index++)
		{
			if (flags[matrix_index] == 1)
			{
				printf("%d Eigenvalues of %d-th matrix are NOT computed correctly\n", n, matrix_index);
				for (i = 0; i < n; i++)
				{
					printf("%e %e\n", wr[matrix_index][i], wi[matrix_index][i]);
				}
				exit(-1);

			}

		}
		printf("%d*%d matrix OK\n", n,n);

		tokura_free(A);
		tokura_free(wr);
		tokura_free(wi);
		tokura_flags_free(flags);


		free(A);
		free(wr);
		free(wi);
	}

	return 0;
}
