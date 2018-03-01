#pragma once



__global__ void tokura_dgehrd_batched_MWB(const int n, const  int mat_num, double*  a, const int hessen_join_num);
__global__ void tokura_dgehrd_batched_SWB(const int n_o, const  int mat_num, double*  a_o, const int hessen_join_num, const int THREADS_PER_MATRIX, const int MATRIX_PER_BLOCK);

//DHSEQR computes all eigenvalues for an N-by-N real Hessenberg matrix A,
//Any eigenvectors are NOT computed.
__global__ void tokura_dhseqr_batched_MWB(const int n, const int mat_num, double*  a_o, const int doubleqr_join_num, double *comp_real, double *comp_imag);
//__global__ void get_eig_gpu_kerne_MWB_SHARED(const int n, const int mat_num, double*  a_o, const int doubleqr_join_num, double *comp_real, double *comp_imag);

__global__ void tokura_matrixrearrangement_MWtoEW(const int n, const double* __restrict__ a, double *a2, const int batchCount_per_stream);

__global__ void tokura_eigenvaluesrearrangement_EWtoMW(const int n, const  int mat_num, const double* __restrict__ comp_real, const double* __restrict__ comp_imag, double *comp_real_out, double *comp_imag_out);


__global__ void tokura_matrixrearrangement_RWtoEW(const int n, const  int mat_num, const double* __restrict__ a, double*  a2);

__global__ void tokura_matrixrearrangement_MWtoRW
(
	int n,
	double *input,
	double *output,
	int mat_num
);

int get_hessenbergreduction_MWBthreads_num(int n);
int get_hessenbergreduction_SWBthreads_num(int n);
int get_doubleshiftQR_MWBthreads_num(int n);

