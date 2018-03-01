#pragma once


//DGEEV computes all eigenvalues for an N-by-N real Hessenberg matrix A,
//Any eigenvectors are NOT computed.
int tokura_dgeev_batched(int n, double** A, double** wr, double** wi, int batchCount);
int tokura_dgeev_batched_MWBtune(int n, double** A, double** wr, double** wi, int batchCount, float* time);
int tokura_dgeev_batched_SWBtune(int n, double** A, double** wr, double** wi, int batchCount,float* time);


