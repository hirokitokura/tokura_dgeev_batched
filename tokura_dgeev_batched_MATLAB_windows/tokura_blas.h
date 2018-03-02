


//DGEEV computes all eigenvalues for an N-by-N real Hessenberg matrix A,
//Any eigenvectors are NOT computed.
int tokura_dgeev_batched(int n, double** A, double** wr, double** wi, int batchCount,char* flags);


void tokura_malloc(double **A, int size, int  batchCount);
void tokura_flags_malloc(char** flags, int batchCount);
void tokura_free(double** A);
void tokura_flags_free(char* flags);
void tokura_cudaHostRegister(double** mat, int size);
void tokura_cudaHostUnregister(double* mat);


