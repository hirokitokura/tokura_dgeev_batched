#include"mex.h"
#include"matrix.h"

#include"tokura_blas.h"


void mexFunction(int nlhs,mxArray *plhs[],int nrhs,mxArray *prhs[])
{

    if(nrhs!=1)
        /*
		output arguments 
         *double mat;matrices
         */
    {
  mexErrMsgTxt("the number of input arguments should be 1.\n");
    }
    
    if(nlhs!=3)/*出力引数は3*/
        /*
		output arguments 
			*real part of eigenvalues array
			*imaginary part of eigenvalues array
         */
    {
        mexErrMsgTxt("the number of output arguments should be 3.\n");
    }
    

	//int matrix_size=(int)mxGetM(prhs[0]);
	mwSize *dim_mat=(mwSize*)mxGetDimensions(prhs[0]);
	int dim_num=(int)mxGetNumberOfDimensions(prhs[0]);
    	
	if(dim_num<3)
	{
		mexErrMsgTxt("the number of matirces should be more than 1.\n eig() is faster than this function if the number of matrices is 1.\n");
	}
	
	mwSize* dim=(mwSize*)malloc(sizeof(mwSize)*(dim_num-1));
	if(dim==NULL)
	{
		mexErrMsgTxt("内部処理に必要な配列を確保できませんでした\n");
	}

	dim[0]=dim_mat[0];//
	
	int matrix_size=(int)dim[0];
	int matrix_size_tmp=(int)dim_mat[1];
	
	if(matrix_size!=matrix_size_tmp)
	{
		mexErrMsgTxt("input matrices are not square.\n");
	}
	if((matrix_size<1)||(matrix_size>32))
	{
		mexErrMsgTxt("the size of input matrix should be 1 ~ 32.\n");
	}
	int batchCount=1;
	{
		int i;
		for(i=1;i<(dim_num-1);i++)
		{
			dim[i]=dim_mat[i+1];
			batchCount*=(int)dim[i];
		}

	}
    
    double *mat=(double*)mxGetData(prhs[0]);
    if(mat==NULL)
    {
         mexErrMsgTxt("行列の格納に必要なメモリを確保できませんでした\n");
    }
  


/*    dim[1]=dim_mat[2];//
	dim[2]=dim_mat[3];
	dim[3]=dim_mat[4];*/


    plhs[0]=mxCreateNumericArray(dim_num-1,dim,mxDOUBLE_CLASS,mxREAL);
    double *comp_real=(double*)mxGetPr(plhs[0]);
    if(comp_real==NULL)
    {
         mexErrMsgTxt("malloc failed for real part of eigenvalues\n");
    }
    plhs[1]=mxCreateNumericArray(dim_num-1,dim,mxDOUBLE_CLASS,mxREAL);
    double *comp_imag=(double*)mxGetPr(plhs[1]);
    if(comp_imag==NULL)
    {
             mexErrMsgTxt("malloc failed for imaginary part of eigenvalues\n");
    }
    dim[0]=batchCount;
    dim[0]=1;
	plhs[2]=mxCreateNumericArray(dim_num-1,dim,mxINT8_CLASS,mxREAL);
    char *flags=(char*)mxGetPr(plhs[2]);
    if(flags==NULL)
    {
            mexErrMsgTxt("malloc failed for flags\n");
    }
	

	tokura_cudaHostRegister(&mat,matrix_size*matrix_size*batchCount);
	tokura_cudaHostRegister(&comp_real,matrix_size*matrix_size*batchCount);
	tokura_cudaHostRegister(&comp_imag,matrix_size*matrix_size*batchCount);
    int error_return;
	
	double** A = (double**)malloc(sizeof(double*)*batchCount);
	double** wr = (double**)malloc(sizeof(double*)*batchCount);
	double** wi = (double**)malloc(sizeof(double*)*batchCount);
	
	for(int i=0;i<batchCount;i++)
	{
		A[i]=&mat[i*matrix_size*matrix_size];
		wr[i]=&comp_real[i*matrix_size];
		wi[i]=&comp_imag[i*matrix_size];
	}
	tokura_dgeev_batched(matrix_size, A, wr, wi, batchCount, flags);
      
	tokura_cudaHostUnregister(mat);
	tokura_cudaHostUnregister(comp_real);
	tokura_cudaHostUnregister(comp_imag);
	free(A);
	free(wr);
	free(wi);
	free(dim);
  
}