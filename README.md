# An Efficient GPU Implementation of Bulk Computation of the Eigenvalue Problem for Many Small Real Non-symmetric Matrices
We provide the CUDA-program for bulk computation of the eigenvalue problem for many small real non-symmetric matrices in the GPU.
Our GPU implementation can efficiently compute eigenvalues of many small metrices.

Our program supports eigenvalue computation of REAL matrices of which size is equal or less than 32. 
The size of all metrices should be the same.

more details -> http://www.ijnc.org/index.php/ijnc/article/view/152

# License
MIT

# Compile an example for Linux
* `mkdir $HOME/tokura_dgeev_batched`
* `git clone https://github.com/hirokitokura/tokura_dgeev_batched.git tokura_dgeev_batched`
* `cd tokura_dgeev_batched/linux`
* `./compile.sh`
  * Autotuning program is compiled and executed
  * Create libtokura_dgeev_batched.so
  * Test program is compiled and executed
  * Test program computes eigenvalues of 500000 randomly generated matrices.
# Compile an example for windows
* 'mkdir C:\tokura_dgeev_batched`
* `git clone https://github.com/hirokitokura/tokura_dgeev_batched.git C:\tokura_dgeev_batched`
* `cd C:\tokura_dgeev_batched\windows`
* Open `tokura_dgeev_batched.sln`
* Set `tokura_dgeev_batched_AUTOTUNE` as Startup Project
* Build and execute
  * `tokura_dgeev_batched_tuned_thread_parameters.h` is generated in `C:\tokura_dgeev_batched\windows\tokura_dgeev_batched`
* Set `tokura_dgeev_batched` as Startup Project
* Build
  * `tokura_dgeev_batched.dll` and `tokura_dgeev_batched.lib` are generated in `C:\tokura_dgeev_batched\windows\x64\Release`
 
Also, you can compile and execute a sample program
* Set `tokura_dgeev_batched_SAMPLE` as Startup Project
  *Before built, copy `tokura_dgeev_batched.lib` to `C:\tokura_dgeev_batched\windows\tokura_dgeev_batched_SAMPLE`
  * A sample program computes eigenvalues of 500000 randomly generated matrices.

# Functions
`int tokura_dgeev_batched
 (
  int n,
  double** A,
  double** wr, 
  double** wi,
  int batchCount,
  char* flags
  )`
  
This function computes the eigenvalues for N-by-N real non-symmetric matrices.
No eigenvectors are computed.
  
__Arguments__

* [in] `n` INTEGER, The order of the matrix A[i]. 0<=N<=32
* [in] `A` DOUBLE**, Array of pointers to double array with each matrix A[i] of which size is N×N.
 * `A[i]` should be allocated by tokura_malloc. 
* [out] `wr` DOUBLE**, Array of pointers to double array with each array wr[i] of which size is N.
  * `wr[i]` is stored real part of eigenvalues of matrix A[i].
  * `wr[i]` should be allocated by tokura_malloc. 
* [out] `wi` DOUBLE**, Array of pointers to double array with each array wi[i] of which size is N.
  * `wi[i]` is stored imaginary part of eigenvalues of matrix A[i].
  * `wi[i]` should be allocated by tokura_malloc. 
* [in] `batchCount` INTEGER, The number of pointers contained in A array
* [out] `flags` CHAR*, Array of flag.
  * `flag[i] == 0` -> This function successes to computes all the eigenvalues of A[i].
  * `flag[i] == 1` -> This function falis to computes all the eigenvalues of A[i].
  
__Error value__

`return value == 0 -> All arguments are not illegal value.`

`return value == -i -> i-th argument is illegal value.`

---
`void tokura_malloc(double **A, int numberofelements, int  batchCount)`

This function allocate `numberofelements*sizeof(double)` bytes for each A[i].
Use `tokura_free()` to free this memory.

__Arguments__
[out] `A` On putput, set to the pointer A[i] that was allocated.

[in] `numberofelements` The number of elements of matrix.

[in] `batchCount` The number of matrices.


---
`void tokura_free(double** A)`

This function free memory which is allocated by `tokura_malloc()'.

__Arguments__

[in] `A` Pointer to free


---
`void tokura_flags_malloc(char** flags, int batchCount)`

This function allocate 'sizeof(char)*batchCount' bytes for array flags.
Use `tokura_flags_free()` to free this memory.

__Arguments__

[out] `flags` On putput, set to the pointer flags that wa allocated.

[in] `batchCount` The number of matrices.


---
`void tokura_flags_free(char** flags, int batchCount)`

This function free memory which is allocated by `tokura_flags_malloc()`.

__Arguments__

[in] `flags` Pointer to free



# Limitation

* Computer Capability >= 3.0 (CUDA 9.x no longer supports previous hardware).
* The matrix size n is 1 <= n <= 32.
* Balancing is not performed.
* Input matrices must be stored in host memory.




