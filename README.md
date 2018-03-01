# An Efficient GPU Implementation of Bulk Computation of the Eigenvalue Problem for Many Small Real Non-symmetric Matrices
We provide the CUDA-program for bulk computation of the eigenvalue problem for many small real non-symmetric matrices in the GPU.
Our GPU implementation can efficiently compute eigenvalues of many small metrices.

Our program supports eigenvalue computation of REAL matrices of which size is equal or less than 32. 
The size of all metrices should be the same.

more details -> http://www.ijnc.org/index.php/ijnc/article/view/152

# License
MIT

# Compile an example for Linux
* mkdir $HOME/tokura_dgeev_batched
* git clone https://github.com/hirokitokura/tokura_dgeev_batched.git tokura_dgeev_batched
* cd tokura_dgeev_batched/linux
* ./compile.sh
  * Autotuning program is compiled and executed
  * Create libtokura_dgeev_batched.so
  * Test program is compiled and executed
# Compile an example for windows
