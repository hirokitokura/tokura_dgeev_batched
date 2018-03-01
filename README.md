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
