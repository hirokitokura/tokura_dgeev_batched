 nvcc -Xcompiler "-fPIC" -Xcompiler "-O2" -o ../libtokuradgeevbatched.so --shared tokura_dgeev_batched.cu -gencode arch=compute_70,code=compute_70
