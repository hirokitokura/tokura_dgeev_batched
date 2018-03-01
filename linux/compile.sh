cd Autotune
nvcc -o Autotune *.cu -O2 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_32,code=sm_32 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50  -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70
./Autotune
cd ../

cd tokura_dgeev_batched
nvcc -Xcompiler "-fPIC" -Xcompiler "-O2" -o ../libtokura_dgeev_batched.so --shared tokura_dgeev_batched.cu -gencode arch=compute_30,code=sm_30 -gencode arch=compute_32,code=sm_32 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50  -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70
cd ../

nvcc -o test test.cu -L./ -ltokura_dgeev_batched
./test
