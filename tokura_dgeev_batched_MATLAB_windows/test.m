%% Settings
mat_num=500000;
matrix_size=5;

error_value=zeros(mat_num*matrix_size,1);
%% Generate ramdon matrices
%mat=zeros(matrix_size,matrix_size,mat_num);
%for k=1:mat_num
%   mat(:,:,k)=rand(matrix_size,matrix_size);
%   mat(:,:,k)=sprandsym(matrix_size,density);
%end

%% Generate ramdon matrices which are used as test matrix in our work.
mat=get_matrix_MT(matrix_size,mat_num);

%% compute eigenvalues by our program
tic
[eig_real,eig_comp,flags]=tokura_dgeev_batched(mat);
toc


%% compute eigenvalues by MATLAB and relative errors of eigenvalues computed by our program over by MATLAB
error_ave=0.0;
count=0;
for k=1:mat_num
    eigtmp=eig_real(:,k)+eig_comp(:,k)*1i;
    tmp=eig(mat(:,:,k));
    tmp=sort(complex(tmp));
    tmp2=sort(complex(eigtmp));
    for m=1:matrix_size
        error_value(k*matrix_size+m,1)=(norm(complex(tmp(m))-complex(tmp2(m))))/norm(complex(tmp(m)));
        error_ave=error_ave+error_value(k*matrix_size+m,1);
    end 
end
mean(error_value)
[M,I]=max(error_value)
histogram(log10(error_value))