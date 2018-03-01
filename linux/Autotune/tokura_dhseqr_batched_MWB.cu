
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"tokura_blas_define.h"
#include"tokura_blas_const.h"






///*副対角要素チェック,バルジ生成,バルジチェーシング並列*/
//DHSEQR computes eigenvalues of Hessenberg matrix
//Any eigenvectors are NOT computed.
__global__ void tokura_dhseqr_batched_MWB(const int n, const int mat_num, double*  a, const int DOUBLE_QR_KERNEL_THREAD_JOIN_NUM_VAR, double *comp_real, double *comp_imag)
{

	const int matrix_id = threadIdx.x + threadIdx.z*blockDim.x;
	const int warpid = threadIdx.y;
	const int mat_pad = (blockIdx.x*blockDim.x*blockDim.z) + matrix_id;
	const int MATRIX_NUM_PER_BLOCK = blockDim.x*blockDim.z;
	if (!((mat_pad < mat_num)))
	{
		return;
	}

	int nn, m, j, its, i, mmin;
	double /*z, y, x,*/ r, q, p;
	double tmp = 0.0;
	double tau = 0.0;
	int k;


	int ZERO_FLAG = 0;
	extern __shared__ int dynamic_shared[];

	int *zero_index = &dynamic_shared[0];/*副対角要素が0であるインデックス(列)を格納*/
	int *eig_num_shared = &zero_index[(n + 1)*MATRIX_NUM_PER_BLOCK];

	if (warpid == 0)
	{
		//zero_index[0][matrix_id] = 0;
		zero_index[matrix_id + (0 * MATRIX_NUM_PER_BLOCK)] = 0;
		//zero_index[n][matrix_id] = n;
		zero_index[matrix_id + (n *MATRIX_NUM_PER_BLOCK)] = n;
	}


	nn = n;
	its = 0;
	i = 1 + warpid;
	while (i < n /*+ 1*/)
	{
		zero_index[matrix_id + (i*MATRIX_NUM_PER_BLOCK)] = 0;
		i += DOUBLE_QR_KERNEL_THREAD_JOIN_NUM_VAR;
	}

	__syncthreads();
	//	__syncwarp();
	while (1/*its != 100000*/)
	{


		/*行列内ゴミの値除去開始*/
		i = 2 + warpid;
		while (i < n)
		{
			a[((i - 2) + i*n)*mat_num + mat_pad] = 0.0;
			i += DOUBLE_QR_KERNEL_THREAD_JOIN_NUM_VAR;
		}
		i = 3 + warpid;
		while (i < n)
		{
			a[((i - 3) + i*n)*mat_num + mat_pad] = 0.0;
			i += DOUBLE_QR_KERNEL_THREAD_JOIN_NUM_VAR;
		}
		__syncthreads();
		//	__syncwarp();

		m = (n - 1) - warpid;
		while (0 < m)
		{
			if (zero_index[matrix_id + (m*MATRIX_NUM_PER_BLOCK)] == 0)
			{
				tmp = fabs(a[(m + m*n)*mat_num + mat_pad]) + fabs(a[((m - 1) + (m - 1)*n)*mat_num + mat_pad]);
				if (tmp == 0)
				{
					tmp = 1.0;
				}

				if ((fabs(a[((m - 1) + m*n)*mat_num + mat_pad]) + tmp) == tmp)
				{
					zero_index[matrix_id + (m*MATRIX_NUM_PER_BLOCK)] = m;
				}
			}
			m -= DOUBLE_QR_KERNEL_THREAD_JOIN_NUM_VAR;

		}
		//__syncwarp();

		__syncthreads();
		if (warpid == 0)
		{
			/*変数割り当て変更
			eig_num->i*/
			//eig_num = 0;
			i = 0;
			/*変数割り当て変更
			EIG_END->j*/
			//EIG_END = n;
			j = n;
			for (m = n - 1; 0 < m; m--)
			{
				if (zero_index[matrix_id + (m *MATRIX_NUM_PER_BLOCK)] != 0)
				{
					a[((m - 1) + m*n)*mat_num + mat_pad] = 0.0;
					nn = j - m;
					j = m;
					if (nn < 3)
					{
						i += nn;
					}
				}
			}

			nn = j - 0;
			if (nn < 3)
			{
				i += nn;
			}
			eig_num_shared[matrix_id] = i;
		}
		__syncthreads();
		//__syncwarp();



		/*if (eig_num_shared[matrix_id] == n)
		{

		break;
		}*/
		unsigned int tmp_flag_ballot = eig_num_shared[matrix_id] == n;
		__syncthreads();
		//__syncwarp();
		unsigned int ballot_mask;
		ballot_mask = __ballot_sync(__activemask(), tmp_flag_ballot);

		if (ballot_mask == __activemask())
		{
			break;
		}
		its++;

		__syncthreads();
		//bulge generating 
		{
			m = 0;
			nn = n;

			k = 0;
			for (m = 0; m + 2 < n; m++)
			{
				nn = n;

				/*ZERO_FLAG==1のときのみバルジができる*/
				ZERO_FLAG = ((zero_index[matrix_id + (m *MATRIX_NUM_PER_BLOCK)] != 0) || (m == 0)) && ((zero_index[matrix_id + ((m + 1) *MATRIX_NUM_PER_BLOCK)] == 0) && (zero_index[matrix_id + ((m + 2) *MATRIX_NUM_PER_BLOCK)] == 0));
				//	__syncwarp();

				if (ZERO_FLAG == 1)
				{
					for (i = m + 3; i < n + 1; i++)
					{
						if (zero_index[matrix_id + (i*MATRIX_NUM_PER_BLOCK)] != 0)
						{
							nn = zero_index[matrix_id + (i *MATRIX_NUM_PER_BLOCK)];
							break;
						}
					}
				}
				//	__syncwarp();
				if (ZERO_FLAG == 1)
				{

					//}



					if (warpid < (nn - m/*k*/))
					{
						/*バルジ生成開始*/
						//	if (ZERO_FLAG == 1)
						//	{
						/*シフトの選択*/
						if (its % 10 == 0)
						{
							p = fabs(a[(((nn - 1 - 1) + (nn - 1)*n))*mat_num + mat_pad]);
							p = p + fabs(a[(((nn - 2 - 1) + (nn - 1 - 1)*n))*mat_num + mat_pad]);
							r = 1.5*(p);
							tmp = p*p;
						}
						else
						{
							p = a[((nn - 1) + (nn - 1)*n)*mat_num + mat_pad];
							q = a[((nn - 1 - 1) + (nn - 1 - 1)*n)*mat_num + mat_pad];
							r = p + q;
							tmp = p*q - a[((nn - 1 - 1) + (nn - 1)*n)*mat_num + mat_pad] * a[((nn - 1) + (nn - 1 - 1)*n)*mat_num + mat_pad]/*w*/;
						}

					}
				}


				//	__syncwarp();
				if (ZERO_FLAG == 1)
				{

					ZERO_FLAG = ZERO_FLAG && (a[((m)+(m + 1)*n)*mat_num + mat_pad] != 0.0);
				}

				//	__syncwarp();
				__syncthreads();

				/*ハウスホルダー変換構築開始*/
				if (ZERO_FLAG == 1)
				{
					if (warpid < (nn - m/*k*/))
					{
						p = (a[(((m)+(m)*n))*mat_num + mat_pad] * (a[(((m)+(m)*n))*mat_num + mat_pad] - r) + tmp) / a[((m)+(m + 1)*n)*mat_num + mat_pad] + a[((m + 1) + (m)*n)*mat_num + mat_pad];
						q = a[(((m)+(m)*n))*mat_num + mat_pad] + a[((m + 1) + (m + 1)*n)*mat_num + mat_pad] - r;
						r = a[((m + 1) + (m + 2)*n)*mat_num + mat_pad];


						/*精度確保のための処理開始*/
						tmp = fabs(p) + fabs(q) + fabs(r);
						p /= tmp;
						q /= tmp;
						r /= tmp;
						/*精度確保のための処理終了*/

						tau = q*q + r*r;
						tmp = p*p + tau;
						tmp = sqrt(fabs(tmp));

						p += (p > 0.0 ? tmp : -tmp);
						tmp = p*p + tau;
						tau = 2.0 / tmp;
						/*x = tmp*p;
						y = tmp*q;
						z = tmp*r;*/

						/*E-[x;y;z][p q r]*/
						/*ハウスホルダー変換構築終了*/
			}

		}
				__syncthreads();
				//	__syncwarp();

#if USE_DOUBLE==0		
				if (isnan(tau) || isnan(p) || isnan(q) || isnan(r))
				{
					a[((m)+(m + 1)*n)*mat_num + mat_pad] = 0.0;
					continue;
				}
				if (isinf(tau) || isinf(p) || isinf(q) || isinf(r))
				{
					a[((m)+(m + 1)*n)*mat_num + mat_pad] = 0.0;
					continue;
				}
#endif
				if (ZERO_FLAG == 1)
				{
					/*バルジ生成開始*/
					j = m + warpid;
					if (warpid < (nn - m/*k*/))
					{
						while (j < nn)
						{
							tmp = p*a[(j + (m + 0)*n)*mat_num + mat_pad] + q*a[(j + (m + 1)*n)*mat_num + mat_pad] + r*a[(j + (m + 2)*n)*mat_num + mat_pad];
							tmp *= tau;
							a[(j + (m + 2)*n)*mat_num + mat_pad] -= r*tmp;
							a[(j + (m + 1)*n)*mat_num + mat_pad] -= q*tmp;
							a[(j + (m + 0)*n)*mat_num + mat_pad] -= p*tmp;
							j += DOUBLE_QR_KERNEL_THREAD_JOIN_NUM_VAR;
						}
					}
					mmin = ((nn < (m + 3 + 1)) ? nn : m + 3 + 1);
				}

				{
					__syncthreads();
					//	__syncwarp();

				}

				if (ZERO_FLAG == 1)
				{
					//for (i = m; i < mmin; i++)
					i = m + warpid;
					if (warpid < (nn - m/*k*/))
					{
						while (i < mmin)
						{
							tmp = p*a[((m + 0) + i*n)*mat_num + mat_pad] + q*a[((m + 1) + i*n)*mat_num + mat_pad] + r*a[((m + 2) + i*n)*mat_num + mat_pad];
							tmp *= tau;
							a[((m + 2) + i*n)*mat_num + mat_pad] -= r*tmp;
							a[((m + 1) + i*n)*mat_num + mat_pad] -= q*tmp;
							a[((m + 0) + i*n)*mat_num + mat_pad] -= p*tmp;
							i += DOUBLE_QR_KERNEL_THREAD_JOIN_NUM_VAR;
						}
					}
					/*バルジ生成終了*/
					k = nn;
					//m = nn - 1;
				}
				{
					__syncthreads();
					//	__syncwarp();

				}


				//m = nn;/*次のバルジを作る*/

	}

}

		__syncthreads();
		//return;

		//bulge chasing
		nn = n;
		k = 0;
		for (m = 0; m + 1 + 2 < n + 1/*m < nn-1-1*/; m++)
		{


			if (zero_index[matrix_id + ((m + 1) *MATRIX_NUM_PER_BLOCK)] != 0)
			{
				k = m + 1;

			}
			else if (zero_index[matrix_id + ((m + 1 + 1)*MATRIX_NUM_PER_BLOCK)] != 0)
			{
				k = m + 1 + 1;
			}
			//__syncwarp();

			ZERO_FLAG = (zero_index[matrix_id + ((m + 1) *MATRIX_NUM_PER_BLOCK)] == 0) && (zero_index[matrix_id + ((m + 1 + 1) *MATRIX_NUM_PER_BLOCK)] == 0);

			if (ZERO_FLAG == 1)
			{
				/*バルジを生成できるか判断*/
				for (i = m + 1 + 1 + 1; i < n + 1; i++)
				{
					if (zero_index[matrix_id + (i*MATRIX_NUM_PER_BLOCK)] != 0)
					{
						nn = zero_index[matrix_id + (i *MATRIX_NUM_PER_BLOCK)];

						break;
					}
				}
				//}
			}
			//	__syncwarp();
			if (ZERO_FLAG == 1)
			{


				if (warpid < (nn - k))
				{
					//	if (ZERO_FLAG == 1)
					//	{
					p = a[(m + (m + 1 + 0)*n)*mat_num + mat_pad];
					q = a[(m + (m + 1 + 1)*n)*mat_num + mat_pad];
					if ((m + 1 + 2) < nn)
					{
						r = a[(m + (m + 1 + 2)*n)*mat_num + mat_pad];
					}
					else
					{
						r = 0.0;
					}

					tau = q*q + r*r;

					tmp = p*p + tau;
					tmp = sqrt(fabs(tmp));


					p += ((p > 0.0) ? tmp : -tmp);


					tmp = p*p + tau;


					if ((ZERO_FLAG == 1) && (tmp != 0.0))
					{
						tau = 2.0 / tmp;

					}
					ZERO_FLAG = ZERO_FLAG && (tmp != 0.0);
				}
					}
			__syncthreads();
			//__syncwarp();

			if ((ZERO_FLAG == 1))
			{

				j = m + warpid;
				if (warpid < (nn - k))
				{
					while (j < nn)
					{

						tmp = p*a[(j + (m + 1 + 0)*n)*mat_num + mat_pad] + q*a[(j + (m + 1 + 1)*n)*mat_num + mat_pad];
						tmp *= tau;
						if ((m + 1 + 2) < nn)
						{
							tmp += tau*r*a[(j + (m + 1 + 2)*n)*mat_num + mat_pad];
							a[(j + (m + 1 + 2)*n)*mat_num + mat_pad] -= r*tmp;
						}
						a[(j + (m + 1 + 1)*n)*mat_num + mat_pad] -= q*tmp;
						a[(j + (m + 1 + 0)*n)*mat_num + mat_pad] -= p*tmp;

						j += DOUBLE_QR_KERNEL_THREAD_JOIN_NUM_VAR;
					}
				}
			}
			__syncwarp();

			mmin = (nn < (m + 3 + 1 + 1) ? nn : (m + 3 + 1 + 1));

			__syncthreads();
		

			if ((ZERO_FLAG == 1))
			{

				i = k + warpid;
				if (warpid < (nn - k))
				{
					while (i < mmin)
					{

						tmp = p*a[((m + 1 + 0) + i*n)*mat_num + mat_pad] + q*a[((m + 1 + 1) + i*n)*mat_num + mat_pad];
						tmp *= tau;
						if ((m + 1 + 2) < nn)
						{
							tmp += tau*r*a[((m + 1 + 2) + i*n)*mat_num + mat_pad];
							a[((m + 1 + 2) + i*n)*mat_num + mat_pad] -= r*tmp;
						}
						a[((m + 1 + 1) + i*n)*mat_num + mat_pad] -= q*tmp;
						a[((m + 1 + 0) + i*n)*mat_num + mat_pad] -= p*tmp;

						i += DOUBLE_QR_KERNEL_THREAD_JOIN_NUM_VAR;
					}
				}
			}
			__syncthreads();

				}
		__syncthreads();

			}

	//Eigenvalues of 1*1 or 2*2 computation
	//Stride access may be pefrmed at eigenvalue storing to global memory
	//However, the overhead is much lower than above computation.
	//So, the overhead can be ignored.
	if (warpid == 0)
	{
		{
			double x, y, z;
			nn = n - 1;
			while (nn >= 0)
			{
				m = (zero_index[matrix_id + (nn *MATRIX_NUM_PER_BLOCK)] != 0) || (nn == 0) ? nn : nn - 1;
				if (m == nn)
				{
					comp_real[nn*mat_num + mat_pad] = a[(nn + nn*n)*mat_num + mat_pad];
					comp_imag[nn*mat_num + mat_pad] = 0.0;		
					nn--;
				}
				else
				{
					tau = a[((nn - 1) + (nn - 1)*n)*mat_num + mat_pad];
					p = 0.5*(a[(nn + nn*n)*mat_num + mat_pad] + tau/*y*/);
					q = 4.0*(a[((nn - 1) + nn*n)*mat_num + mat_pad] * a[(nn + (nn - 1)*n)*mat_num + mat_pad]) + (a[(nn + nn*n)*mat_num + mat_pad]/*x*/ - tau/*y*/)*(a[(nn + nn*n)*mat_num + mat_pad]/*x*/ - tau/*y*/);
					tmp = 0.5*sqrt(fabs(q));

					comp_real[nn*mat_num + mat_pad] = p + ((q >= 0.0) ? tmp/*z*/ : 0.0);
					comp_imag[nn*mat_num + mat_pad] = q < 0.0 ? tmp/*z*/ : 0.0;
					comp_real[(nn - 1)*mat_num + mat_pad] = p - ((q >= 0.0) ? tmp/*z*/ : 0.0);
					comp_imag[(nn - 1)*mat_num + mat_pad] = q < 0.0 ? -tmp/*-z*/ : 0.0;
					nn -= 2;
				}
			}
		}
	}


}
