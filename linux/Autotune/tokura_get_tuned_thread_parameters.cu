
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"tokura_blas_define.h"
#include"tokura_blas_const.h"
//#include"tokura_tuned_thread_parameters.h"
int get_hessenbergreduction_MWBthreads_num(int n)
{
	int parameter;
	switch (n)
	{
/*	case 1:
		parameter = TOKURA_MWB_HRD_1;
		break;
	case 2:
		parameter = TOKURA_MWB_HRD_2;
		break;
	case 3:
		parameter = TOKURA_MWB_HRD_3;
		break;
	case 4:
		parameter = TOKURA_MWB_HRD_4;
		break;
	case 5:
		parameter = TOKURA_MWB_HRD_5;
		break;
	case 6:
		parameter = TOKURA_MWB_HRD_6;
		break;
	case 7:
		parameter = TOKURA_MWB_HRD_7;
		break;
	case 8:
		parameter = TOKURA_MWB_HRD_8;
		break;
	case 9:
		parameter = TOKURA_MWB_HRD_9;
		break;
	case 10:
		parameter = TOKURA_MWB_HRD_10;
		break;
	case 11:
		parameter = TOKURA_MWB_HRD_11;
		break;
	case 12:
		parameter = TOKURA_MWB_HRD_12;
		break;
	case 13:
		parameter = TOKURA_MWB_HRD_13;
		break;
	case 14:
		parameter = TOKURA_MWB_HRD_14;
		break;
	case 15:
		parameter = TOKURA_MWB_HRD_15;
		break;
	case 16:
		parameter = TOKURA_MWB_HRD_16;
		break;
	case 17:
		parameter = TOKURA_MWB_HRD_17;
		break;
	case 18:
		parameter = TOKURA_MWB_HRD_18;
		break;
	case 19:
		parameter = TOKURA_MWB_HRD_19;
		break;
	case 20:
		parameter = TOKURA_MWB_HRD_20;
		break;
	case 21:
		parameter = TOKURA_MWB_HRD_21;
		break;
	case 22:
		parameter = TOKURA_MWB_HRD_22;
		break;
	case 23:
		parameter = TOKURA_MWB_HRD_23;
		break;
	case 24:
		parameter = TOKURA_MWB_HRD_24;
		break;
	case 25:
		parameter = TOKURA_MWB_HRD_25;
		break;
	case 26:
		parameter = TOKURA_MWB_HRD_26;
		break;
	case 27:
		parameter = TOKURA_MWB_HRD_27;
		break;
	case 28:
		parameter = TOKURA_MWB_HRD_28;
		break;
	case 29:
		parameter = TOKURA_MWB_HRD_29;
		break;
	case 30:
		parameter = TOKURA_MWB_HRD_30;
		break;
	case 31:
		parameter = TOKURA_MWB_HRD_31;
		break;
	case 32:
		parameter = TOKURA_MWB_HRD_32;
		break;*/
	default:
		parameter = n;
		break;
	}
	return parameter;
}

int get_hessenbergreduction_SWBthreads_num(int n)
{
	int parameter;
	switch (n)
	{
/*	case 1:
		parameter = TOKURA_SWB_HRD_1;
		break;
	case 2:
		parameter = TOKURA_SWB_HRD_2;
		break;
	case 3:
		parameter = TOKURA_SWB_HRD_3;
		break;
	case 4:
		parameter = TOKURA_SWB_HRD_4;
		break;
	case 5:
		parameter = TOKURA_SWB_HRD_5;
		break;
	case 6:
		parameter = TOKURA_SWB_HRD_6;
		break;
	case 7:
		parameter = TOKURA_SWB_HRD_7;
		break;
	case 8:
		parameter = TOKURA_SWB_HRD_8;
		break;
	case 9:
		parameter = TOKURA_SWB_HRD_9;
		break;
	case 10:
		parameter = TOKURA_SWB_HRD_10;
		break;
	case 11:
		parameter = TOKURA_SWB_HRD_11;
		break;
	case 12:
		parameter = TOKURA_SWB_HRD_12;
		break;
	case 13:
		parameter = TOKURA_SWB_HRD_13;
		break;
	case 14:
		parameter = TOKURA_SWB_HRD_14;
		break;
	case 15:
		parameter = TOKURA_SWB_HRD_15;
		break;
	case 16:
		parameter = TOKURA_SWB_HRD_16;
		break;
	case 17:
		parameter = TOKURA_SWB_HRD_17;
		break;
	case 18:
		parameter = TOKURA_SWB_HRD_18;
		break;
	case 19:
		parameter = TOKURA_SWB_HRD_19;
		break;
	case 20:
		parameter = TOKURA_SWB_HRD_20;
		break;
	case 21:
		parameter = TOKURA_SWB_HRD_21;
		break;
	case 22:
		parameter = TOKURA_SWB_HRD_22;
		break;
	case 23:
		parameter = TOKURA_SWB_HRD_23;
		break;
	case 24:
		parameter = TOKURA_SWB_HRD_24;
		break;
	case 25:
		parameter = TOKURA_SWB_HRD_25;
		break;
	case 26:
		parameter = TOKURA_SWB_HRD_26;
		break;
	case 27:
		parameter = TOKURA_SWB_HRD_27;
		break;
	case 28:
		parameter = TOKURA_SWB_HRD_28;
		break;
	case 29:
		parameter = TOKURA_SWB_HRD_29;
		break;
	case 30:
		parameter = TOKURA_SWB_HRD_30;
		break;
	case 31:
		parameter = TOKURA_SWB_HRD_31;
		break;
	case 32:
		parameter = TOKURA_SWB_HRD_32;
		break;*/
	default:
		parameter = n;
		break;
	}
	return parameter;


}
int get_doubleshiftQR_MWBthreads_num(int n)
{
	int parameter;
	switch (n)
	{
	/*case 1:
		parameter = TOKURA_MWB_DOUBLESHIFTQR_1;
		break;
	case 2:
		parameter = TOKURA_MWB_DOUBLESHIFTQR_2;
		break;
	case 3:
		parameter = TOKURA_MWB_DOUBLESHIFTQR_3;
		break;
	case 4:
		parameter = TOKURA_MWB_DOUBLESHIFTQR_4;
		break;
	case 5:
		parameter = TOKURA_MWB_DOUBLESHIFTQR_5;
		break;
	case 6:
		parameter = TOKURA_MWB_DOUBLESHIFTQR_6;
		break;
	case 7:
		parameter = TOKURA_MWB_DOUBLESHIFTQR_7;
		break;
	case 8:
		parameter = TOKURA_MWB_DOUBLESHIFTQR_8;
		break;
	case 9:
		parameter = TOKURA_MWB_DOUBLESHIFTQR_9;
		break;
	case 10:
		parameter = TOKURA_MWB_DOUBLESHIFTQR_10;
		break;
	case 11:
		parameter = TOKURA_MWB_DOUBLESHIFTQR_11;
		break;
	case 12:
		parameter = TOKURA_MWB_DOUBLESHIFTQR_12;
		break;
	case 13:
		parameter = TOKURA_MWB_DOUBLESHIFTQR_13;
		break;
	case 14:
		parameter = TOKURA_MWB_DOUBLESHIFTQR_14;
		break;
	case 15:
		parameter = TOKURA_MWB_DOUBLESHIFTQR_15;
		break;
	case 16:
		parameter = TOKURA_MWB_DOUBLESHIFTQR_16;
		break;
	case 17:
		parameter = TOKURA_MWB_DOUBLESHIFTQR_17;
		break;
	case 18:
		parameter = TOKURA_MWB_DOUBLESHIFTQR_18;
		break;
	case 19:
		parameter = TOKURA_MWB_DOUBLESHIFTQR_19;
		break;
	case 20:
		parameter = TOKURA_MWB_DOUBLESHIFTQR_20;
		break;
	case 21:
		parameter = TOKURA_MWB_DOUBLESHIFTQR_21;
		break;
	case 22:
		parameter = TOKURA_MWB_DOUBLESHIFTQR_22;
		break;
	case 23:
		parameter = TOKURA_MWB_DOUBLESHIFTQR_23;
		break;
	case 24:
		parameter = TOKURA_MWB_DOUBLESHIFTQR_24;
		break;
	case 25:
		parameter = TOKURA_MWB_DOUBLESHIFTQR_25;
		break;
	case 26:
		parameter = TOKURA_MWB_DOUBLESHIFTQR_26;
		break;
	case 27:
		parameter = TOKURA_MWB_DOUBLESHIFTQR_27;
		break;
	case 28:
		parameter = TOKURA_MWB_DOUBLESHIFTQR_28;
		break;
	case 29:
		parameter = TOKURA_MWB_DOUBLESHIFTQR_29;
		break;
	case 30:
		parameter = TOKURA_MWB_DOUBLESHIFTQR_30;
		break;
	case 31:
		parameter = TOKURA_MWB_DOUBLESHIFTQR_31;
		break;
	case 32:
		parameter = TOKURA_MWB_DOUBLESHIFTQR_32;
		break;*/
	default:
		parameter = n;
		break;
	}
	return parameter;
}
