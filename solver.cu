#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>
#include "wtime.h"
#include "definitions.h"
#include "energy_struct.h"
#include "cuda_runtime.h"
#include "solver.h"

using namespace std;

/**
* Kernel del calculo de la solvation. Se debe anadir los parametros 
*/
__global__ void solvation () {


}


/**
* Funcion para manejar el lanzamiento de CUDA 
*/
void forces_GPU_AU (int atoms_r, int atoms_l, int nlig, float *rec_x, float *rec_y, float *rec_z, float *lig_x, float *lig_y, float *lig_z, int* rectype, int* ligtype, float *ql ,float *qr, float *energy, struct autodock_param_t *a_params, int nconformations){



	//Definir numero de hilos y bloques
	dim3 thread (1);
	dim3 block (1);

	//Anadir parametros
	solvation <<< block,thread>>> ();




}



/**
* Funcion que implementa la solvatacion en CPU 
*/
void forces_CPU_AU (int atoms_r, int atoms_l, int nlig, float *rec_x, float *rec_y, float *rec_z, float *lig_x, float *lig_y, float *lig_z, int* rectype, int* ligtype, float *ql ,float *qr, float *energy, struct autodock_param_t *a_params, int nconformations){

	float dist, temp_desolv = 0,miatomo[3], e_desolv;
	int j,i;
	int ind1, ind2;
	int total;

	float difx,dify,difz, solv_asp_1, solv_asp_2, solv_vol_1, solv_vol_2,solv_qasp_1,solv_qasp_2;
	float  mod2x, mod2y, mod2z;	

	total = nconformations * nlig;

	for (int k=0; k < (nconformations*nlig); k+=nlig)
	{
		for(int i=0;i<atoms_l;i++){					
			e_desolv = 0;
			ind1 = ligtype[i];
			miatomo[0] = *(lig_x + k + i);
			miatomo[1] = *(lig_y + k + i);
			miatomo[2] = *(lig_z + k + i);
			solv_asp_1 = a_params[ind1].asp;
			solv_vol_1 = a_params[ind1].vol;
			for(int j=0;j<atoms_r;j++){				
				e_desolv = 0;
				ind2 = rectype[j];
				solv_asp_2 = a_params[ind2].asp;
				solv_vol_2 = a_params[ind2].vol;
				difx= (rec_x[j]) - miatomo[0];
				dify= (rec_y[j]) - miatomo[1];
				difz= (rec_z[j]) - miatomo[2];
				mod2x=difx*difx;
				mod2y=dify*dify;
				mod2z=difz*difz;
			
				difx=mod2x+mod2y+mod2z;
				dist = sqrtf(difx);
							
				e_desolv = ((solv_asp_1 * solv_vol_2) + (QASP * fabs(ql[i]) *  solv_vol_2) + (solv_asp_2 * solv_vol_1) + (QASP * fabs(qr[j]) * solv_vol_1)) * exp(-difx/(2*G_D_2));
				temp_desolv += e_desolv;	
			
			}
		}
		energy[k/nlig] = temp_desolv;
		temp_desolv = 0;
	}
	printf("Desolvation term value: %f\n",energy[0]);
}


extern void solver_AU(int mode, int atoms_r, int atoms_l,  int nlig, float *rec_x, float *rec_y, float *rec_z, float *lig_x, float *lig_y, float *lig_z, int* rectype, int* ligtype, float *ql, float *qr, float *energy_desolv, struct autodock_param_t *a_params, int nconformaciones) 
{
	double elapsed_i, elapsed_o;
	
	switch (mode) {
		case 0://Sequential execution
			printf("\* DESOLVATION TERM FUNCTION CPU MODE *\n");
			printf("**************************************\n");			
			printf("Conformations: %d\tMode: %d\n",nconformaciones,mode);			
			elapsed_i = wtime();
			forces_CPU_AU (atoms_r,atoms_l,nlig,rec_x,rec_y,rec_z,lig_x,lig_y,lig_z,rectype,ligtype,ql,qr,energy_desolv,a_params,nconformaciones);
			elapsed_o = wtime() - elapsed_i;
			printf ("CPU Processing time: %f (seg)\n", elapsed_o);
			break;
		case 1: //OpenMP execution
			printf("\* DESOLVATION TERM FUNCTION OPENMP MODE *\n");
			printf("**************************************\n");			
			printf("Conformations: %d\tMode: %d\n",nconformaciones,mode);			
			elapsed_i = wtime();
			forces_OMP_AU (atoms_r,atoms_l,nlig,rec_x,rec_y,rec_z,lig_x,lig_y,lig_z,rectype,ligtype,ql,qr,energy_desolv,a_params,nconformaciones);
			elapsed_o = wtime() - elapsed_i;
			printf ("OpenMP Processing time: %f (seg)\n", elapsed_o);
			break;
		case 2: //CUDA exeuction
            printf("\* DESOLVATION TERM FUNCTION CUDA MODE *\n");
            printf("**************************************\n");
            printf("Conformations: %d\tMode: %d\n",nconformaciones,mode);
			elapsed_i = wtime();
			forces_GPU_AU (atoms_r,atoms_l,nlig,rec_x,rec_y,rec_z,lig_x,lig_y,lig_z,rectype,ligtype,ql,qr,energy_desolv,a_params,nconformaciones);
			elapsed_o = wtime() - elapsed_i;
			printf ("GPU Processing time: %f (seg)\n", elapsed_o);			
			break; 	
	  	default:
 	     	printf("Wrong mode type: %d.  Use -h for help.\n", mode);
			exit (-1);	
	} 		
}
