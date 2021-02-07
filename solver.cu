#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>
#include "wtime.h"
#include "definitions.h"
#include "energy_struct.h"
#include "cuda_runtime.h"
#include "solver.h"
#define HILOS 1
using namespace std;

/**
* Kernel del calculo de la solvation. Se debe anadir los parametros 
*/

__constant__ struct autodock_param_t a_params_c [MAXTYPES];

extern void save_params (struct autodock_param_t *a_params)
{
  cudaError_t cudaStatus;
  cudaStatus = cudaMemcpyToSymbol(a_params_c, a_params, MAXTYPES * sizeof(struct autodock_param_t));
  cudaDeviceSynchronize();
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) printf("Error memoria constante %d\n",cudaStatus);
}

__global__ void solvation (int nlig, int atoms_r, int atoms_l, float *d_rec_x, float *d_rec_y, float *d_rec_z, float *d_lig_x, float *d_lig_y, float *d_lig_z, int* d_rectype, int* d_ligtype, float *d_ql ,float *d_qr, float *d_energy, int nconformations) {
  
 // __shared__ float energy_shared[128];
 
  int tid = threadIdx.x;
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  int j, i;
  float dist, temp_desolv = 0,miatomo[3], e_desolv;
  int ind1, ind2;
 
  float difx,dify,difz, solv_asp_1, solv_asp_2, solv_vol_1, solv_vol_2,solv_qasp_1,solv_qasp_2;
  float  mod2x, mod2y, mod2z;
  int atom = index % nlig;
  int conform = index / nlig;
  int pos;
  //printf("tid: %d, index: %d , bloque: %d, total: %d\n", tid, index, blockIdx.x, (nconformations*nlig));
  /*if (tid ==0) {
    printf("atom %d, conform %d\n", atom, conform); 
    printf("total: %d\n", (nconformations));
  }*/
  if(index < (nconformations))
  { 
     pos = conform * nlig;   
     for(int i=0; i < atoms_l; i++){ 
      e_desolv = 0;
      ind1 = d_ligtype[i];
      miatomo[0] = *(d_lig_x + pos + i);
      miatomo[1] = *(d_lig_y + pos + i);
      miatomo[2] = *(d_lig_z + pos + i);
      solv_asp_1 = a_params_c[ind1].asp;
      solv_vol_1 = a_params_c[ind1].vol;
   
      for(int j=0;j<atoms_r;j++){
        e_desolv = 0;
        ind2 = d_rectype[j];
        solv_asp_2 = a_params_c[ind2].asp;
        solv_vol_2 = a_params_c[ind2].vol;
        difx= (d_rec_x[j]) - miatomo[0];
        dify= (d_rec_y[j]) - miatomo[1];
        difz= (d_rec_z[j]) - miatomo[2];
        mod2x=difx*difx;
        mod2y=dify*dify;
        mod2z=difz*difz;
    
        difx=mod2x+mod2y+mod2z;
        dist = sqrtf(difx);
        
        e_desolv = ((solv_asp_1 * solv_vol_2) + (QASP * fabs(d_ql[i+pos]) *  solv_vol_2) + (solv_asp_2 * solv_vol_1) + (QASP * fabs(d_qr[j]) * solv_vol_1)) * exp(-difx/(2*G_D_2));
        
        temp_desolv += e_desolv;
      }
    }
    d_energy[index] = temp_desolv;
    temp_desolv = 0;
    
  }
 
 
}


/**
* Funcion para manejar el lanzamiento de CUDA 
*/
void forces_GPU_AU (int atoms_r, int atoms_l, int nlig, float *rec_x, float *rec_y, float *rec_z, float *lig_x, float *lig_y, float *lig_z, int* rectype, int* ligtype, float *ql ,float *qr, float *energy, struct autodock_param_t *a_params, int nconformations){
  
  //DEFINIR
  float *d_rec_x, *d_rec_y, *d_rec_z, *d_lig_x, *d_lig_y, *d_lig_z, *d_ql, *d_qr, *d_energy;
  int *d_rectype, *d_ligtype;
  int d_atoms_r, d_nlig;
  int total;
  total = nconformations * nlig;

  //MEMSIZE
  int memsize = nconformations * sizeof(float);
  int atomsRsize = atoms_r *sizeof(float);
  int atomsLsize = atoms_l * sizeof(float);
  int atomsLsize_int = atoms_l * sizeof(int);
  int atomsRsize_int = atoms_r * sizeof(int);
  
  save_params(a_params);
  
  //RESERVA DE MEMORIA GPU
  cudaMalloc((void**)&d_rec_x, atomsRsize);
  cudaMalloc((void**)&d_rec_y, atomsRsize);
  cudaMalloc((void**)&d_rec_z, atomsRsize);
  cudaMalloc((void**)&d_lig_x, atomsLsize);
  cudaMalloc((void**)&d_lig_y, atomsLsize);
  cudaMalloc((void**)&d_lig_z, atomsLsize);
  cudaMalloc((void**)&d_ql, atomsLsize);
  cudaMalloc((void**)&d_qr, atomsRsize);
  cudaMalloc((void**)&d_energy, memsize);
  //cudaMemset(d_energy,0,memsize);
  cudaMalloc((void**)&d_rectype, atomsRsize_int);
  cudaMalloc((void**)&d_ligtype, atomsLsize_int);

  //TRANSFERIR A GPU
  cudaMemcpy(d_rec_x, rec_x, atomsRsize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_rec_y, rec_y, atomsRsize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_rec_z, rec_z, atomsRsize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_lig_x, lig_x, atomsLsize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_lig_y, lig_y, atomsLsize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_lig_z, lig_z, atomsLsize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_ql, ql, atomsLsize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_qr, qr,  atomsRsize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_energy, energy, memsize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_rectype, rectype, atomsRsize_int, cudaMemcpyHostToDevice);
  cudaMemcpy(d_ligtype, ligtype, atomsLsize_int, cudaMemcpyHostToDevice);
  
	//Definir ero de hilos y bloques
	dim3 thread (HILOS);
	dim3 block (ceil(nconformations/HILOS));

	//Anadir parametros
 // for (int i=0; i<total; i+=nlig){ 
	  solvation <<< block,thread>>> (nlig, atoms_r, atoms_l, d_rec_x, d_rec_y, d_rec_z, d_lig_x, d_lig_y, d_lig_z, d_rectype, d_ligtype, d_ql , d_qr, d_energy, nconformations);
 //}
  
  //TRAER RESULTADO
  cudaMemcpy(energy, d_energy, memsize, cudaMemcpyDeviceToHost);
  printf("Desolvation term value: %f\n",energy[0]);
 
  
  cudaFree(d_rec_x);
  cudaFree(d_rec_y);
  cudaFree(d_rec_z);
  cudaFree(d_lig_x);
  cudaFree(d_lig_y);
  cudaFree(d_lig_z);
  cudaFree(d_energy);
  cudaFree(d_rectype);
  cudaFree(d_ligtype);
  cudaFree(d_ql);
  cudaFree(d_qr);
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
			  /*if(j == 4 && i == 0){
  printf("rec_x[4]: %f, miatomo[0]: %f\n", rec_x[4], miatomo[0]);
  printf("difx: %f = rec_x4: %f - miatomo: %f\n", difx, rec_x[4], miatomo[0]);
  printf("rec_y[4]: %f, miatomo[1]: %f\n", rec_y[4], miatomo[1]);
  printf("dify: %f = rec_y4: %f - miatomo: %f\n", dify, rec_y[4], miatomo[1]);
  
}*/
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


extern void solver_AU(int mode, int atoms_r, int atoms_l,  int nlig, float *rec_x, float *rec_y, float *rec_z, float *lig_x, float *lig_y, float *lig_z, int* rectype, int* ligtype, float *ql, float *qr, float *energy_desolv, struct autodock_param_t *a_params, int nconformaciones,int nthreads) 
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
			forces_OMP_AU (atoms_r,atoms_l,nlig,rec_x,rec_y,rec_z,lig_x,lig_y,lig_z,rectype,ligtype,ql,qr,energy_desolv,a_params,nconformaciones,nthreads);
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
