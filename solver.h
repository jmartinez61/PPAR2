#ifndef __SOLVER_H
#define __SOLVER_H

extern void solver_AU(int mode, int atoms_r, int atoms_l,  int nlig, float *rec_x, float *rec_y, float *rec_z, float *lig_x, float *lig_y, float *lig_z, int* rectype, int* ligtype, float *ql, float *qr, float *energy_desolv, struct autodock_param_t *a_params, int nconformaciones, int nthreads);

extern void forces_OMP_AU (int atoms_r, int atoms_l, int nlig, float *rec_x, float *rec_y, float *rec_z, float *lig_x, float *lig_y, float *lig_z, int* rectype, int* ligtype, float *ql ,float *qr, float *energy, struct autodock_param_t *a_params, int nconformations, int nthreads);
#endif
