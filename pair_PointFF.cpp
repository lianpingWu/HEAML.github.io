// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

// lammps (CPU)
// PointFF (GPU) -> CPU

#include "pair_PointFF.h"

#include "atom.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"
#include "update.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"

#include "torch/torch.h"
#include "torch/script.h"
#include <iostream>
#include <memory>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairPointFF::PairPointFF(LAMMPS *lmp) : Pair(lmp) // 
	writedata = 1; // 
	cut_global = 0.0; // 
	required_atoms = 0; // 
	feature_dims = 4; //
}

/* ---------------------------------------------------------------------- */

PairPointFF::~PairPointFF() // 
{
  if (allocated) { 
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cut);
  }
}

/* ----------------------------------------------------------------------
   Neural network forces calculation
------------------------------------------------------------------------- */

void PairPointFF::compute(int eflag, int vflag)
{
	/* ---------- basic information ---------- */
	int nlocal = atom->nlocal;	// 
	int step = update->ntimestep;	// 
	double **f = atom->f;	// 
  double **x = atom->x;	// 
	int *type = atom->type; // 
	
	int loopi, loopj, loopk;
	int i, j, ii, jj, jnum;
	int *ilist, *jlist, *numneigh, **firstneigh;
	double delx, dely, delz;
	
	/* ---------- dim: nlocal x feature_dims x required_atoms ---------- */
	ilist = list->ilist; // 
  numneigh = list->numneigh; // 
  firstneigh = list->firstneigh; // 
	
	double *sort_temp = new double[5]; // 
	float neighbor_array[nlocal][feature_dims][required_atoms]; // 
	
  for (ii = 0; ii < nlocal; ii++) {
    i = ilist[ii]; // 
    jlist = firstneigh[i]; // list, 
		jnum = numneigh[i]; // int, 
		
		double neighbor_temp[5][jnum];
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj]; // 
      j &= NEIGHMASK;
			
      delx = x[j][0] - x[i][0]; // 
      dely = x[j][1] - x[i][1]; // 
      delz = x[j][2] - x[i][2; // 

      neighbor_temp[0][jj] = delx;
      neighbor_temp[1][jj] = dely;
      neighbor_temp[2][jj] = delz;
      neighbor_temp[3][jj] = abs(electorn[type[j] - 1] - electorn[type[i] - 1]); // 
      neighbor_temp[4][jj] = delx * delx + dely * dely + delz * delz; // 
    }
		
		/* ----------  ---------- */
    for (loopi = 0; loopi < jnum; loopi++) {
			for (loopj = loopi + 1; loopj < jnum; loopj++) {
				
				for (loopk = 0; loopk < 5; loopk++) {
					sort_temp[loopk] = neighbor_temp[loopk][loopi];
				}
				
				if (neighbor_temp[4][loopi] > neighbor_temp[4][loopj]) {
					for (loopk = 0; loopk < 5; loopk++) {
						neighbor_temp[loopk][loopi] = neighbor_temp[loopk][loopj];
						neighbor_temp[loopk][loopj] = sort_temp[loopk];
					}
				}
				
			}
    }
		
		/* ---------- ---------- */
    for (loopi = 0; loopi < feature_dims; loopi++) {
			for (loopj = 0; loopj < required_atoms; loopj++) {
				neighbor_array[ii][loopi][loopj] = (float)neighbor_temp[loopi][loopj];
      }
    }

	}
	
	/* ---------- define the inputs ---------- */
	torch::Tensor inputs_tensor = torch::from_blob(neighbor_array, {nlocal, feature_dims, required_atoms}).requires_grad_(true);	// 
	inputs_tensor = inputs_tensor.to(torch::Device("cuda:" + std::to_string(CUDA_device)));	//
	std::vector<torch::jit::IValue> inputs;	// 
	inputs.push_back(inputs_tensor);	// 
	
	/* ---------- torch  ---------- */
	torch::Tensor energy = model.forward(inputs).toTensor(); // 
	torch::Tensor grad_output = torch::ones_like(energy);	// 
	
	torch::Tensor atomic_forces = torch::autograd::grad(	// 
		/*outputs=*/{energy},
		/*inputs=*/{inputs_tensor},
		/*grad_outputs=*/{grad_output},
		/*retain_graph=*/true,
		/*create_graph=*/true,
		/*allow_unused=*/false
	)[0];
	
	/* ---------- lammps ---------- */
	if (eflag) eng_vdwl = energy.item<float>();	// 
	
	/* ---------- lammps ---------- */
	atomic_forces = atomic_forces.cpu(); // 
	torch::Tensor forces_lmp = torch::sum(atomic_forces.index_select(1, torch::arange(0, 3)), 2); // 力
	
  /* ---------- lammps ---------- */
	for (loopi = 0; loopi < nlocal; loopi++) {
		f[loopi][0] = forces_lmp[loopi][0].item<float>();
		f[loopi][1] = forces_lmp[loopi][1].item<float>();
		f[loopi][2] = forces_lmp[loopi][2].item<float>();
	}
  
  /* ----------  ---------- */
  if (vflag) {
		
		inputs_tensor = inputs_tensor.cpu();							// 
		
		torch::Tensor Fx = atomic_forces.index_select(1, at::tensor(0).toType(at::kLong)); // 
		torch::Tensor Fy = atomic_forces.index_select(1, at::tensor(1).toType(at::kLong)); // 
		torch::Tensor Fz = atomic_forces.index_select(1, at::tensor(2).toType(at::kLong)); // 
		torch::Tensor loc_x = inputs_tensor.index_select(1, at::tensor(0).toType(at::kLong));  // 
		torch::Tensor loc_y = inputs_tensor.index_select(1, at::tensor(1).toType(at::kLong));  // 
		torch::Tensor loc_z = inputs_tensor.index_select(1, at::tensor(2).toType(at::kLong));  // 
		
		// lammps
		virial[0] = -	at::sum(at::mul(Fx, loc_x)).item<float>(); 
		virial[1] = - at::sum(at::mul(Fy, loc_y)).item<float>();
		virial[2] = - at::sum(at::mul(Fz, loc_z)).item<float>();
		virial[3] = - at::sum(at::mul(Fy, loc_z)).item<float>();
		virial[4] = - at::sum(at::mul(Fx, loc_z)).item<float>();
		virial[5] = - at::sum(at::mul(Fx, loc_y)).item<float>();
		
  }
	
}


/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairPointFF::allocate() 
{ // 
  allocated = 1; 
  int n = atom->ntypes;
  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(cut,n+1,n+1,"pair:cut");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairPointFF::settings(int narg, char **arg) // 
{ 
  cut_global = utils::numeric(FLERR, arg[0], false, lmp); // 
  required_atoms = utils::numeric(FLERR, arg[1], false, lmp); // 
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairPointFF::coeff(int narg, char **arg) // 
{
  int ntypes = atom->ntypes;
  if (!allocated) allocate();
	
  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error);
  utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error);

  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      setflag[i][j] = 1;
      cutsq[i][j] = cut_global*cut_global;
      cut[i][j] = cut_global;
    }
  }
	
	/* ----------  ---------- */
	int mpi_rank, GPU_counts; // C
	MPI_Comm_rank(world, &mpi_rank); // 
	GPU_counts = torch::cuda::device_count(); // 
	CUDA_device = mpi_rank % GPU_counts; // 
	model = torch::jit::load(std::string(arg[2]), torch::Device("cuda:" + std::to_string(CUDA_device)));		// load torch model 
	model.eval();
	
	/* ---------- ---------- */
	electorn.clear();
	for (int loopi = 3; loopi < narg; loopi++) {
    electorn.push_back(utils::numeric(FLERR, arg[loopi], false, lmp));
  }
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairPointFF::init_one(int i, int j)
{
  return cut_global;
}


void PairPointFF::init_style()
{
	int irequest = neighbor->request(this,instance_me);
	neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
	neighbor->requests[irequest]->ghost = 0;
}

