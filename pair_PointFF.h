/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(PointFF,PairPointFF);
// clang-format on
#else

#ifndef LMP_PAIR_Point_FF_H
#define LMP_PAIR_Point_FF_H

#include "pair.h"
#include "torch/torch.h"

namespace LAMMPS_NS {

class PairPointFF : public Pair {
  friend class Pair;

	public:
		PairPointFF(class LAMMPS *);
		virtual ~PairPointFF();

		virtual void compute(int, int);
		void settings(int, char **);
		void coeff(int, char **);
		void init_style();
		double init_one(int, int);
	
	
	protected:
		double cut_global;
		double **cut;
		int required_atoms;
		void allocate();
	
	
	private:
		std::vector<float> electorn;                
		torch::jit::Module model;
		int CUDA_device; 
		int feature_dims;
	
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

*/
