# include "qm_project.hpp"

double chi_on_atom(int q);

int atom(int index)
{
  return index / orbitals_per_atom;
}
int orb(int index)
{
  int orb_index = index % orbitals_per_atom;
  return orbital_types[orb_index];
}

int ao_index(int atom, int orbital)
{
  int p = atom * orbitals_per_atom + orbital;
  return p;
}
double chi_on_atom(int o1,int o2, int o3, double model_parameter)
{
  if(o1 == o2 && o3 == 0)
    return 1.0;
  else if (o1 == o3 && o3 >0 && o2 ==0)
    return model_parameter;
  else if(o2 == o3 && o3>0 && o1 ==0)
    return model_parameter;
  return 0.0;
}
Eigen::MatrixXd calculate_fock_matrix_fast(Eigen::MatrixXd hamiltonian_matrix, Eigen::MatrixXd interaction_matrix, Eigen::MatrixXd density_matrix, double model_parameter)
{
  // Returns the Fock materi defined by the input hamiltonian, interaction and density matrices.
 
  int ndof = hamiltonian_matrix.rows();
  //    std::cout << ndof << std::endl;
  Eigen::MatrixXd fock_matrix = hamiltonian_matrix;
    
  // Hartree potential term
  for(int p = 0; p < ndof; p++)
    {
      for(auto orb_q: orbital_types)
	{
	  int q = ao_index(atom(p),orb_q); // p & q on same atom
	  for(auto orb_t: orbital_types)
	    {
	      int t = ao_index(atom(p),orb_t); // p & t on same atom
	      double chi_pqt = chi_on_atom(orb(p), orb_q,orb_t, model_parameter);
	      for(int r = 0; r<ndof; r++)
		{
		  for(auto orb_s: orbital_types)
		    {
		      int s = ao_index(atom(r),orb_s); // r & s on same atom
		      for(auto orb_u: orbital_types)
			{
			  int u = ao_index(atom(r),orb_u); // r & u on the same atom
			  double chi_rsu = chi_on_atom(orb(r),orb_s,orb_u,model_parameter);
			  fock_matrix(p,q) += 2.0 * chi_pqt * chi_rsu * interaction_matrix(t,u) * density_matrix(r,s);
			}
		    }
		}
	    }
	}
    }
    
    
  // fock exchange term
  for(int p = 0;p < ndof; p++)
    {
      for(auto orb_s: orbital_types)
        {
	  int s = ao_index(atom(p),orb_s); // p & s on same atom
	  for(auto orb_u: orbital_types)
            {
	      int u = ao_index(atom(p),orb_u); //p & u on same atom
	      double chi_psu = chi_on_atom(orb(p),orb_s,orb_u, model_parameter);
	      for(int q = 0; q< ndof; q++)
                {
		  for(auto orb_r: orbital_types)
                    {
		      int r = ao_index(atom(q),orb_r); // q & r on the same atom
		      for(auto orb_t: orbital_types)
                        {
			  int t = ao_index(atom(q),orb_t); //q & t on the same atom
			  double chi_rqt = chi_on_atom(orb_r,orb(q),orb_t, model_parameter);
			  fock_matrix(p,q) -= chi_rqt * chi_psu * interaction_matrix(t,u) * density_matrix(r,s);
                            
                        }
                    }
                }
            }
        }
    }
  return fock_matrix;
}

