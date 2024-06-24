# include <iostream>
# include <Eigen/Dense>
# include <vector>
// model parameters = dipole only
// string comparison or finding the index of an element in an array
Eigen::MatrixXd calculate_fock_matrix_fast(Eigen::MatrixXd hamiltonian_matrix, Eigen::MatrixXd interaction_matrix, Eigen::MatrixXd density_matrix, double model_parameter);
const std::vector<int> orbital_types = {0, 1, 2, 3};
const int orbitals_per_atom = 4;

