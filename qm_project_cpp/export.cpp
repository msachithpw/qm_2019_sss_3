# include <pybind11/pybind11.h>
# include "qm_project.hpp"
# include <pybind11/eigen.h>
# include <pybind11/stl.h>

PYBIND11_MODULE(sss_cpp,m)
{
	m.doc() = "This is the c++ module for the scf_cycle";
	m.def("calculate_fock_matrix_fast",calculate_fock_matrix_fast,"C++ code to calculate fock matrix");

}

