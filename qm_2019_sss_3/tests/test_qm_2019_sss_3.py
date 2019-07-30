"""
Unit and regression test for the qm_2019_sss_3 package.
"""

# Import package, test suite, and other packages as needed
from qm_2019_sss_3 import noblegas
import pytest
import sys
import numpy as np

@pytest.fixture()
def noble_gas():
    name = 'Argon'
    atomic_coordinates = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 5.0]])
    argon = noblegas.NobleGas(name,atomic_coordinates)
    return argon

def test_qm_2019_sss_3_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "qm_2019_sss_3" in sys.modules

@pytest.mark.parametrize("name,coordinates,error",[(25,np.array([[0.0, 0.0, 0.0], [1.0, 4.0, 5.0]]),TypeError),('abc',np.array([[0.0, 0.0, 0.0], [1.0, 4.0, 5.0]]),NameError)])
def test_noble_gas_failure_type_error(name,coordinates,error):
    """Testing the noblegas class"""
    with pytest.raises(error):
        noble_gas = noblegas.NobleGas(name,coordinates)

def test_noblegas_atom(noble_gas):
    atom_no = noble_gas.atom(1)
    assert atom_no == 0

def test_noblegas_orb(noble_gas):
    orb_type = noble_gas.orb(1)
    assert orb_type == 'px'

def test_noble_gas_ao_index(noble_gas):
    ao_index = noble_gas.ao_index(1,'px')
    assert ao_index == 5

def test_scf_cycle(noble_gas):
        hf = noblegas.HartreeFock(noble_gas)
        a,b = hf.scf_cycle()
        a1 =  [[ 5.8e-08,-4.4e-05, -5.8e-05, -7.3e-05, -4.5e-08, -9.2e-05, -1.2e-04, -1.5e-04],[-4.4e-05,  1.0e+00 ,-1.4e-08 ,-1.7e-08,  9.2e-05, -8.0e-09, -1.1e-08, -1.3e-08],[-5.8e-05, -1.4e-08,  1.0e+00, -2.3e-08,  1.2e-04, -1.1e-08, -1.4e-08 ,-1.8e-08],[-7.3e-05, -1.7e-08, -2.3e-08,  1.0e+00,  1.5e-04, -1.3e-08, -1.8e-08, -2.2e-08],[-4.5e-08,  9.2e-05,  1.2e-04  ,1.5e-04,  5.8e-08,  4.4e-05,  5.8e-05,  7.3e-05],[-9.2e-05, -8.0e-09, -1.1e-08, -1.3e-08,  4.4e-05,  1.0e+00, -1.4e-08 ,-1.7e-08],[-1.2e-04, -1.1e-08 ,-1.4e-08, -1.8e-08,  5.8e-05, -1.4e-08,  1.0e+00, -2.3e-08],[-1.5e-04, -1.3e-08, -1.8e-08, -2.2e-08,  7.3e-05, -1.7e-08, -2.3e-08,  1.0e+00]]

        b1 = [[ 5.4e+00 ,2.6e-04 , 3.5e-04 , 4.4e-04 , 6.3e-04 , 5.5e-04  ,7.3e-04 , 9.2e-04],[ 2.6e-04 ,-5.9e-01 , 4.7e-09 , 5.8e-09 ,-5.5e-04 , 3.0e-04  ,2.2e-03 , 2.7e-03],[ 3.5e-04 , 4.7e-09 ,-5.9e-01 , 7.8e-09, -7.3e-04 , 2.2e-03 , 1.6e-03,  3.6e-03],[ 4.4e-04 , 5.8e-09  ,7.8e-09 ,-5.9e-01 ,-9.2e-04  ,2.7e-03 , 3.6e-03  ,3.2e-03],[ 6.3e-04 ,-5.5e-04, -7.3e-04 ,-9.2e-04 , 5.4e+00 ,-2.6e-04 ,-3.5e-04, -4.4e-04],[ 5.5e-04 , 3.0e-04 , 2.2e-03 , 2.7e-03 ,-2.6e-04 ,-5.9e-01,  4.7e-09 , 5.8e-09],[ 7.3e-04  ,2.2e-03  ,1.6e-03 , 3.6e-03 ,-3.5e-04 , 4.7e-09 ,-5.9e-01 , 7.8e-09],[ 9.2e-04 , 2.7e-03 , 3.6e-03  ,3.2e-03 ,-4.4e-04 , 5.8e-09  ,7.8e-09, -5.9e-01]]
        # np.linalg.norm(a - a1) == 0 and 
        #  
        assert (np.linalg.norm(a-a1) <= 1e-4 ) and (np.linalg.norm(b - b1) <= 1e-2)