"""
Unit and regression test for the qm_2019_sss_3 package.
"""

# Import package, test suite, and other packages as needed
import qm_2019_sss_3
import pytest
import sys
import numpy as np

@pytest.fixture()
def noble_gas():
    name = 'Argon'
    atomic_coordinates = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 5.0]])
    argon = qm_2019_sss_3.NobleGas(name,atomic_coordinates)
    return argon

def test_qm_2019_sss_3_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "qm_2019_sss_3" in sys.modules

@pytest.mark.parametrize("name,coordinates,error",[(25,np.array([[0.0, 0.0, 0.0], [1.0, 4.0, 5.0]]),TypeError),('abc',np.array([[0.0, 0.0, 0.0], [1.0, 4.0, 5.0]]),NameError)])
def test_noble_gas_failure_type_error(name,coordinates,error):
    """Testing the noblegas class"""
    with pytest.raises(error):
        noble_gas = qm_2019_sss_3.NobleGas(name,coordinates)

def test_noblegas_atom():
    atom_no = noble_gas.atom(1)
    assert atom_no == 0

def test_noblegas_orb(noble_gas):
    orb_type = noble_gas.orb(1)
    assert orb_type == 'px'

def test_noble_gas_ao_index(noble_gas):
    ao_index = noble_gas.ao_index(1,'px')
    assert ao_index == 5