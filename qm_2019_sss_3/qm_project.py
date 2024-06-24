import numpy as np

def atom(ao_index):
    """
    Returns the atom index part of an atomic orbital index.

    Parameters
    ----------
    ao_index: int
        index of the atomic orbital

    Returns
    -------
    ao_index // orbitals_per_atom: int
        index of the atom
    """
    return ao_index // orbitals_per_atom

def orb(ao_index):
    """
    Returns the orbital type of an atomic orbital index.

    Parameters
    ----------
    ao_index: int
        index of the atomic orbital

    Returns
    -------
    orbital_types[orb_index]: str
        orbital type
    """
    orb_index = ao_index % orbitals_per_atom
    return orbital_types[orb_index]

def ao_index(atom_p, orb_p):
    """
    Returns the atomic orbital index for a given atom index and orbital type.

    Parameters
    ----------
    atom_p: int
        atomic index
    orb_p: str
        atomic orbital
    Returns
    -------
    p: int
        Returns the atomic orbital index for a given atom index and orbital type. 

    """
    p = atom_p * orbitals_per_atom
    p += orbital_types.index(orb_p)
    return p

def hopping_energy(o1, o2, r12, model_parameters):
    """Returns the hopping matrix element for a pair of orbitals of type o1 & o2 separated by a vector r12.

    Parameters
    ----------
    o1 : str
        Orbital type of the 1st atom
    o2 : str
        Orbital type of the 2nd atom
    r12 : np.array
       A vector pointing from the second to the first atom 
    model_parameters : dict
        **A dictionary of model parameters fit to this data using semiempirical model parameterization**

    Returns
    -------
    ans : float
        The answer is the hopping energy.
    """
    r12_rescaled = r12 / model_parameters['r_hop']
    r12_length = np.linalg.norm(r12_rescaled)
    ans = np.exp( 1.0 - r12_length**2 )
    if o1 == 's' and o2 == 's':
        ans *= model_parameters['t_ss']
    if o1 == 's' and o2 in p_orbitals:
        ans *= np.dot(vec[o2], r12_rescaled) * model_parameters['t_sp']
    if o2 == 's' and o1 in p_orbitals:
        ans *= -np.dot(vec[o1], r12_rescaled)* model_parameters['t_sp']
    if o1 in p_orbitals and o2 in p_orbitals:
        ans *= ( (r12_length**2) * np.dot(vec[o1], vec[o2]) * model_parameters['t_pp2']
                 - np.dot(vec[o1], r12_rescaled) * np.dot(vec[o2], r12_rescaled)
                 * ( model_parameters['t_pp1'] + model_parameters['t_pp2'] ) )
    return ans

def coulomb_energy(o1, o2, r12):
    """
    Returns the Coulomb matrix element for a pair of multipoles of type o1 & o2 separated by a vector r12.

    Parameters
    ----------
    o1: str
        type of first orbital
    o2: str
        type of second orbital
    r12: np.array
        separation of orbital one and two
    Returns
    -------
    ans: flot
        Coulomb matrix element for a pair of multipoles of type o1 & o2 separated by a vector r12.

    """
    r12_length = np.linalg.norm(r12)
    if o1 == 's' and o2 == 's':
        ans = 1.0 / r12_length
    if o1 == 's' and o2 in p_orbitals:
        ans = np.dot(vec[o2], r12) / r12_length**3
    if o2 == 's' and o1 in p_orbitals:
        ans = -1 * np.dot(vec[o1], r12) / r12_length**3
    if o1 in p_orbitals and o2 in p_orbitals:
        ans = (
            np.dot(vec[o1], vec[o2]) / r12_length**3 -
            3.0 * np.dot(vec[o1], r12) * np.dot(vec[o2], r12) / r12_length**5)
    return ans

def pseudopotential_energy(o, r, model_parameters):
    """
    Returns the energy of a pseudopotential between a multipole of type o and an atom separated by a vector r.

    Parameters
    ----------
    o: str
        orbital type
    r: np.array
        pseudo_vector*
    model_parameters: dict
        *

    Returns
    -------
    ans: float
        pseudopotential between a multipole of type o and an atom
    """
    ans = model_parameters['v_pseudo']
    r_rescaled = r / model_parameters['r_pseudo']
    ans *= np.exp(1.0 - np.dot(r_rescaled, r_rescaled))
    if o in p_orbitals:
        ans *= -2.0 * np.dot(vec[o], r_rescaled)
    return ans


def calculate_energy_ion(atomic_coordinates):
    """
    Returns the ionic contribution to the total energy for an input list of atomic coordinates.

    Parameters
    ----------
    atomic_coordinates: np.array
        An array of xyz coordinates of atoms

    Returns
    -------
    energy_ion: float
        Ionic energy of the system.
    """
    energy_ion = 0.0
    for i, r_i in enumerate(atomic_coordinates):
        for j, r_j in enumerate(atomic_coordinates):
            if i < j:
                energy_ion += (ionic_charge**2) * coulomb_energy(
                    's', 's', r_i - r_j)
    return energy_ion

def calculate_potential_vector(atomic_coordinates, model_parameters):
    """Returns the electron-ion potential energy vector for an input list of atomic coordinates.

    Parameters
    ----------
    atomic_coordinates: np.array
        An array of atomic coordinates. 
    model_parameters: dict
        A dictionary of model parameters fit to this data using semiempirical model parameterization

    Returns
    -------
    potential_vector: np.array
    An array of electron-ion potential energy vectors for an input list of atomic coordinates.
    """
    ndof = len(atomic_coordinates) * orbitals_per_atom
    potential_vector = np.zeros(ndof)
    for p in range(ndof):
        potential_vector[p] = 0.0
        for atom_i, r_i in enumerate(atomic_coordinates):
            r_pi = atomic_coordinates[atom(p)] - r_i
            if atom_i != atom(p):
                potential_vector[p] += (
                    pseudopotential_energy(orb(p), r_pi, model_parameters) -
                    ionic_charge * coulomb_energy(orb(p), 's', r_pi))
    return potential_vector

def calculate_interaction_matrix(atomic_coordinates, model_parameters):
    """
    Returns the electron-electron interaction energy matrix for an input list of atomic coordinates.

    Parameters
    ----------
    atomic_coordinates: np.array
        An array of xyz coordinates of atoms
    model_parameters: dict
        A dictionary of model parameters fit to this data using semiempirical model parameterization.

    Returns
    -------
    interaction_matrix: np.array
        An array of the electron-electron interaction energy.
    """
    ndof = len(atomic_coordinates)*orbitals_per_atom
    interaction_matrix = np.zeros( (ndof,ndof) )
    for p in range(ndof):
        for q in range(ndof):
            if atom(p) != atom(q):
                r_pq = atomic_coordinates[atom(p)] - atomic_coordinates[atom(q)]
                interaction_matrix[p,q] = coulomb_energy(orb(p), orb(q), r_pq)
            if p == q and orb(p) == 's':
                interaction_matrix[p,q] = model_parameters['coulomb_s']
            if p == q and orb(p) in p_orbitals:
                interaction_matrix[p,q] = model_parameters['coulomb_p']                
    return interaction_matrix

def chi_on_atom(o1, o2, o3, model_parameters):
    """
    Returns the value of the chi tensor for 3 orbital indices on the same atom.

    Parameters
    ----------
    o1: str
        The orbital type.
    o2: str
        The orbital type.
    o3: str
        The orbital type.
    model_parameters: np.array
        A dictionary of model parameters fit to this data using semiempirical model parameterization.

    Returns
    -------
    model_parameters['dipole'] or constant: float
        The value of the chi tensor.
    """
    if o1 == o2 and o3 == 's':
        return 1.0
    if o1 == o3 and o3 in p_orbitals and o2 == 's':
        return model_parameters['dipole']
    if o2 == o3 and o3 in p_orbitals and o1 == 's':
        return model_parameters['dipole']
    return 0.0

def calculate_chi_tensor(atomic_coordinates, model_parameters):
    """
    Returns the chi tensor for an input list of atomic coordinate.

    Parameters
    ----------
    atomic_coordinates: str
        coordinate of atoms
    model_parameters: dict
        *

    Returns
    -------
    chi_tensor: np.array
        An array of transformation rules between atomic orbitals and multipole moments
    """
    chi_tensor = np.zeros((self.ndof, self.ndof, self.ndof))
       for p in range(self.ndof):
           for orb_q in model.orbital_types:
               q = p % model.orbitals_per_atom + model.orbital_types.index(orb_q)
               for orb_r in model.orbital_types:
                   r = p % model.orbitals_per_atom + model.orbital_types.index(orb_r)
                   chi_tensor[p, q, r] = self.chi_on_atom(self.orb(p), self.orb(q), self.orb(r), model)
    return chi_tensor

def calculate_hamiltonian_matrix(atomic_coordinates, model_parameters):
    """
    Returns the 1-body Hamiltonian matrix for an input list of atomic coordinates.

    Parameters
    ----------
    atomic_coordinates: np.array
        An array of xyz coordinates of atoms
    model_parameters: dict
        A dictionary of model parameters fit to this data using semiempirical model parameterization.

    Returns
    -------
    hamiltonian_matrix: np.array
        The 1-body Hamiltonian matrix.
    """
    ndof = len(atomic_coordinates) * orbitals_per_atom
    hamiltonian_matrix = np.zeros((ndof, ndof))
    potential_vector = calculate_potential_vector(atomic_coordinates,
                                                  model_parameters)
    for p in range(ndof):
        for q in range(ndof):
            if atom(p) != atom(q):
                r_pq = atomic_coordinates[atom(p)] - atomic_coordinates[atom(
                    q)]
                hamiltonian_matrix[p, q] = hopping_energy(
                    orb(p), orb(q), r_pq, model_parameters)
            if atom(p) == atom(q):
                if p == q and orb(p) == 's':
                    hamiltonian_matrix[p, q] += model_parameters['energy_s']
                if p == q and orb(p) in p_orbitals:
                    hamiltonian_matrix[p, q] += model_parameters['energy_p']
                for orb_r in orbital_types:
                    r = ao_index(atom(p), orb_r)
                    hamiltonian_matrix[p, q] += (
                        chi_on_atom(orb(p), orb(q), orb_r, model_parameters) *
                        potential_vector[r])
    return hamiltonian_matrix

def calculate_atomic_density_matrix(atomic_coordinates):
    """
    Returns a trial 1-electron density matrix for an input list of atomic coordinates.

    Parameters
    ----------
    atomic_coordinates: str
        coordinate of atoms

    Returns
    -------
    density_matrix: np.array
        1-electron density matrix
    """

def calculate_fock_matrix(hamiltonian_matrix, interaction_matrix,density_matrix, chi_tensor):
    """Returns the Fock matrix defined by the input Hamiltonian, interaction, & density matrices.

    Parameters
    ----------
    hamiltonian_matrix: np.array
        An array of a list of atomic coordinates.
    interaction_matrix: np.array
        An array of electron-electron interaction energies.
    density_matrix: np.array
        An array defined by the occupied orbitals, where the lowest-energy orbitals are occupied with all of the electrons available.
    chi_tensor: np.array
        An array of transformation rules between atomic orbitals and multiple moments.

    Returns
    -------
    fock_matrix: np.array
        An array consisting of a nonlinear set of equations defined by the input Hamiltonian, interaction, & density matrices.

    """
    fock_matrix = hamiltonian_matrix.copy()
    fock_matrix += 2.0 * np.einsum('pqt,rsu,tu,rs',
                                   chi_tensor,
                                   chi_tensor,
                                   interaction_matrix,
                                   density_matrix,
                                   optimize=True)
    fock_matrix -= np.einsum('rqt,psu,tu,rs',
                             chi_tensor,
                             chi_tensor,
                             interaction_matrix,
                             density_matrix,
                             optimize=True)
    return fock_matrix

def calculate_density_matrix(fock_matrix):
    """
    Returns the 1-electron density matrix defined by the input Fock matrix.

    Parameters
    ----------
    fock_matrix: np.array
        Fock matrix

    Returns
    -------
    density_matrix: np.array
       1-electron density matrix defined by the input Fock matrix 

    """
    num_occ = (ionic_charge // 2) * np.size(fock_matrix,
                                            0) // orbitals_per_atom
    orbital_energy, orbital_matrix = np.linalg.eigh(fock_matrix)
    occupied_matrix = orbital_matrix[:, :num_occ]
    density_matrix = occupied_matrix @ occupied_matrix.T
    return density_matrix

def scf_cycle(hamiltonian_matrix, interaction_matrix, density_matrix,chi_tensor, max_scf_iterations = 100,mixing_fraction = 0.25, convergence_tolerance = 1e-4):
    """
    Returns converged density & Fock matrices defined by the input Hamiltonian, interaction, & density matrices.

    Parameters
    ----------
    hamiltonian_matrix: np.array
        The 1-body Hamiltonian matrix.
    interaction_matrix: np.array
        The electron-electron interaction energy matrix.
    density_matrix: np.array
        The 1-electron density matrix.
    chi_tensor: np.array
        An array of transformation rules between atomic orbitals and multipole moments. 
    max_scf_iterations: int, default is 100
        The maximum number scf cycles.
    mixing_fraction: float, default is 0.25
        The fraction of the density matrix that will be retained for the subsequent cycle.
    convergence_tolerance: np.exp, default is 1e-4
        The maximum error tolerance beyond which the the scf calculations will sieze.

    Returns
    -------
    new_density_matrix: np.array
        The converged density matrix.
    new_fock_matrix: np.array
        The converged Fock matrix. 
    """
    old_density_matrix = density_matrix.copy()
    for iteration in range(max_scf_iterations):
        new_fock_matrix = calculate_fock_matrix(hamiltonian_matrix, interaction_matrix, old_density_matrix, chi_tensor)
        new_density_matrix = calculate_density_matrix(new_fock_matrix)

        error_norm = np.linalg.norm( old_density_matrix - new_density_matrix )
        if error_norm < convergence_tolerance:
            return new_density_matrix, new_fock_matrix

        old_density_matrix = (mixing_fraction * new_density_matrix
                              + (1.0 - mixing_fraction) * old_density_matrix)
    print("WARNING: SCF cycle didn't converge")
    return new_density_matrix, new_fock_matrix

def calculate_energy_scf(hamiltonian_matrix, fock_matrix, density_matrix):
    """Returns the Hartree-Fock total energy defined by the input Hamiltonian, Fock, & density matrices.

    Parameters
    ----------
    hamiltonian_matrix: np.array
        The 1-body Hamiltonian matrix.
    fock_matrix: np.array
        The Fock matrix.
    density_matrix: np.array
        1-electron density matrix

    Returns
    -------
    energy_scf: float
        The Hartree-Fock total energy.
    """
    energy_scf = np.einsum('pq,pq', hamiltonian_matrix + fock_matrix,
                           density_matrix)
    return energy_scf

def partition_orbitals(fock_matrix):
    """
    Returns a list with the occupied/virtual energies & orbitals defined by the input Fock matrix.

    Parameters
    ----------
    fock_matrix: np.array
        Fock matrix

    Returns
    -------
    occupied_energy: np.array
        an array of energies of occupied orbitals
    virtual_energy: np.array
        an array of energies of virtual orbitals
    occupied_matrix: np.array
        an array of occupied orbitals
    virtual_matrix: np.array
        an array of virtual orbitals

    """
    num_occ = (ionic_charge // 2) * np.size(fock_matrix,
                                            0) // orbitals_per_atom
    orbital_energy, orbital_matrix = np.linalg.eigh(fock_matrix)
    occupied_energy = orbital_energy[:num_occ]
    virtual_energy = orbital_energy[num_occ:]
    occupied_matrix = orbital_matrix[:, :num_occ]
    virtual_matrix = orbital_matrix[:, num_occ:]

    return occupied_energy, virtual_energy, occupied_matrix, virtual_matrix

def transform_interaction_tensor(occupied_matrix, virtual_matrix, interaction_matrix, chi_tensor):
    """
    Returns a transformed V tensor defined by the input occupied, virtual, & interaction matrices.

    Parameters
    ----------
    occupied_matrix: np.array
        An array of the occupied orbitals
    virtual_matrix: np.array
        An array of the occupied orbitals
    interaction_matrix: np.array
        The electron-electron interaction energy matrix
    chi_tensor: np.array
        An array of transformation rules between atomic orbitals and multipole moments


    Returns
    -------
    interaction_tensor: np.array
        An array of the electron-electron interaction energy*.
    """                           
    
    chi2_tensor = np.einsum('qa,ri,qrp',
                            virtual_matrix,
                            occupied_matrix,
                            chi_tensor,
                            optimize=True)
    interaction_tensor = np.einsum('aip,pq,bjq->aibj',
                                   chi2_tensor,
                                   interaction_matrix,
                                   chi2_tensor,
                                   optimize=True)
    return interaction_tensor

def calculate_energy_mp2(fock_matrix, interaction_matrix, chi_tensor):
    """Returns the MP2 contribution to the total energy defined by the input Fock & interaction matrices.

    Parameters
    ----------
    fock_matrix: np.array
        An array consisting of a nonlinear set of equations defined by the input Hamiltonian, interaction, & density matrices.
    interaction_matrix: np.array
        An array of electron-electron interaction energies.
    chi_tensor: np.array
        An array of transformation rules between atomic orbitals and multiple moments.

    Returns
    -------
    energy_mp2: float
        The total energy is defined by the input Fock and interaction matrices.

    """
    E_occ, E_virt, occupied_matrix, virtual_matrix = partition_orbitals(
        fock_matrix)
    V_tilde = transform_interaction_tensor(occupied_matrix, virtual_matrix,
                                           interaction_matrix, chi_tensor)

    energy_mp2 = 0.0
    num_occ = len(E_occ)
    num_virt = len(E_virt)
    for a in range(num_virt):
        for b in range(num_virt):
            for i in range(num_occ):
                for j in range(num_occ):
                    energy_mp2 -= (
                        (2.0 * V_tilde[a, i, b, j]**2 -
                         V_tilde[a, i, b, j] * V_tilde[a, j, b, i]) /
                        (E_virt[a] + E_virt[b] - E_occ[i] - E_occ[j]))
    return energy_mp2

## --------------------
## Noble Gas Parameters
## --------------------
ionic_charge = 6
orbital_types = ['s', 'px', 'py', 'pz']
orbitals_per_atom = len(orbital_types)
p_orbitals = orbital_types[1:]
vec = {'px': [1, 0, 0], 'py': [0, 1, 0], 'pz': [0, 0, 1]}
orbital_occupation = { 's':0, 'px':1, 'py':1, 'pz':1 }


if __name__ == "__main__":

    # User input
    atomic_coordinates = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 5.0]])
    # Derived from user input
    number_of_atoms = len(atomic_coordinates)

    # Argon parameters - these would change for other noble gases.
    model_parameters = {
    'r_hop' : 3.1810226927827516,
    't_ss' : 0.03365982238611262,
    't_sp' : -0.029154833035109226,
    't_pp1' : -0.0804163845390335,
    't_pp2' : -0.01393611496959445,
    'r_pseudo' : 2.60342991362958,
    'v_pseudo' : 0.022972992186364977,
    'dipole' : 2.781629275106456,
    'energy_s' : 3.1659446174413004,
    'energy_p' : -2.3926873325346554,
    'coulomb_s' : 0.3603533286088998,
    'coulomb_p' : -0.003267991835806299
    }
    
    # Start energy calculation

    # electron-electron interaction matrix
    interaction_matrix = calculate_interaction_matrix(atomic_coordinates, model_parameters)
    # chi tensor
    chi_tensor = calculate_chi_tensor(atomic_coordinates, model_parameters)

    # Initial Hamiltonian and Density matrix
    hamiltonian_matrix = calculate_hamiltonian_matrix(atomic_coordinates, model_parameters)
    density_matrix = calculate_atomic_density_matrix(atomic_coordinates)
    
    # Use density matrix to calculate Fock matrix, then use Fock matrix to calculate new density matrix??
    fock_matrix = calculate_fock_matrix(hamiltonian_matrix, interaction_matrix, density_matrix, chi_tensor)
    density_matrix = calculate_density_matrix(fock_matrix)

    # SCF Cycle
    density_matrix, fock_matrix = scf_cycle(hamiltonian_matrix, interaction_matrix, density_matrix, chi_tensor)  
    energy_ion = calculate_energy_ion(atomic_coordinates)
    energy_scf = calculate_energy_scf(hamiltonian_matrix, fock_matrix, density_matrix)

    # Hartree Fock Energy
    print(energy_scf + energy_ion)

    # MP 2 - Fock matrix from SCF
    occupied_energy, virtual_energy, occupied_matrix, virtual_matrix = partition_orbitals(fock_matrix)
    interaction_tensor = transform_interaction_tensor(occupied_matrix, virtual_matrix, interaction_matrix, chi_tensor)
    energy_mp2 = calculate_energy_mp2(fock_matrix, interaction_matrix, chi_tensor)
    print(energy_mp2)