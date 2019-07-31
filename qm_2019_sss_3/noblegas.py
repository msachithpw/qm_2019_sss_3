import numpy as np
class NobleGas():
    def __init__(self, name,atomic_coordinates):
        if isinstance(name, str):
            self.name = name
            if self.name.lower() == 'argon' or 'ar' == self.name.lower():
                self.model_parameters = {
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
            elif self.name.lower() == 'neon' or 'ne' == self.name.lower():
                self.model_parameters = {
                    'coulomb_p': -0.010255409806855187,
                    'coulomb_s': 0.4536486561938202,
                    'dipole': 1.6692376991516769,
                    'energy_p': -3.1186533988406335,
                    'energy_s': 11.334912902362603,
                    'r_hop': 2.739689713337267,
                    'r_pseudo': 1.1800779720963734,
                    't_pp1': -0.029546671673199854,
                    't_pp2': -0.0041958662271044875,
                    't_sp': 0.000450562836426027,
                    't_ss': 0.0289251941290921,
                    'v_pseudo': -0.015945813280635074
                    }
            else:
                raise NameError('Enter a different noble gas.')
        else:
            raise TypeError('Name is not a String')

        self.atomic_coordinates = atomic_coordinates
        self.ionic_charge = 6
        self.number_of_atoms = len(self.atomic_coordinates)
        self.orbital_types = ['s', 'px', 'py', 'pz']
        self.orbitals_per_atom = len(self.orbital_types)
        self.p_orbitals = self.orbital_types[1:]
        self.vec = {'px': [1, 0, 0], 'py': [0, 1, 0], 'pz': [0, 0, 1]}
        self.orbital_occupation = { 's':0, 'px':1, 'py':1, 'pz':1 }

    def __str__(self):
        return 'name: '+str(self.name)+'\nmodel_parameters:\n '+str(self.model_parameters)+'\natomic_coordinates:\n '+str(self.atomic_coordinates)+'\nionic_charge='+str(self.ionic_charge)

    def atom(self,ao_index):
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
        return ao_index // self.orbitals_per_atom

    def orb(self,ao_index):
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
        orb_index = ao_index % self.orbitals_per_atom
        return self.orbital_types[orb_index]

    def ao_index(self,atom_p, orb_p):
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
        p = atom_p * self.orbitals_per_atom
        p += self.orbital_types.index(orb_p)
        return p

class HartreeFock():
    def __init__(self,noblegas):
        # self.name = name
        # self.atomic_coordinates = atomic_coordinates
        self.noblegas = noblegas
        self.ndof = len(self.noblegas.atomic_coordinates) * self.noblegas.orbitals_per_atom
    @property
    def calculate_interaction_matrix(self):
        """
        Returns the electron-electron interaction energy matrix for an input list of atomic coordinates.

        Parameters
        ----------

        Returns
        -------
        interaction_matrix: np.array
            An array of the electron-electron interaction energy.
        """
        # ndof = len(noblegas.atomic_coordinates)*orbitals_per_atom
        interaction_matrix = np.zeros( (self.ndof,self.ndof) )
        for p in range(self.ndof):
            for q in range(self.ndof):
                if self.noblegas.atom(p) != self.noblegas.atom(q):
                    r_pq = self.noblegas.atomic_coordinates[self.noblegas.atom(p)] - self.noblegas.atomic_coordinates[self.noblegas.atom(q)]
                    interaction_matrix[p,q] = self.coulomb_energy(self.noblegas.orb(p), self.noblegas.orb(q), r_pq)
                if p == q and self.noblegas.orb(p) == 's':
                    interaction_matrix[p,q] = self.noblegas.model_parameters['coulomb_s']
                if p == q and self.noblegas.orb(p) in self.noblegas.p_orbitals:
                    interaction_matrix[p,q] = self.noblegas.model_parameters['coulomb_p']                
        return interaction_matrix
    
    def hopping_energy(self,o1, o2, r12):  
        """Returns the hopping matrix element for a pair of orbitals of type o1 & o2 separated by a vector r12.

        Parameters
        ----------
        o1 : str
            Orbital type of the 1st atom
        o2 : str
            Orbital type of the 2nd atom
        r12 : np.array
            A vector pointing from the second to the first atom 
        

        Returns
        -------
        ans : float
            The answer is the hopping energy.
        """
        r12_rescaled = r12 / self.noblegas.model_parameters['r_hop']
        r12_length = np.linalg.norm(r12_rescaled)
        ans = np.exp( 1.0 - r12_length**2 )
        if o1 == 's' and o2 == 's':
            ans *= self.noblegas.model_parameters['t_ss']
        if o1 == 's' and o2 in self.noblegas.p_orbitals:
            ans *= np.dot(self.noblegas.vec[o2], r12_rescaled) * self.noblegas.model_parameters['t_sp']
        if o2 == 's' and o1 in self.noblegas.p_orbitals:
            ans *= -np.dot(self.noblegas.vec[o1], r12_rescaled)* self.noblegas.model_parameters['t_sp']
        if o1 in self.noblegas.p_orbitals and o2 in self.noblegas.p_orbitals:
            ans *= ( (r12_length**2) * np.dot(self.noblegas.vec[o1], self.noblegas.vec[o2]) * self.noblegas.model_parameters['t_pp2']
                    - np.dot(self.noblegas.vec[o1], r12_rescaled) * np.dot(self.noblegas.vec[o2], r12_rescaled)
                    * ( self.noblegas.model_parameters['t_pp1'] + self.noblegas.model_parameters['t_pp2'] ) )
        return ans

    def coulomb_energy(self,o1, o2, r12):
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
        if o1 == 's' and o2 in self.noblegas.p_orbitals:
            ans = np.dot(self.noblegas.vec[o2], r12) / r12_length**3
        if o2 == 's' and o1 in self.noblegas.p_orbitals:
            ans = -1 * np.dot(self.noblegas.vec[o1], r12) / r12_length**3
        if o1 in self.noblegas.p_orbitals and o2 in self.noblegas.p_orbitals:
            ans = (
                np.dot(self.noblegas.vec[o1], self.noblegas.vec[o2]) / r12_length**3 -
                3.0 * np.dot(self.noblegas.vec[o1], r12) * np.dot(self.noblegas.vec[o2], r12) / r12_length**5)
        return ans

    def pseudopotential_energy(self,o, r):
        """
        Returns the energy of a pseudopotential between a multipole of type o and an atom separated by a vector r.

        Parameters
        ----------
        o: str
            orbital type
        r: np.array
            pseudo_vector*

        Returns
        -------
        ans: float
            pseudopotential between a multipole of type o and an atom
        """
        ans = self.noblegas.model_parameters['v_pseudo']
        r_rescaled = r / self.noblegas.model_parameters['r_pseudo']
        ans *= np.exp(1.0 - np.dot(r_rescaled, r_rescaled))
        if o in self.noblegas.p_orbitals:
            ans *= -2.0 * np.dot(self.noblegas.vec[o], r_rescaled)
        return ans

    @property
    def calculate_energy_ion(self):
        """
        Returns the ionic contribution to the total energy for an input list of atomic coordinates.

        Parameters
        ----------

        Returns
        -------
        energy_ion: float
            Ionic energy of the system.
        """
        energy_ion = 0.0
        for i, r_i in enumerate(self.noblegas.atomic_coordinates):
            for j, r_j in enumerate(self.noblegas.atomic_coordinates):
                if i < j:
                    energy_ion += (self.noblegas.ionic_charge**2) * self.coulomb_energy(
                        's', 's', r_i - r_j)
        return energy_ion

    @property
    def calculate_potential_vector(self):
        """Returns the electron-ion potential energy vector for an input list of atomic coordinates.

        Parameters
        ----------

        Returns
        -------
        potential_vector: np.array
        An array of electron-ion potential energy vectors for an input list of atomic coordinates.
        """
        ndof = len(self.noblegas.atomic_coordinates) * self.noblegas.orbitals_per_atom
        potential_vector = np.zeros(self.ndof)
        for p in range(self.ndof):
            potential_vector[p] = 0.0
            for atom_i, r_i in enumerate(self.noblegas.atomic_coordinates):
                r_pi = self.noblegas.atomic_coordinates[self.noblegas.atom(p)] - r_i
                if atom_i != self.noblegas.atom(p):
                    potential_vector[p] += (
                        self.pseudopotential_energy(self.noblegas.orb(p), r_pi) -
                        self.noblegas.ionic_charge * self.coulomb_energy(self.noblegas.orb(p), 's', r_pi))
        return potential_vector

    def chi_on_atom(self,o1, o2, o3):
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

        Returns
        -------
        model_parameters['dipole'] or constant: float
            The value of the chi tensor.
        """
        if o1 == o2 and o3 == 's':
            return 1.0
        if o1 == o3 and o3 in self.noblegas.p_orbitals and o2 == 's':
            return self.noblegas.model_parameters['dipole']
        if o2 == o3 and o3 in self.noblegas.p_orbitals and o1 == 's':
            return self.noblegas.model_parameters['dipole']
        return 0.0

    @property
    def calculate_chi_tensor(self):
        """
        Returns the chi tensor for an input list of atomic coordinate.

        Parameters
        ----------

        Returns
        -------
        chi_tensor: np.array
            An array of transformation rules between atomic orbitals and multipole moments
        """
        chi_tensor = np.zeros((self.ndof, self.ndof, self.ndof))
        for p in range(self.ndof):
            for orb_q in self.noblegas.orbital_types:
                q = self.noblegas.ao_index(self.noblegas.atom(p),orb_q)
                # q = p % self.noblegas.orbitals_per_atom + self.noblegas.orbital_types.index(orb_q)
                for orb_r in self.noblegas.orbital_types:
                    # r = p % self.noblegas.orbitals_per_atom + self.noblegas.orbital_types.index(orb_r)
                    r = self.noblegas.ao_index(self.noblegas.atom(p),orb_r)
                    chi_tensor[p, q, r] = self.chi_on_atom(self.noblegas.orb(p), self.noblegas.orb(q), self.noblegas.orb(r))
        return chi_tensor

    @property
    def calculate_hamiltonian_matrix(self):
        """
        Returns the 1-body Hamiltonian matrix for an input list of atomic coordinates.

        Parameters
        ----------

        Returns
        -------
        hamiltonian_matrix: np.array
            The 1-body Hamiltonian matrix.
        """
        # ndof = len(noblegas.atomic_coordinates) * noblegas.orbitals_per_atom
        hamiltonian_matrix = np.zeros([self.ndof, self.ndof])
        potential_vector = self.calculate_potential_vector
        for p in range(self.ndof):
            for q in range(self.ndof):
                if self.noblegas.atom(p) != self.noblegas.atom(q):
                    r_pq = self.noblegas.atomic_coordinates[self.noblegas.atom(p)] - self.noblegas.atomic_coordinates[self.noblegas.atom(
                        q)]
                    hamiltonian_matrix[p, q] = self.hopping_energy(
                        self.noblegas.orb(p), self.noblegas.orb(q), r_pq)
                if self.noblegas.atom(p) == self.noblegas.atom(q):
                    if p == q and self.noblegas.orb(p) == 's':
                        hamiltonian_matrix[p, q] += self.noblegas.model_parameters['energy_s']
                    if p == q and self.noblegas.orb(p) in self.noblegas.p_orbitals:
                        hamiltonian_matrix[p, q] += self.noblegas.model_parameters['energy_p']
                    for orb_r in self.noblegas.orbital_types:
                        r = self.noblegas.ao_index(self.noblegas.atom(p), orb_r)
                        hamiltonian_matrix[p, q] += (
                            self.chi_on_atom(self.noblegas.orb(p), self.noblegas.orb(q), orb_r) *
                            potential_vector[r])
        return hamiltonian_matrix

    @property
    def calculate_atomic_density_matrix(self):
        """
        Returns a trial 1-electron density matrix for an input list of atomic coordinates.

        Parameters
        ----------

        Returns
        -------
        density_matrix: np.array
            1-electron density matrix
        """
        # print(self.ndof)
        density_matrix = np.zeros([self.ndof, self.ndof])
        print(range(self.ndof))
        for p in range(self.ndof):
            density_matrix[p, p] = self.noblegas.orbital_occupation[self.noblegas.orb(p)]
            print(F"({p},{p}): {self.noblegas.orb(p)}, {self.noblegas.orbital_occupation[self.noblegas.orb(p)]}")
        return density_matrix

    def calculate_fock_matrix(self,density_matrix):
        """Returns the Fock matrix defined by the input Hamiltonian, interaction, & density matrices.

        Parameters
        ----------
        density_matrix: np.array
            An array defined by the occupied orbitals, where the lowest-energy orbitals are occupied with all of the electrons available.

        Returns
        -------
        fock_matrix: np.array
            An array consisting of a nonlinear set of equations defined by the input Hamiltonian, interaction, & density matrices.

        """
        
        fock_matrix = self.calculate_hamiltonian_matrix.copy()
        # print("interaction: \n",self.calculate_interaction_matrix)
        fock_matrix += 2.0 * np.einsum('pqt,rsu,tu,rs',
                                    self.calculate_chi_tensor,
                                    self.calculate_chi_tensor,
                                    self.calculate_interaction_matrix,
                                    density_matrix,
                                    optimize=True)
        
        fock_matrix -= np.einsum('rqt,psu,tu,rs',
                                self.calculate_chi_tensor,
                                self.calculate_chi_tensor,
                                self.calculate_interaction_matrix,
                                density_matrix,
                                optimize=True)
        return fock_matrix
    
    def calculate_density_matrix(self,fock_matrix):
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
        num_occ = (self.noblegas.ionic_charge // 2) * np.size(fock_matrix,0) // self.noblegas.orbitals_per_atom
        # print('no of occupied:\n',num_occ)
        orbital_energy, orbital_matrix = np.linalg.eigh(fock_matrix)
        occupied_matrix = orbital_matrix[:, :num_occ]
        # print('occ=\n',occupied_matrix)
        density_matrix = occupied_matrix @ occupied_matrix.T
        print(density_matrix)
        return density_matrix

    def calculate_hartree_fock_energy(self, fock_matrix, density_matrix):
        """
        Calculates the Hartree Fock Energy.

        Parameters
        ----------
        fock_matrix: np.array
            An array consisting of a nonlinear set of equations defined by the input Hamiltonian, interaction, & density matrices.

        density_matrix: np.array
            An array defined by the occupied orbitals, where the lowest-energy orbitals are occupied with all of the electrons available. 

        Returns
        -------
        energy_hf: float
            Hartree Fock energy which is the sum of the ionic energy nad the scf energy
        """   
        energy_ionic = self.calculate_energy_ion
        energy_scf = np.einsum('pq,pq', self.calculate_hamiltonian_matrix + fock_matrix, density_matrix)
        energy_hf = energy_ionic + energy_scf
        print(energy_hf)
        return energy_hf

    def scf_cycle(self, max_scf_iterations = 100,mixing_fraction = 0.25, convergence_tolerance = 1e-4):
        """
        Returns converged density & Fock matrices defined by the input Hamiltonian, interaction, & density matrices.

        Parameters
        ----------
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
        initial_guess_density = self.calculate_atomic_density_matrix.copy()
        initial_guess_fock = self.calculate_fock_matrix(initial_guess_density)
        old_density_matrix = self.calculate_density_matrix(initial_guess_fock)
        # print(old_density_matrix)
        for iteration in range(max_scf_iterations):
            new_fock_matrix = self.calculate_fock_matrix(old_density_matrix)
            new_density_matrix = self.calculate_density_matrix(new_fock_matrix)

            error_norm = np.linalg.norm( old_density_matrix - new_density_matrix )
            # print(iteration)
            if error_norm < convergence_tolerance:
                # print('den:\n',new_density_matrix)
                # print('fock:\n',new_fock_matrix)
                hf_energy = self.calculate_hartree_fock_energy(new_fock_matrix,new_density_matrix)
                return new_density_matrix, new_fock_matrix, hf_energy

            old_density_matrix = (mixing_fraction * new_density_matrix
                                + (1.0 - mixing_fraction) * old_density_matrix)
            hf_energy = self.calculate_hartree_fock_energy(new_fock_matrix,new_density_matrix)
        print("WARNING: SCF cycle didn't converge")

        return new_density_matrix, new_fock_matrix, hf_energy

class Mp2(HartreeFock):
    def __init__(self,fock_matrix,noblegas):
        self.fock_matrix = fock_matrix
        self.noblegas = noblegas
        super().__init__(self.noblegas)

    def partition_orbitals(self):
        """
        Returns a list with the occupied/virtual energies & orbitals defined by the input Fock matrix.

        Parameters
        ----------

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
        num_occ = (self.noblegas.ionic_charge // 2) * np.size(self.fock_matrix,
                                                0) // self.noblegas.orbitals_per_atom
        orbital_energy, orbital_matrix = np.linalg.eigh(self.fock_matrix)
        occupied_energy = orbital_energy[:num_occ]
        virtual_energy = orbital_energy[num_occ:]
        occupied_matrix = orbital_matrix[:, :num_occ]
        virtual_matrix = orbital_matrix[:, num_occ:]

        return occupied_energy, virtual_energy, occupied_matrix, virtual_matrix

    def transform_interaction_tensor(self,occupied_matrix,virtual_matrix,interaction_matrix, chi_tensor):
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
        
        # chi2_tensor = np.einsum('qa,ri,qrp',virtual_matrix,occupied_matrix,chi_tensor,optimize=True)
        chi2_tensor = np.einsum('qa,ri,qrp', virtual_matrix, occupied_matrix, chi_tensor, optimize=True)
        # interaction_tensor = np.einsum('aip,pq,bjq->aibj',chi2_tensor,interaction_matrix,chi2_tensor,optimize=True)
        interaction_tensor = np.einsum('aip,pq,bjq->aibj', chi2_tensor, interaction_matrix, chi2_tensor, optimize=True)
        # print(interaction_tensor)
        return interaction_tensor

    def calculate_energy_mp2(self):
        """Returns the MP2 contribution to the total energy defined by the input Fock & interaction matrices.

        Parameters
        ----------

        Returns
        -------
        energy_mp2: float
            The total energy is defined by the input Fock and interaction matrices.

        """
        E_occ, E_virt, occupied_matrix, virtual_matrix = self.partition_orbitals()
        V_tilde = self.transform_interaction_tensor(occupied_matrix, virtual_matrix,
                                            super().calculate_interaction_matrix, super().calculate_chi_tensor)
        energy_mp2 = 0.0
        num_occ = len(E_occ)
        num_virt = len(E_virt)
        for a in range(num_virt):
            for b in range(num_virt):
                for i in range(num_occ):
                    for j in range(num_occ):
                        energy_mp2 -= ((2.0 * V_tilde[a, i, b, j]**2 - V_tilde[a, i, b, j] * V_tilde[a, j, b, i]) / (E_virt[a] + E_virt[b] - E_occ[i] - E_occ[j]))
        return energy_mp2

