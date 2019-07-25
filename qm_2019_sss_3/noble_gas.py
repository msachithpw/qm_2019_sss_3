import numpy as np
class NobleGasModel():
    def __init__(self, name, model_parameters,atomic_coordinates):
        self.name = name
        self.model_parameters = model_parameters
        self.atomic_coordinates = atomic_coordinates
        self.ionic_charge = 6
        self.orbital_types = ['s', 'px', 'py', 'pz']
        self.orbitals_per_atom = len(self.orbital_types)
        self.p_orbitals = self.orbital_types[1:]
        self.vec = {'px': [1, 0, 0], 'py': [0, 1, 0], 'pz': [0, 0, 1]}
        self.orbital_occupation = { 's':0, 'px':1, 'py':1, 'pz':1 }

    def __str__(self):
        return 'name: '+str(self.name)+'\nmodel_parameters:\n '+str(self.model_parameters)+'\natomic_coordinates:\n '+str(self.atomic_coordinates)

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
        return ao_index // self.orbitals_per_atom

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
        orb_index = ao_index % self.orbitals_per_atom
        return self.orbital_types[orb_index]

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
        p = atom_p * self.orbitals_per_atom
        p += self.orbital_types.index(orb_p)
        return p


name = 'Argon'

atomic_coordinates = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 5.0]])

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

argon = NobleGasModel(name, model_parameters,atomic_coordinates)

print(argon)
