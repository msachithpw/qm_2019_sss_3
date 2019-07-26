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
    