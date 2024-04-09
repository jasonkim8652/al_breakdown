import numpy as np
from openbabel import pybel

ref_mol2_path='/home/jasonkjh/works/data/JAK2/crystal_ligand.mol2'


format_ = ref_mol2_path.split('.')[-1]
mol = list(pybel.readfile(format_,ref_mol2_path))[0]

coord_list=[list(atom.coords) for atom in mol.atoms]
coord_max = np.max(coord_list,axis=0)
coord_min = np.min(coord_list,axis=0)
center= 0.5*(coord_max+coord_min)

print(center)

