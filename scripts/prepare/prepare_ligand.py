import os
import argparse
import warnings
from openbabel import pybel


def main(args):
	warnings.filterwarnings("error")
	cwd = os.getcwd()
	os.chdir(args.dir_)
	ligand_smi = os.path.join(
		args.id_ + '.smi'
	)
	ligand_mol2 = os.path.join(
		args.id_ + '.mol2'
	)
	ligand_prep_mol2 = os.path.join(
		args.id_ + '_prep.mol2'
	)
	ligand_pdbqt = os.path.join(
		args.id_ + '.pdbqt'
	)
	# Run Corina
	f = open(ligand_smi, 'w')
	f.write(args.smi)
	f.close()
	command = '/applic/corina/corina'
	command += ' -i t=smiles ' + ligand_smi
	command += ' -o t=mol2 ' + ligand_mol2
	os.system(command)

	# Run prep_ligand
	command = 'python /home/jasonkjh/works/projects/active_learning/prep_ligand.py'
	command += ' -l ' + ligand_mol2
	command += ' --delH --addH'
	os.system(command)
	ms = list(pybel.readfile("mol2", ligand_mol2))
	m = ms[0]
	m.write("pdbqt", ligand_pdbqt, overwrite=True)
	os.chdir(cwd)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--smi', type=str, required=True,
						help='')
	parser.add_argument('--id_', type=str, required=True,
						help='')
	parser.add_argument('--dir_', type=str, required=True,
						help='')
	parser.add_argument('--pH', type=float, default=7.0,
						help='')
	args = parser.parse_args()

	main(args)
