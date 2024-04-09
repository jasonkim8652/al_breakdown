import os

from openbabel import pybel
from plip.structure.preparation import PDBComplex 

from plip.basic import config


def extract_interaction(
		complex_pdb,
		ligand_id='UNK'
	):
	mol = PDBComplex()
	mol.load_pdb(complex_pdb)
	mol.analyze()
	
	interactions = mol.interaction_sets
	key_list = list(interactions.keys())
	for key in key_list:
		if ligand_id in key:
			interaction = interactions[key]
			res_list = []
			for content in interaction.hbonds_ldon:
				res_ = str(content.resnr)
				if res_ not in res_list:
						res_list.append(res_)
			for content in interaction.hbonds_pdon:
				res_ = str(content.resnr)
				if res_ not in res_list:
					res_list.append(res_)
			res_list = sorted(res_list)
			res_hbonds = ':'.join(res_list)
			n_hbonds = len(res_list)
			res_list = []
			for content in interaction.pication_laro:
				res_ = str(content.resnr)
				if res_ not in res_list:
					res_list.append(res_)
			for content in interaction.pication_paro:
				res_ = str(content.resnr)
				if res_ not in res_list:
					res_list.append(res_)
			res_list = sorted(res_list)
			res_pication = ':'.join(res_list)
			n_pication = len(res_list)
			res_list = []
			for content in interaction.pistacking:
				res_ = str(content.resnr)
				if res_ not in res_list:
					res_list.append(res_)
			res_list = sorted(res_list)
			res_pistack = ':'.join(res_list)
			n_pistack = len(res_list)
			res_list = []
			for content in interaction.hydrophobic_contacts:
				res_ = str(content.resnr)
				if res_ not in res_list:
					res_list.append(res_)
			res_list = sorted(res_list)
			res_hydrophobic = ':'.join(res_list)
			n_hydrophobic = len(res_list)
			res_list = []
			for content in interaction.halogen_bonds:
				res_ = str(content.resnr)
				if res_ not in res_list:
					res_list.append(res_)
			res_list = sorted(res_list)
			res_halogen = ':'.join(res_list)
			n_halogen = len(res_list)
			output = [
				n_hbonds,
				n_pication,
				n_pistack,
				n_hydrophobic,
				n_halogen,
				res_hbonds,
				res_pication,
				res_pistack,
				res_hydrophobic,
				res_halogen,
			]
			return output



def prepare_complex(
			id_,
			job_dir,
			ligand_path,
			receptor_path,
		):
	ligand_pdb = os.path.join(job_dir, id_+'.pdb')
	complex_pdb = os.path.join(job_dir, id_+'_complex.pdb')

	format_ = ligand_path.split('.')[-1]
	m = list(pybel.readfile(format_, ligand_path))[0]
	m.write('pdb', ligand_pdb, overwrite=True)

	f_ligand = open(ligand_pdb, 'r')
	ligand_lines = f_ligand.readlines()
	f_ligand.close()

	f_receptor = open(receptor_path, 'r')
	receptor_lines = f_receptor.readlines()
	f_receptor.close()

	f = open(complex_pdb, 'w')
	for line in receptor_lines:
		if line.startswith('ATOM'):
			f.write(line)
	f.write('TER\n')
	for line in ligand_lines[2:]:
		f.write(line)
	f.write('END\n')
	f.close()


def flush(id_, job_dir):
	ligand_pdb = os.path.join(job_dir, id_+'.pdb')
	if os.path.exists(ligand_pdb):
		command = 'rm ' + ligand_pdb
		os.system(command)

	complex_pdb = os.path.join(job_dir, id_+'_complex.pdb')
	if os.path.exists(complex_pdb):
		command = 'rm ' + complex_pdb
		os.system(command)
