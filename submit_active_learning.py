import os
import argparse

SEED_LIST = [
	0,1,2,3
]

METHOD_LIST = [
	'greedy',
	'ucb',
	'unc',
]


def run_active_learning(
		title,
		csv_path,
		num_iter,
		seed,
	):
	cwd = os.getcwd()
	dir_ = os.path.join(
		cwd,
		'slurm',
	)

	script_path = 'submit_active_learning'
	script_path += '_' + title
	script_path += '_' + str(seed)
	script_path += '.sh'
	script_path = os.path.join(
		dir_,
		script_path
	)
	f = open(script_path, 'w')
	f.write('#!/bin/sh' + '\n')
	f.write('#SBATCH -J AL_' + title + '_' + str(seed) + '\n')
	f.write('#SBATCH -p gpu.q' + '\n')
	f.write('#SBATCH --gres=gpu:1'+'\n')
	f.write('#SBATCH --exclude=nova[005,008-010]'+'\n')
	f.write('#SBATCH -N 1' + '\n')
	f.write('#SBATCH -n 1' + '\n'+'#SBATCH -c 6' + '\n')
	f.write('#SBATCH -o '+title+'_al'+str(seed)+'.out\n')
	f.write('#SBATCH -e '+title+'_al'+str(seed)+'.err\n')
	f.write('\n')

	step = 0
	method = 'initial'
	
    command = 'STARTTIME=$(date +%s)'+'\n'
	f.write(command)
	
	# 1. edit datasets
	command = 'python -u edit_dataset.py'
	command += ' --title ' + title
	command += ' --csv_path ' + csv_path
	command += ' --seed ' + str(seed)
	command += ' --method ' + str(method)
	command += ' --step ' + str(step) + '\n'
	f.write(command)
	command = 'ENDTIME=$(date +%s)'+'\n'
	f.write(command)
	command = 'echo "It takes $(($ENDTIME - $STARTTIME)) seconds to edit datasets..."'+'\n'
	f.write(command)
    
	command = 'STARTTIME=$(date +%s)'+'\n'
	f.write(command)
	
	# 2. train on the train/valid/test set
	log_path = 'logs/'
	log_path += title + '_' + str(step) + '_' + str(seed) + '_' + method + '.log' 
	command = 'python -u train_map.py'
	command += ' --title ' + title
	command += ' --step ' + str(step)
	command += ' --seed ' + str(seed)
	command += ' --method ' + str(method) + '\n'
	f.write(command)
	command = 'ENDTIME=$(date +%s)'+'\n'
	f.write(command)
	command = 'echo "It takes $(($ENDTIME - $STARTTIME)) seconds to train..."'+'\n'
	f.write(command)
	
	command = 'STARTTIME=$(date +%s)'+'\n'
	f.write(command)
	# 3. inference on the remained set
	command = 'python -u inference_map.py'
	command += ' --title ' + title
	command += ' --step ' + str(step)
	command += ' --seed ' + str(seed)
	command += ' --method ' + str(method) + '\n' 
	f.write(command)
	command = 'ENDTIME=$(date +%s)'+'\n'
	f.write(command)
	command = 'echo "It takes $(($ENDTIME - $STARTTIME)) seconds to inference..."'+'\n'
	f.write(command)
	f.write('\n')
    
	for method in METHOD_LIST:
		for step in range(1, num_iter):
			
			# 1. edit datasets
			command = 'STARTTIME=$(date +%s)'+'\n'
			f.write(command)
			command = 'python -u edit_dataset.py'
			command += ' --title ' + title
			command += ' --csv_path ' + csv_path
			command += ' --step ' + str(step)
			command += ' --method ' + str(method)
			command += ' --seed ' + str(seed) + '\n'
			f.write(command)
			command = 'ENDTIME=$(date +%s)'+'\n'
			f.write(command)
			command = 'echo "It takes $(($ENDTIME - $STARTTIME)) seconds to edit dataset..."'+'\n'
			f.write(command)
			

			# 2. train on the train/valid/test set
			command = 'STARTTIME=$(date +%s)'+'\n'
			f.write(command)
			log_path = 'logs/'
			log_path += title + '_' + str(step) + '_' + str(seed) + '_' + method + '.log' 
			command = 'python -u train_map.py'
			command += ' --title ' + title
			command += ' --step ' + str(step)
			command += ' --seed ' + str(seed)
			command += ' --method ' + str(method) + '\n'
			f.write(command)
			command = 'ENDTIME=$(date +%s)'+'\n'
			f.write(command)
			command = 'echo "It takes $(($ENDTIME - $STARTTIME)) seconds to train..."'+'\n'
			f.write(command)
			


			# 3. inference on the remained set
			command = 'STARTTIME=$(date +%s)'+'\n'
			f.write(command)
			command = 'python -u inference_map.py'
			command += ' --title ' + title
			command += ' --step ' + str(step)
			command += ' --seed ' + str(seed)
			command += ' --method ' + str(method) + '\n'
			f.write(command)
			command = 'ENDTIME=$(date +%s)'+'\n'
			f.write(command)
			command = 'echo "It takes $(($ENDTIME - $STARTTIME)) seconds to inference..."'+'\n'
			f.write(command)

			f.write('\n')
			
	f.close()

	os.chdir(dir_)
	#os.system('sbatch ' + script_path)
	os.chdir(cwd)

def main(args):
	for i, seed in enumerate(SEED_LIST):
		run_active_learning(
			title=args.title,
			csv_path=args.csv_path,
			num_iter=args.num_iter,
			seed=seed,
		)
	


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--title', type=str, required=True, 
						help='')
	parser.add_argument('--csv_path', type=str, required=True, 
						help='')
	parser.add_argument('--num_iter', type=int, default=10, 
						help='')
	args = parser.parse_args()

	main(args)
