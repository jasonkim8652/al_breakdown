import os
import random
import argparse
import pandas as pd


def edit_initial_dataset(
		csv_path,
		title,
		seed,
		num_train,
		num_valid,
		num_test,
	):
	df = pd.read_csv(args.csv_path)
	condition = (df['Dock'] < 0.0)
	df_ = df[condition]

	random.seed(seed)
	num_ = len(df_)
	idx_list = [i for i in range(num_)]
	random.shuffle(idx_list)

	idx_valid = idx_list[:num_valid]
	idx_test = idx_list[num_valid:num_valid+num_test]
	idx_train = idx_list[num_valid+num_test:num_valid+num_test+num_train]
	idx_remain = idx_list[num_valid+num_test+num_train:]

	df_train = df_.iloc[idx_train]
	df_valid = df_.iloc[idx_valid]
	df_test = df_.iloc[idx_test]
	df_remain = df_.iloc[idx_remain]

	train_path = os.path.join(
		'/home/jasonkjh/works/projects/active_learning',
		'data',
		title + '_seed'+str(seed)+'_step0_train.csv'
	)
	df_train.to_csv(train_path, index=False)

	valid_path = os.path.join(
		'/home/jasonkjh/works/projects/active_learning',
		'data',
		title + '_seed'+str(seed)+'_step0_valid.csv'
	)
	df_valid.to_csv(valid_path, index=False)

	test_path = os.path.join(
		'/home/jasonkjh/works/projects/active_learning',
		'data',
		title + '_seed'+str(seed)+'_step0_test.csv'
	)
	df_test.to_csv(test_path, index=False)

	remain_path = os.path.join(
		'/home/jasonkjh/works/projects/active_learning',
		'data',
		title + '_seed'+str(seed)+'_step0_remain.csv'
	)
	df_remain.to_csv(remain_path, index=False)


def edit_active_learning_dataet(
		title,
		seed,
		step,
		num_train,
		method,
	):
	if step == 1:
		train_path = os.path.join(
			'/home/jasonkjh/works/projects/active_learning',
			'data',
			title + '_seed'+str(seed)+'_step'+str(step-1)+'_train.csv'
		)
		remain_path = os.path.join(
			'/home/jasonkjh/works/projects/active_learning',
			'data',
			title + '_seed'+str(seed)+'_step'+str(step-1)+'_remain.csv'
		)
	else:
		train_path = os.path.join(
			'/home/jasonkjh/works/projects/active_learning',
			'data',
			title + '_seed'+str(seed)+'_step'+str(step-1)+'_'+method+'_train.csv'
		)
		remain_path = os.path.join(
			'/home/jasonkjh/works/projects/active_learning',
			'data',
			title + '_seed'+str(seed)+'_step'+str(step-1)+'_'+method+'_remain.csv'
		)
	df_train = pd.read_csv(train_path)
	df_remain = pd.read_csv(remain_path)
	pred_list = list(df_remain['Pred'])
	unc_list = list(df_remain['Unc'])
	ucb_list = [pred_list[i] - 2.0*unc_list[i] for i in range(len(df_remain))]
	df_remain['Ucb'] = ucb_list

	if method == 'greedy':
		df_remain = df_remain.sort_values(by=['Pred'])
	elif method == 'ucb':
		df_remain = df_remain.sort_values(by=['Ucb'])
	elif method == 'unc':
		df_remain = df_remain.sort_values(by=['Unc'], ascending=False)
	
	df_sampled = df_remain[:num_train]
	df_sampled = df_sampled[['ID','SMILES','Dock']]
	df_train = pd.concat([df_train, df_sampled])

	df_remain = df_remain[num_train:]
	df_remain = df_remain[['ID','SMILES','Dock']]

	train_path = os.path.join(
		'/home/jasonkjh/works/projects/active_learning',
		'data',
		title + '_seed'+str(seed)+'_step'+str(step)+'_'+method+'_train.csv'
	)
	df_train.to_csv(train_path, index=False)
	remain_path = os.path.join(
		'/home/jasonkjh/works/projects/active_learning',
		'data',
		title + '_seed'+str(seed)+'_step'+str(step)+'_'+method+'_remain.csv'
	)
	df_remain.to_csv(remain_path, index=False)
	

def main(args):
	if args.step == 0:
		edit_initial_dataset(
			csv_path=args.csv_path,
			title=args.title,
			seed=args.seed,
			num_train=args.num_train,
			num_valid=args.num_valid,
			num_test=args.num_test,
		)
	else:
		edit_active_learning_dataet(
			title=args.title,
			seed=args.seed,
			step=args.step,
			num_train=args.num_train,
			method=args.method,
		)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--title', type=str, required=True, 
						help='')
	parser.add_argument('--csv_path', type=str, required=True, 
						help='')
	parser.add_argument('--seed', type=int, required=True, 
						help='Seed used for dataset splitting')
	parser.add_argument('--step', type=int, required=True, 
						help='')
	parser.add_argument('--method', type=str, required=True, 
						help='')

	parser.add_argument('--num_train', type=int, default=10000, 
						help='')
	parser.add_argument('--num_valid', type=int, default=10000, 
						help='')
	parser.add_argument('--num_test', type=int, default=10000, 
						help='')
	args = parser.parse_args()
	main(args)
