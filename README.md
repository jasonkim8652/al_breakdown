# Understanding active learning of molecular docking and its applications
This repository consists of following two parts:

- Running active learning pipeline for ultra-large-scale docking
- Anlyzing why active learning was so effective

Technical details and thorough analysis can be found in our paper, [Understanding active learning of molecular docking and its applications](https://arxiv.org/abs/2406.12919), written by Jeonghyeon Kim, Juno Nam and Seogok Ryu. If you have any question, feel free to open an issue or reach out at jasonkjh@snu.ac.kr.

## Installation

``` bash
conda create -n activelearning python=3.10 numpy scipy matplotlib pandas scikit-learn pytorch pytorch-cuda=11.7 cuda=11.7 dgl parmap openbabel rdkit -c pytorch -c dglteam/label/cu117 -c nvidia -c conda-forge --override-channels
conda install conda-forge::plip
conda activate activelearning
```

## Running Active Learning

 `edit_dataset.py` is to acquire the molecules to dock following acquisition function. `train_map.py` is to train your model on the acquired molecules before. `inference_map.py` is to inference on remaining dataset.
If you use slurm, `submit_active_learning.py` would write down whole active learning pipeline script. Following is the example.
``` bash
python submit_active_learning.py --title Test --csv_path Enamine_HTS.csv --num_iter 10 
```

Before you start training, you need to dock acquired molecules in to prepared receptor. All the scripts need for docking is in `scripts/prepare` and `scripts/docking`.

## Analysis in paper

 All the analysis performed in our paper is in `scripts/analysis`. They are seperated into the figure number wrote down in our paper. 

- `fig2-6`: Analysis about model's RMSE, $R^2$, Success rate, and ordering. 
- `fig7`: Analysis about 3D pose similarity of docked molecules into each receptor.
- `fig9`: Analysis about interaction pattern of top scored compounds.  
- `tab3`: Linear factor analysis between number of functional group and docking score. 
- `fig11`: Calculating AUROC of DUD-E active and decoy set when using surrogate model for virtual screening
- `fig12`: Compare ability to acquire higher docking score between fingerprint screening and our model's inference. Chemical space visualization using t-sne also. 

## License

## Citation 
