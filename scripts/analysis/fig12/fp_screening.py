from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pandas as pd
import sys

def generate_fingerprints(smiles_list):
    """
    Generate Morgan fingerprints for a list of SMILES.
    """
    mols = (Chem.MolFromSmiles(smiles) for smiles in smiles_list)
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in mols if mol]
    return fps

def compute_similarity(args):
    """
    Compute similarity for a chunk of library molecules against reference molecules.
    """
    lib_fp, reference_fps, threshold = args
    return any(DataStructs.TanimotoSimilarity(lib_fp, ref_fp) >= threshold for ref_fp in reference_fps)

def process_chunk(chunk_smiles, reference_fps, threshold=0.85):
    """
    Process a chunk of library molecules.
    """
    chunk_fps = generate_fingerprints(chunk_smiles)
    with Pool(cpu_count()) as pool:
        args = [(lib_fp, reference_fps, threshold) for lib_fp in chunk_fps]
        results = pool.map(compute_similarity, args)
    return [smiles for smiles, is_similar in zip(chunk_smiles, results) if is_similar]

def screen_library(reference_smiles, threshold=0.85, chunk_size=200000):
    """
    Screen a chemical library using fingerprint similarity against reference molecules.
    """
    reference_fps = generate_fingerprints(reference_smiles)
    selected_molecules = []

    library_path = "/home/jasonkjh/works/data/Real/REAL_AL.csv"

    # Get the total number of chunks for the progress bar
    total_chunks = sum(1 for _ in pd.read_csv(library_path, usecols=["SMILES"], chunksize=chunk_size))

    # Use pandas to read in chunks with a progress bar
    for chunk in tqdm(pd.read_csv(library_path, usecols=["SMILES"], chunksize=chunk_size), total=total_chunks, desc="Processing chunks"):
        chunk_smiles = chunk["SMILES"].tolist()
        selected_molecules.extend(process_chunk(chunk_smiles, reference_fps, threshold))
    
    return selected_molecules


# You would call the function like this:
# library_path = "/path/to/your/library/files/"
# num_files = 50  # or whatever the number of files you have
# selected_molecules = screen_library(library_path, num_files, reference_smiles_list)


title = sys.argv[1]
reference_path = sys.argv[2]

reference_df = pd.read_csv(reference_path)
reference_smiles_list = reference_df['SMILES'].tolist()
'''
library_num = 3
library_dict = {"20":43, "21-22": 57, "23":50, "24":76, "25_1":50, "25_2": 50, "26_1": 50, "26_2": 50, "27": 57}
#library_path = "/home/share/Enamine_Real/"+library_num+"/Scaffold_"

def read_smiles_in_chunks(file_path, chunk_size=100000):
    """
    Reads SMILES from a CSV in chunks for memory efficiency.
    """
    smiles_list = []
    chunk_iter = pd.read_csv(file_path, usecols=["SMILES"], chunksize=chunk_size)
    
    for chunk in tqdm(chunk_iter, desc=f"Reading {file_path}"):
        smiles_list.extend(chunk["SMILES"].tolist())
        
    return smiles_list
'''
selected_molecules = screen_library(reference_smiles_list)

df = pd.DataFrame(selected_molecules, columns=["SMILES"])
df.to_csv(title+"_selected_molecules.csv", index=False)
