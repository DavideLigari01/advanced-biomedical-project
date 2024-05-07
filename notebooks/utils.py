# Utility functions for the notebooks

# ----------------- Import Libraries ---------------------------

import os
import torchaudio
from tqdm.notebook import tqdm


# ----------------- Functions ---------------------------


def move_corrupted_files(src_dir, dest_dir):
    """
    Check for corrupted files in a directory. If a file is corrupted, it is moved to a new directory.
    It prints the names of corrupted files and the total count of corrupted files found in the directory.

    Args:
        dir (str): The directory path to check for corrupted files.

    Returns:
        count (int): The total count of corrupted files found in the directory.
        filenames (list): The names of corrupted files.

    Prints the names of corrupted files and the total count of corrupted files found in the directory.
    """
    count = 0
    filenames = []
    

    os.makedirs(dest_dir, exist_ok=True)
    
    for file in tqdm(os.listdir(src_dir), desc='Checking files'):
        try:
            _ = torchaudio.load(src_dir + file)
        except:
            # cut the file and paste in a new directory
            os.system(f'mv {src_dir + file} {dest_dir + file}')
            
            print(ValueError(f'File {file} is corrupted'))
            count += 1
            filenames.append(file)
            
    print(f'Files corrupted in {src_dir}:\t{count}\n')
    return count, filenames
    
    
# --------------------------------------------------------------------
# --------------------------------------------------------------------