import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import librosa
from librosa.display import specshow
import soundfile as sf
import pickle
from tqdm import tqdm
from constants import HOP_LENGTH, N_FFT

# use non-interactive backend to write to files
matplotlib.use('Agg')

# set hyperparameters
NUM_MFCC = 13

# set input and output directories
INPUT_DIR = "/scratch/rn2214/data/final_stems_22050"
OUTPUT_DIR = "/scratch/rn2214/data/mfccs"
PLOT_DIR = "/scratch/rn2214/plots/mfccs"


def extract_mfccs(in_dir, out_dir, plot_dir):
    """
    Extract Mel-Frequency Cepstral Coefficients (MFCCs) from vocal stem WAV files.
    Plot the MFCCs over time and save the plots.

    :param in_dir: (str) directory of WAV files to extract MFCCs from
    :param out_dir: (str) directory of where to save the extracted MFCCs
    :param plot_dir: (str) directory of where to save the MFCC plots
    """
    # get all of the files in the input directory
    print("Loading list of files...")
    file_list = os.listdir(in_dir)
    print(f"There are {len(file_list)} files in the input directory.")

    # create the output directory if it does not already exist
    print("Creating output directory, if it does not already exist...")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # create lists of MFCC arrays
    # each array consists of the stack means and standard deviations of the
    # MFCCs (excluding the 0th coefficient) averaged over time
    # as an example, for n_mfccs=13, the dimensions of the array would be 24 x 1
    name_list = []
    mfcc_list = []
    print("Extracting MFCCs for each audio file...")
    for i in tqdm(range(len(file_list))):
        # only process wav files
        if file_list[i].endswith(".wav"):   
            # read the soundfile
            in_path = os.path.join(in_dir, file_list[i])
            y, sr = sf.read(in_path)
            
            # set the output file name
            s = file_list[i].split("_")
            out_name = f"{s[0]}_{s[1]}_mfcc.npy"
            plot_name = f"{s[0]}_{s[1]}_mfcc.png"

            # extract MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, hop_length=HOP_LENGTH, n_mfcc=NUM_MFCC)
            
            # collapse the arrays across time
            # ignore the first coefficient
            mfcc_mean = np.mean(mfccs[1:, :], axis=1)
            mfcc_std = np.std(mfccs[1:, :], axis=1)
            mfcc_vec = np.hstack((mfcc_mean, mfcc_std))
            
            name_list.append(out_name)
            mfcc_list.append(mfcc_vec)

             # save individual vector file
            out_path = os.path.join(out_dir, out_name)
            np.save(out_path, mfcc_vec)

            # plot the mfccs
            fig, ax = plt.subplots()
            img = specshow(mfccs, x_axis='time', hop_length=HOP_LENGTH, ax=ax)
            fig.colorbar(img, ax=ax)
            ax.set(title=f'MFCCs\n{s[0]} {s[1]}')
            plot_out_path = os.path.join(plot_dir, plot_name)
            plt.savefig(plot_out_path)
            plt.close()
    
    print("Dumping all vectors and file names as a single pickle file...")
    # save all data as one object for easy loading
    data_obj = (mfcc_list, name_list)
    out_path = os.path.join(out_dir, f"all_mfccs.pkl")

    # dump pickle file
    with open(out_path, "wb") as f:
        pickle.dump(data_obj, f)
    print("Data dumped successfully!")

    print("Processing complete!")


if __name__ == '__main__':
    # run function
    extract_mfccs(INPUT_DIR, OUTPUT_DIR, PLOT_DIR)
    