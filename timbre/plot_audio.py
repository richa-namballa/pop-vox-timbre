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


# set input and output directories
INPUT_DIR = "/scratch/rn2214/data/final_stems_22050"
PLOT_DIR = "/scratch/rn2214/plots/spectrograms"

def plot_audio(in_dir, plot_dir):
    """
    Plot and save waveforms of vocal stem WAV files.

    :param in_dir: (str) directory of WAV files to plot waveforms of
    :param plot_dir: (str) directory of where to save the waveform plots
    """
    # get all of the files in the input directory
    print("Loading list of files...")
    file_list = os.listdir(in_dir)
    print(f"There are {len(file_list)} files in the input directory.")

    # create the output directory if it does not already exist
    print("Creating plot directory, if it does not already exist...")
    os.makedirs(plot_dir, exist_ok=True)
    
    print("Plotting waveforms for each audio file...")
    for i in tqdm(range(len(file_list))):
        # only process wav files
        if file_list[i].endswith(".wav"):   
            # read the soundfile
            in_path = os.path.join(in_dir, file_list[i])
            y, sr = sf.read(in_path)
            
            # set the output file name
            s = file_list[i].split("_")
            plot_name = f"{s[0]}_{s[1]}_waveform.png"
            plot_out_path = os.path.join(plot_dir, plot_name)

            # plot the mfccs
            duration = len(y) / sr
            t = np.linspace(0, duration, num=len(y))
            plt.plot(t, y)
            plt.xlabel("Time (seconds)")
            plt.ylabel("Amplitude")
            plt.title(f"{s[0]} {s[1]} Audio")
            plt.savefig(plot_out_path)
            plt.close()

    print("Processing complete!")


if __name__ == '__main__':
    # run function
    plot_audio(INPUT_DIR, PLOT_DIR)
    