import os
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
from utils import normalize_data, fade_in_out


# set input and output directories
INPUT_DIR = "/scratch/rn2214/data/choruses"
OUTPUT_DIR = "/scratch/rn2214/data/standardized"


def preprocess(in_dir, out_dir, target_sr=44100, to_mono=False, normalize=False, fade=True):
    """
    Preprocess WAV files.

    :param in_dir: (str) directory of WAV files to process
    :param out_dir: (str) directory of where to save the processed WAV files
    :param target_sr: (int) sample rate to resample all WAV files to
    :param to_mono: (bool) whether to downmix stereo files to mono
    :param normalize: (bool) whether to normalize data (center, amplitude)
    """

    print("Loading list of files...")
    # get all of the files in the input directory
    file_list = os.listdir(in_dir)
    print(f"There are {len(file_list)} files in the input directory.")

    # create the output directory if it does not already exist
    print("Creating output directory, if it does not already exist...")
    os.makedirs(out_dir, exist_ok=True)

    # iterate through each file
    print("Beginning to process files...")
    print(f"Target Sampling Rate: {target_sr} Hz")

    for i in tqdm(range(len(file_list))):
        # only process .wav files
        if file_list[i].endswith(".wav"):
            # read the soundfile
            in_path = os.path.join(in_dir, file_list[i])

            # load native sampling rate
            # mix down to mono, if enabled
            # otherwise, keep in stereo
            y, sr = librosa.load(in_path, sr=None, mono=to_mono)

            # resample to target sampling rate
            y_hat = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

            if normalize:
                # normalize data to have a mean of 0, standard deviation of 1
                # scale between -1 and 1
                y_norm = normalize_data(y_hat)
            else:
                y_norm = y_hat

            if fade:
                # add a 0.5 second fade in and out to the audio
                y_out = fade_in_out(y_norm, target_sr, duration=0.5)
            else:
                y_out = y_norm

            # save file
            name, ext = file_list[i].split('.')
            out_file = name + "_Standardized." + ext
            out_path = os.path.join(out_dir, out_file)
            sf.write(out_path, y_out.T, target_sr)

    print("Processing complete!")


if __name__ == '__main__':
	# run function
	preprocess(INPUT_DIR, OUTPUT_DIR,
               target_sr=44100, to_mono=False,
               normalize=False, fade=True)
