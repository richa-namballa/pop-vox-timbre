import os
import torch
import numpy as np
import soundfile as sf
from demucs import pretrained
from demucs.apply import apply_model

# set input and output directories
INPUT_DIR = "/scratch/rn2214/data/standardized"
OUTPUT_DIR = "/scratch/rn2214/data/separated"


def separate(in_dir, out_dir, model_name='htdemucs', gpu=True):
    """
    Run Demucs source separation model on WAV files to isolate vocal stems.

    :param in_dir: (str) directory of WAV files to run source separation model on
    :param out_dir: (str) directory of where to save the separated vocal stems
    :param model_name: (str) name of the pretrained demucs model to use,
                        default model is the hybrid transformer demucs
    :param gpu: (bool) if a gpu is available for use, set to True
    """
    # get all of the files in the input directory
    print("Loading list of files...")
    file_list = os.listdir(in_dir)
    print(f"There are {len(file_list)} files in the input directory.")

    # create the output directory if it does not already exist
    print("Creating output directory, if it does not already exist...")
    os.makedirs(out_dir, exist_ok=True)

    # load the model
    print("Loading pretrained model...")
    model = pretrained.get_model(model_name)
    print("Model loaded successfully.")

    # iterate through each file
    print("Beginning to process files...")
    for file in file_list:
        # only process wav files
        if file.endswith(".wav"):
            # read the soundfile
            in_path = os.path.join(in_dir, file)
            y, sr = sf.read(in_path)

            # check if audio is in mono
            if len(y.shape) == 1:
                # if the audio is in mono,
                # duplicate channels to create a stereo track
                # demucs network expects two channels of audio
                y = np.vstack([y, y])

            # get dimensions
            num_samples, num_channels = y.shape

            # convert to 3 dimensional tensor (1, num_channels, num_samples)
            x = torch.from_numpy(y.T.reshape(1, num_channels, num_samples).astype(np.float32))

            # output is [1, S, C, T] where S is the number of sources
            if gpu:
                # use current gpu device
                out = apply_model(model, x, progress=True, device=torch.cuda.current_device())
            else:
                # use cpu
                out = apply_model(model, x, progress=True)

            # vocals are the 4th source
            # drums.wav, bass.wav, other.wav, vocals.wav
            vox = out[0][3]

            # convert tensor back to numpy array
            vox_np = np.array(vox).T

            # save file
            name, ext = file.split('.')
            out_file = name + "_Vox." + ext
            out_path = os.path.join(out_dir, out_file)
            sf.write(out_path, vox_np, sr)

    print("Processing complete!")


if __name__ == '__main__':
    # run function
    separate(INPUT_DIR, OUTPUT_DIR)
