import os
import numpy as np
import openl3
import soundfile as sf
import pickle
from tqdm import tqdm


# set hyperparameters
INPUT_REPRESENTATION = "mel256"
EMBEDDING_SIZE = 512


# set input and output directories
INPUT_DIR = "/scratch/rn2214/data/final_stems_44100"
OUTPUT_DIR = f"/scratch/rn2214/data/embeddings_{EMBEDDING_SIZE}"


def extract_embeddings(in_dir, out_dir):
    """
    Extract OpenL3 audio embeddings from vocal stem WAV files.

    :param in_dir: (str) directory of WAV files to extract embeddings from
    :param out_dir: (str) directory of where to save the extracted embeddings
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
    model = openl3.models.load_audio_embedding_model(input_repr=INPUT_REPRESENTATION,
                                                     content_type="music",
                                                     embedding_size=EMBEDDING_SIZE)
    print("Model loaded successfully.")
    
    # create lists of audio arrays and sample rate arrays
    name_list = []
    emb_list = []
    ts_list = []
    print("Extracting emebddings for each audio file...")
    for i in tqdm(range(len(file_list))):
        # only process wav files
        if file_list[i].endswith(".wav"):   
            # read the soundfile
            in_path = os.path.join(in_dir, file_list[i])
            y, sr = sf.read(in_path)
            
            # set the output file name
            s = file_list[i].split("_")
            out_name = f"{s[0]}_{s[1]}_Emb_{EMBEDDING_SIZE}.npy"

            # extract embedding
            emb, ts = openl3.get_audio_embedding(y, sr, model=model)
            
            name_list.append(out_name)
            emb_list.append(emb)
            ts_list.append(ts)

             # save individual embedding file
            out_path = os.path.join(out_dir, out_name)
            np.save(out_path, emb)
    
    print("Dumping all embeddings, timestamps, and file names as a single pickle file...")
    # save all data as one object for easy loading
    data_obj = (emb_list, ts_list, name_list)
    out_path = os.path.join(out_dir, f"all_embeddings_{EMBEDDING_SIZE}.pkl")

    # dump pickle file
    with open(out_path, "wb") as f:
        pickle.dump(data_obj, f)
    print("Data dumped successfully!")

    print("Processing complete!")


if __name__ == '__main__':
    # run function
    extract_embeddings(INPUT_DIR, OUTPUT_DIR)
    