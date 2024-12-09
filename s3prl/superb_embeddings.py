# Import required libraries
import torchaudio
from s3prl.nn import S3PRLUpstream
import torch
import os
import pickle
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# List of S3PRL upstream model names to iterate over
s3prl_model_names = [
    "apc", "hubert", "wav2vec2", "tera", "mockingjay", "vq_apc", "vq_wav2vec",
    "wav2vec", "decoar", "decoar2", "npc", "audio_albert", "modified_cpc",
    "pase", "pase_plus", "distilhubert", "hubert_large_ll60k", "wav2vec2_large_ll60k"
]

# Load all S3PRL models once
models_s3prl = {}
for model_name in s3prl_model_names:
    print(f"Loading model: {model_name}")
    models_s3prl[model_name] = S3PRLUpstream(model_name).to(device).eval()

# Feature extraction function for S3PRL
def extract_s3prl_features(model, file_path):
    # Load the audio file
    waveform, sample_rate = torchaudio.load(file_path)

    # Ensure the waveform has the appropriate shape (batch, samples)
    if waveform.ndimension() == 1:
        waveform = waveform.unsqueeze(0)

    # Truncate or pad to a uniform length (e.g., 16000 * 10 samples for 10 seconds)
    max_length = 16000 * 10
    wavs = torch.zeros(waveform.size(0), max_length)
    for i, wav in enumerate(waveform):
        wavs[i, :min(max_length, wav.size(0))] = wav[:max_length]

    # Create a tensor of wave lengths
    wavs_len = torch.LongTensor([min(max_length, waveform.size(1)) for _ in range(waveform.size(0))])

    # Run through the model
    with torch.no_grad():
        all_hs, all_hs_len = model(wavs.to(device), wavs_len.to(device))

    # Compute mean embedding over axis=1
    embeddings = all_hs[0].mean(dim=1).squeeze().cpu().numpy()
    return embeddings

# Function to get subfolders in a directory
def get_subfolders(directory):
    return [folder.name for folder in os.scandir(directory) if folder.is_dir()]

# Function to process a speaker for a given model
def process_speaker_for_model(model_name, model, speaker):
    input_base_path = f"/data/Deep_Fake_Data/Raw_data/DFADD/{speaker}/train"
    deepfake_folders = get_subfolders(input_base_path)

    for folder in deepfake_folders:
        train_dir = os.path.join(input_base_path, folder)
        if not os.path.exists(train_dir):
            print(f"Train directory for {folder} does not exist.")
            continue

        # Create the output directory similar to input audio directory
        output_dir = os.path.join(f"/data/Deep_Fake_Data/Raw_data/Features_x/DFADD/{speaker}/train/{folder}/{model_name}")
        os.makedirs(output_dir, exist_ok=True)

        # List all wave files in the current train folder
        wave_files = np.sort([f for f in os.listdir(train_dir) if f.endswith(('.wav', '.flac'))])

        print(f"{model_name} processing: {folder}")

        for f in wave_files:
            file_path = os.path.join(train_dir, f)

            # Extract features
            features = extract_s3prl_features(model, file_path)

            # Save features as a pickle file
            output_file_path = os.path.join(output_dir, f"{os.path.splitext(f)[0]}.pkl")
            with open(output_file_path, 'wb') as handle:
                pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Main processing loop
def main():
    # Directory containing speaker subfolders
    directory_path = '/data/Deep_Fake_Data/Raw_data/DFADD'
    speakers = get_subfolders(directory_path)

    # Use ThreadPoolExecutor to parallelize feature extraction
    for speaker in speakers:
        try:
            with ThreadPoolExecutor() as executor:
                futures = []
                for model_name, model in models_s3prl.items():
                    futures.append(executor.submit(process_speaker_for_model, model_name, model, speaker))

                # Wait for all futures to complete
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        print(f"Generated an exception: {exc}")
        except Exception as e:
            print(f"Exception occurred while processing speaker {speaker}: {e}")
            continue

if __name__ == "__main__":
    main()
