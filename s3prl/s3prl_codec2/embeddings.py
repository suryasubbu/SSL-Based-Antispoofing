from transformers import Wav2Vec2Model, HubertModel, AutoProcessor, WavLMModel, AlbertModel, UniSpeechSatModel, Data2VecAudioModel,Wav2Vec2FeatureExtractor, AutoFeatureExtractor
from transformers import AutoModel as AM
import torchaudio
import torch
import librosa
from transformers import SeamlessM4TFeatureExtractor
import os
import pandas as pd
import numpy as np
import concurrent.futures
from funasr import AutoModel
import tensorflow_hub as hub
import tensorflow as tf
import wav2clip  # Added wav2clip import
import pickle
from speechbrain.inference.speaker import EncoderClassifier 
# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Models and Processors as Public Variables
print("Loading Models...")

# # Load HuBERT model
# processor_hubert = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
# model_hubert = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft").to(device)



# processor_wavlm_large = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")
# model_wavlm_large = AM.from_pretrained("microsoft/wavlm-large").to(device)

# # Load Wav2Vec2.0 model facebook/wav2vec2-base-960h
# processor_wav2vec2 = AutoProcessor.from_pretrained("facebook/wav2vec2-large-960h")
# model_wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h").to(device)



# Load Wav2Vec2.0xls model
# processor_wav2vec2_xls = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
# model_wav2vec2_xls = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53").to(device)

# # Load GCT model
# processor_gct = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")


# # Load YamNet model
# yamnet_model_handle = "https://tfhub.dev/google/yamnet/1"
# yamnet_model = hub.load(yamnet_model_handle)

# # Load UniSpeech-SAT model
# processor_unispeech_sat = AutoProcessor.from_pretrained("microsoft/unispeech-sat-base-100h-libri-ft")
# model_unispeech_sat = UniSpeechSatModel.from_pretrained("microsoft/unispeech-sat-base-100h-libri-ft").to(device)

# # Load wav2clip model
# model_wav2clip = wav2clip.get_model()

# # Load Data2VecAudio model
# processor_data2vec = AutoProcessor.from_pretrained("facebook/data2vec-audio-base-960h")
# model_data2vec = Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-base-960h").to(device)

# # Load X-Vector model from SpeechBrain (updated)
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb",
    savedir="pretrained_models/spkrec-xvect-voxceleb"
)

# Define Feature Extraction Functions
def extract_xvector_features(file_path):
    signal, fs = torchaudio.load(file_path)  # Load audio signal
    embeddings = classifier.encode_batch(signal)  # Extract embeddings
    return embeddings.squeeze().cpu().numpy()



# # Define Feature Extraction Functions
# def extract_hubert_features(file_path):
#     waveform, _ = librosa.load(file_path, sr=16000)
#     inputs = processor_hubert(waveform, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
#     with torch.no_grad():
#         features = model_hubert(inputs.input_values)
#     return features.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

# def extract_wavlm_features(file_path):
#     waveform, _ = librosa.load(file_path, sr=16000)
#     inputs = processor_wavlm(waveform, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
#     with torch.no_grad():
#         features = model_wavlm(inputs.input_values)
#     return features.extract_features.mean(dim=1).squeeze().cpu().numpy()

# def extract_wav2vec2_base_features(file_path):
#     waveform, _ = librosa.load(file_path, sr=16000)
#     inputs = processor_wav2vec2_base(waveform, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
#     with torch.no_grad():
#         features = model_wav2vec2_base(inputs.input_values)
#     return features.extract_features.mean(dim=1).squeeze().cpu().numpy()
# def extract_wavlm_large_features(file_path):
#     waveform, _ = librosa.load(file_path, sr=16000)
#     inputs = processor_wavlm_large(waveform, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
#     with torch.no_grad():
#         features = model_wavlm_large(inputs.input_values)
#     return features.extract_features.mean(dim=1).squeeze().cpu().numpy()

# def extract_wav2vec2_features(file_path):
#     waveform, _ = librosa.load(file_path, sr=16000)
#     inputs = processor_wav2vec2(waveform, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
#     with torch.no_grad():
#         features = model_wav2vec2(inputs.input_values)
#     return features.extract_features.mean(dim=1).squeeze().cpu().numpy()

# def extract_wav2vec2_xls_features(file_path):
#     waveform, _ = librosa.load(file_path, sr=16000)
#     inputs = processor_wav2vec2_xls(waveform, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
#     with torch.no_grad():
#         features = model_wav2vec2_xls(inputs.input_values)
#     return features.extract_features.mean(dim=1).squeeze().cpu().numpy()

# def extract_gct_features(file_path):
#     waveform, _ = librosa.load(file_path, sr=16000)
#     inputs = processor_gct(waveform, sampling_rate=16000, return_tensors="pt").to(device)
#     input_features = inputs["input_features"][0]
#     return input_features.squeeze().cpu().numpy().mean(axis=0)

# def extract_emotion2vec_features(file_path):
#     rec_result = model_emotion2vec.generate(file_path, output_dir="outputs_emo", granularity="utterance", extract_embedding=True)
#     float_array = rec_result[0]["scores"]
#     max_index = float_array.index(max(rec_result[0]["scores"]))
#     emo = rec_result[0]["labels"][max_index]
#     aa = np.array(rec_result[0]["feats"])
#     return aa, emo

# def extract_yamnet_features(file_path):
#     waveform, sr = librosa.load(file_path, sr=16000)
#     waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)
#     scores, embeddings, spectrogram = yamnet_model(waveform)
#     return embeddings.numpy().mean(axis=0)

# def extract_unispeech_sat_features(file_path):
#     waveform, _ = librosa.load(file_path, sr=16000)
#     inputs = processor_unispeech_sat(waveform, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
#     with torch.no_grad():
#         features = model_unispeech_sat(inputs.input_values)
#     return features.extract_features.mean(dim=1).squeeze().cpu().numpy()

# def extract_wav2clip_features(file_path):
#     waveform, _ = librosa.load(file_path, sr=16000)
#     embeddings = wav2clip.embed_audio(waveform, model_wav2clip)
#     return embeddings

# def extract_data2vec_features(file_path):
#     waveform, _ = librosa.load(file_path, sr=16000)
#     inputs = processor_data2vec(waveform, sampling_rate=16000, return_tensors="pt").to(device)
#     with torch.no_grad():
#         outputs = model_data2vec(**inputs)
#     return outputs.extract_features.mean(dim=1).squeeze().cpu().numpy()

# Corresponding feature extraction functions
feature_extraction_functions = {
    # "HuBERT": extract_hubert_features,
    # "wavlmlarge": extract_wavlm_large_features,
    # "w2vxls": extract_wav2vec2_xls_features,
#     "YamNet": extract_yamnet_features,
    #  "unispeech": extract_unispeech_sat_features,
#     "wav2clip": extract_wav2clip_features,
#     "Data2Vec": extract_data2vec_features,
    "xvector": extract_xvector_features
 }

def get_subfolders(directory):
    subfolders = [folder.name for folder in os.scandir(directory) if folder.is_dir()]
    return subfolders

# Example usage
directory_path = '/data/Deep_Fake_Data/Raw_data/CODEC2'
# speakers = get_subfolders(directory_path)
speakers = ["p282","p351"]
def process_speaker_for_model(model_name, feature_extractor, speaker):
    input_base_path = f"/data/Deep_Fake_Data/Raw_data/CODEC2/{speaker}/train"

    deepfake_folders =get_subfolders(input_base_path)

    for folder in deepfake_folders:
        train_dir = os.path.join(input_base_path, folder)
        if not os.path.exists(train_dir):
            print(f"Train directory for {folder} does not exist.")
            continue

        # Create the output directory similar to input audio directory
        output_dir = os.path.join(f"/data/Deep_Fake_Data/Raw_data/Features_superb/CODEC2/{speaker}/train/{folder}/{model_name}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # List all wave files in the current train folder
        wave_files = np.sort([f for f in os.listdir(train_dir) if f.endswith(('.wav', '.flac'))])

        print(f"{model_name} processing: {folder}",speaker)

        for f in wave_files:
            # try:
                file_path = os.path.join(train_dir, f)

                features = feature_extractor(file_path)
                # Flatten the features and save them as a pickle file
                features_flattened = features.flatten()
                output_file_path = os.path.join(output_dir, f"{os.path.splitext(f)[0]}.pkl")
                with open(output_file_path, 'wb') as handle:
                    pickle.dump(features_flattened, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # except:
            #     print("yes")
            #     continue
# Use ThreadPoolExecutor to parallelize feature extraction
for speaker in speakers:
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for model_name, feature_extractor in feature_extraction_functions.items():
                futures.append(executor.submit(process_speaker_for_model, model_name, feature_extractor, speaker))

            # Wait for all futures to complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print(f"Generated an exception: {exc}")
                    continue
    except Exception as e:
        print(f"Exception occurred while processing speaker {speaker}: {e}")
        continue

    print(speaker,"done")

