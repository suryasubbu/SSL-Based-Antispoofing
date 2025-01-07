import torchaudio
from s3prl.nn import S3PRLUpstream
import torch
import os
import pickle
import numpy as np
import concurrent.futures
import torch.multiprocessing as mp

# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

s3prl_model_names = [
    "mockingjay_origin",
    "mockingjay_100hr",
    "mockingjay_960hr",
    "mockingjay_logMelBase_T_AdamW_b32_200k_100hr",
    "mockingjay_logMelBase_T_AdamW_b32_1m_960hr_drop1",
    "mockingjay_logMelLinearLarge_T_AdamW_b32_500k_360hr_drop1",
    "tera_100hr",
    "tera_logMelBase_T_F_M_AdamW_b32_200k_100hr",
    "tera_logMelBase_T_F_AdamW_b32_1m_960hr_drop1",
    "audio_albert_960hr",
    "audio_albert_logMelBase_T_share_AdamW_b32_1m_960hr_drop1",
    "apc_360hr",
    "apc_960hr",
    "vq_apc_360hr",
    "vq_apc_960hr",
    "npc_360hr",
    "npc_960hr",
    "modified_cpc",
    "decoar_layers",
    "decoar2",
    "wav2vec_large",
    "vq_wav2vec_gumbel",
    "vq_wav2vec_kmeans",
    "discretebert",
    "vq_wav2vec_kmeans_roberta",
    "wav2vec2_base_960",
    "wav2vec2_large_960",
    "wav2vec2_large_ll60k",
    "wav2vec2_large_lv60_cv_swbd_fsh",
    "wav2vec2_conformer_relpos",
    "wav2vec2_conformer_rope",
    "wav2vec2_base_s2st_es_voxpopuli",
    "wav2vec2_base_s2st_en_librilight",
    "wav2vec2_conformer_large_s2st_es_voxpopuli",
    "wav2vec2_conformer_large_s2st_en_librilight",
    "xlsr_53",
    "xls_r_300m",
    "xls_r_1b",
    "xls_r_2b",
    "hubert_base",
    "hubert_large_ll60k",
    "mhubert_base_vp_en_es_fr_it3",
    "espnet_hubert_base_iter0",
    "espnet_hubert_base_iter1",
    "cvhubert",
    "wavlablm_ek_40k",
    "wavlablm_mk_40k",
    "wavlablm_ms_40k",
    "multires_hubert_base",
    "multires_hubert_large",
    "multires_hubert_multilingual_base",
    "multires_hubert_multilingual_large400k",
    "multires_hubert_multilingual_large600k",
    "distilhubert_base",
    "hubert_base_robust_mgr",
    "unispeech_sat_base",
    "unispeech_sat_base_plus",
    "unispeech_sat_large",
    "wavlm_base",
    "wavlm_base_plus",
    "wavlm_large",
    "data2vec_base_960",
    "data2vec_large_ll60k",
    "ast",
    "ssast_frame_base",
    "ssast_patch_base",
    "mae_ast_frame",
    "mae_ast_patch",
    "byol_a_2048",
    "byol_a_1024",
    "byol_a_512",
    "byol_s_default",
    "byol_s_cvt",
    "byol_s_resnetish34",
    "vggish",
    "passt_base"
]

def extract_s3prl_features(model, file_path):
    # Load the audio file
    waveform, sample_rate = torchaudio.load(file_path)

    # Resample to 16 kHz if necessary
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=16000)
        sample_rate = 16000


    # Ensure the waveform has the appropriate shape (batch, samples)
    if waveform.ndimension() == 1:
        waveform = waveform.unsqueeze(0)

    # Truncate or pad to a uniform length (10 seconds at 16kHz)
    max_length = 16000 * 10
    wavs = torch.zeros(waveform.size(0), max_length)
    for i, wav in enumerate(waveform):
        wavs[i, :min(max_length, wav.size(0))] = wav[:max_length]

    # Create a tensor of wave lengths
    wavs_len = torch.LongTensor([min(max_length, waveform.size(1)) for _ in range(waveform.size(0))])

    with torch.no_grad():
        all_hs, all_hs_len = model(wavs.to(device), wavs_len.to(device))

    # Compute mean embedding over axis=1
    embeddings = all_hs[0].mean(dim=1).squeeze().cpu().numpy()
    return embeddings

def get_subfolders(directory):
    return [folder.name for folder in os.scandir(directory) if folder.is_dir()]

def process_speaker_for_model(base_directory, database, output_directory, model_name, speaker):
    """
    This function will be executed in a separate process.
    It loads the model, processes the speaker with this model.
    """
    input_base_path = os.path.join(base_directory, database, speaker, "train")
    if not os.path.exists(input_base_path):
        print(f"No train directory for speaker: {speaker}")
        return

    deepfake_folders = get_subfolders(input_base_path)

    # Load the model
    print(f"[{speaker}] Loading model: {model_name}")
    model = S3PRLUpstream(model_name).to(device).eval()

    # Process all deepfake folders
    for folder in deepfake_folders:
        train_dir = os.path.join(input_base_path, folder)
        if not os.path.exists(train_dir):
            print(f"Train directory for {folder} does not exist.")
            continue

        # Construct the output directory
        model_output_dir = os.path.join(output_directory, database, speaker, "train", folder, model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        # List all wave files in the current train folder
        wave_files = np.sort([f for f in os.listdir(train_dir) if f.endswith(('.wav', '.flac'))])

        for f in wave_files:
            file_path = os.path.join(train_dir, f)
            output_file_path = os.path.join(model_output_dir, f"{os.path.splitext(f)[0]}.pkl")
            if not os.path.exists(output_file_path):
                try:
                    features = extract_s3prl_features(model, file_path)
                    with open(output_file_path, 'wb') as handle:
                        pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    print(f"Error processing {file_path} with model {model_name}: {e}")

    print(f"[{speaker}] Finished with model {model_name}")

def main():
    # Define base_directory and database
    base_directory = "/data/Deep_Fake_Data/Raw_data"
    database = "CODEC2"
    output_directory = "/data/Deep_Fake_Data/Features"
    directory_path = os.path.join(base_directory, database)
    speakers = get_subfolders(directory_path)
    # speakers = ["p225",  "p229",  "p233",  "p238",  "p243",  "p247",  "p251",  "p255",  "p259", "p263",  "p267",  "p271",  "p275",  "p279",  "p283",  "p287",  "p294",  "p299",  "p303",  "p307",  "p312" , "p316" , "p326",  "p334" , "p340",  "p347",  "p362",  "p376"]
    # Process one speaker at a time
    for speaker in speakers:
        # Break the model list into batches of 10
        for i in range(0, len(s3prl_model_names), 10):
            model_chunk = s3prl_model_names[i:i+10]

            # Process these 10 models in parallel for this speaker
            with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
                futures = {
                    executor.submit(
                        process_speaker_for_model,
                        base_directory,
                        database,
                        output_directory,
                        model_name,
                        speaker
                    ): model_name for model_name in model_chunk
                }

                for future in concurrent.futures.as_completed(futures):
                    model_name = futures[future]
                    try:
                        future.result()
                        print(f"[{speaker}] Done with model: {model_name}")
                    except Exception as exc:
                        print(f"[{speaker}] Model {model_name} generated an exception: {exc}")

        print(f"Finished processing speaker: {speaker}")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    import sys
    main()
