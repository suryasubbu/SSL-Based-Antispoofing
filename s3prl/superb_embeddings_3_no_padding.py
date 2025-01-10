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
'ssast_patch_base', 'espnet_hubert_base_iter0', 'ast',
       'wav2vec2_base_s2st_es_voxpopuli', 'xlsr_53', 'hubert_large_ll60k',
       'wav2vec2_large_lv60_cv_swbd_fsh', 'wavlm_large', 'xls_r_1b',
       'data2vec_large_ll60k', 'data2vec_base_960', 'ssast_frame_base',
       'xls_r_300m', 'wav2vec2_large_ll60k', 'wavlablm_mk_40k',
       'unispeech_sat_large', 'wav2vec2_base_960',
       'wav2vec2_conformer_rope', 'wavlm_base_plus', 'byol_a_2048',
       'passt_base', 'byol_s_cvt', 'wav2vec2_large_960',
       'wavlablm_ms_40k', 'xls_r_2b',
       'wav2vec2_conformer_large_s2st_es_voxpopuli',
       'multires_hubert_base', 'unispeech_sat_base_plus', 'modified_cpc',
       'multires_hubert_multilingual_large600k', 'mae_ast_patch',
       'multires_hubert_multilingual_large400k', 'hubert_base_robust_mgr',
       'apc_960hr', 'byol_a_1024', 'wavlablm_ek_40k',
       'multires_hubert_large', 'espnet_hubert_base_iter1', 'hubert_base',
       'wavlm_base', 'wav2vec2_conformer_relpos',
       'wav2vec2_conformer_large_s2st_en_librilight', 'byol_a_512',
       'unispeech_sat_base', 'wav2vec2_base_s2st_en_librilight',
       'cvhubert', 'multires_hubert_multilingual_base', 'mae_ast_frame',
       'mhubert_base_vp_en_es_fr_it3', 'npc_360hr', 'npc_960hr',
       'decoar_layers', 'vq_apc_960hr', 'apc_360hr', 'distilhubert_base',
       'wav2vec_large', 'vq_apc_360hr', 'decoar2', 'byol_s_resnetish34',
       'byol_s_default', 'vq_wav2vec_gumbel', 'vq_wav2vec_kmeans',
       'discretebert', 'vq_wav2vec_kmeans_roberta',
       'tera_logMelBase_T_F_M_AdamW_b32_200k_100hr', 'tera_100hr',
       'mockingjay_100hr', 'mockingjay_logMelBase_T_AdamW_b32_200k_100hr',
       'mockingjay_960hr',
       'mockingjay_logMelBase_T_AdamW_b32_1m_960hr_drop1',
       'tera_logMelBase_T_F_AdamW_b32_1m_960hr_drop1',
       'audio_albert_logMelBase_T_share_AdamW_b32_1m_960hr_drop1',
       'audio_albert_960hr', 'mockingjay_origin',
       'mockingjay_logMelLinearLarge_T_AdamW_b32_500k_360hr_drop1',
       'vggish'
]

def extract_s3prl_features(model, file_path):
    # Load the audio file
    waveform, sample_rate = torchaudio.load(file_path)
    metadata = torchaudio.info(file_path)
    # return metadata.num_frames / metadata.sample_rate
    # # Resample to 16 kHz if necessary
    # if sample_rate != 16000:
    #     waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=16000)
    #     sample_rate = 16000


    # Ensure the waveform has the appropriate shape (batch, samples)
    if waveform.ndimension() == 1:
        waveform = waveform.unsqueeze(0)

    # Truncate or pad to a uniform length (10 seconds at 16kHz)
    max_length = int(sample_rate * (metadata.num_frames / metadata.sample_rate))
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
    database = "COMB_US"
    output_directory = "/data/Deep_Fake_Data/Features_no_padding"
    directory_path = os.path.join(base_directory, database)
    # speakers = get_subfolders(directory_path)
    speakers = ["p266"]
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
