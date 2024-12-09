import os
import pickle
import pandas as pd
import umap
import plotly.express as px


def read_pickle_files(base_directory, feature_type, speaker_no, train_type):
    """
    Reads pickle files for a given feature type, speaker, and train type.

    Parameters:
        base_directory (str): Path to the base directory.
        feature_type (str): Feature type to process (e.g., 'wave2vec2').
        speaker_no (str): Speaker number.
        train_type (str): Train type (e.g., 'train', 'test').

    Returns:
        DataFrame: Combined data with features and labels.
    """
    data = []
    labels = []

    speaker_path = os.path.join(base_directory, 'DFADD', speaker_no, train_type)

    for label in os.listdir(speaker_path):
        label_path = os.path.join(speaker_path, label)
        feature_path = os.path.join(label_path, feature_type)

        if not os.path.exists(feature_path):
            continue

        for file in os.listdir(feature_path):
            if file.endswith('.pkl'):
                file_path = os.path.join(feature_path, file)

                with open(file_path, 'rb') as f:
                    feature_data = pickle.load(f)
                    data.append(feature_data)
                    labels.append(label)

    # Combine into a DataFrame
    return pd.DataFrame({'features': data, 'label': labels})


def visualize_umap(data, labels, feature_type, output_directory):
    """
    Apply UMAP and save the plots using Plotly.

    Parameters:
        data (DataFrame): Data with features.
        labels (List): Corresponding labels for the data.
        feature_type (str): The feature type being visualized (for file naming).
        output_directory (str): Directory to save the plots.
    """
    # UMAP transformation
    umap_model = umap.UMAP(n_components=2, random_state=42)
    reduced_data = umap_model.fit_transform(list(data['features']))

    # Create a DataFrame for visualization
    visualization_df = pd.DataFrame({
        'UMAP1': reduced_data[:, 0],
        'UMAP2': reduced_data[:, 1],
        'label': labels
    })

    # Plot using Plotly
    fig = px.scatter(
        visualization_df, 
        x='UMAP1', 
        y='UMAP2', 
        color='label', 
        title=f'UMAP Visualization of {feature_type}',
        labels={'label': 'Label'}
    )

    # Save the plot as HTML
    # os.makedirs(output_directory, exist_ok=True)
    # html_output_path = os.path.join(output_directory, f'{feature_type}_umap.html')
    # fig.write_html(html_output_path)
    # print(f"Plot saved to {html_output_path}")

    # Save the plot as PNG
    png_output_path = os.path.join(output_directory, f'{feature_type}_umap.png')
    fig.write_image(png_output_path)
    print(f"Plot saved to {png_output_path}")


# Main execution
if __name__ == "__main__":
    base_directory = "/data/Deep_Fake_Data/Raw_data/Features_superb"
    feature_types = [ 
  "mockingjay_origin",
  "mockingjay_100hr",
  "mockingjay_960hr",
  "mockingjay_logMelBase_T_AdamW_b32_200k_100hr",
  "mockingjay_logMelBase_T_AdamW_b32_1m_960hr_drop1",
  "mockingjay_logMelLinearLarge_T_AdamW_b32_500k_360hr_drop1",
  "tera",
  "tera_100hr",
  "tera_logMelBase_T_F_M_AdamW_b32_200k_100hr",
  "tera_logMelBase_T_F_AdamW_b32_1m_960hr_drop1",
  "audio_albert",
  "audio_albert_960hr",
  "audio_albert_logMelBase_T_share_AdamW_b32_1m_960hr_drop1",
  "apc",
  "apc_360hr",
  "apc_960hr",
  "vq_apc",
  "vq_apc_360hr",
  "vq_apc_960hr",
  "npc",
  "npc_360hr",
  "npc_960hr",
  "modified_cpc",
  "decoar",
  "decoar_layers",
  "decoar2",
  "wav2vec",
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
  "unispeech_sat",
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
  "byol_s_resnetish34"
  "vggish",
  "passt_base"]  # List of feature types
    speaker_no = "p236"
    train_type = "train"
    output_directory = "/data/Deep_Fake_Data/umap_plots"  # Directory to save plots

    for feature_type in feature_types:
        print(f"Processing feature type: {feature_type}")

        data_df = read_pickle_files(base_directory, feature_type, speaker_no, train_type)

        if not data_df.empty:
            visualize_umap(data_df, data_df['label'], feature_type, output_directory)
        else:
            print(f"No data found for feature type: {feature_type}")
