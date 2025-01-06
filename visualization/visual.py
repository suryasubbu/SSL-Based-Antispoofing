import os
import pickle
import pandas as pd
import numpy as np
import umap
import plotly.express as px

from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings("ignore")  # Suppress all warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)  # Suppress only DeprecationWarnings


def read_pickle_files(speaker_dir, feature_type):
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

    for label in os.listdir(speaker_dir):
        label_path = os.path.join(speaker_dir, label)
        feature_path = os.path.join(label_path, feature_type)

        if not os.path.exists(feature_path):
            continue

        for file in os.listdir(feature_path):
            if file.endswith('.pkl'):
                file_path = os.path.join(feature_path, file)

                with open(file_path, 'rb') as f:
                    feature_data = pickle.load(f)
                    # feature_data = np.load(f, allow_pickle=True) 
                
                data.append(feature_data)
                labels.append(label)

    # Combine into a Dictionary
    data_dict = {'features': data, 'label': labels}
    return data_dict

def compute_umap_embeddings(data, feature_name, num_components=2, umap_model_dir='./umap', apply_scaler=False, recompute_embeddings=False):

    if not os.path.exists(umap_model_dir):
        os.makedirs(umap_model_dir)
    
    umap_model = os.path.join(umap_model_dir, feature_name + '_' + str(num_components) + 'd.pkl')

    if apply_scaler:
        ss=StandardScaler().fit(data)
        data = ss.transform(data)

    # compute and save the embeddings
    if not os.path.exists(umap_model) or recompute_embeddings:
        # compute umap embedding model
        reducer = umap.UMAP(random_state=42, n_neighbors=20, n_components=num_components)

        print(data.shape)
        embeddings_mapper = reducer.fit(data)

        with open(umap_model,'wb') as f:
            pickle.dump(reducer, f)

    else:
        with open(umap_model, 'rb') as f:
            reducer = pickle.load(f)

    umap_embeddings = reducer.transform(data)

    print(umap_embeddings.shape)

    return umap_embeddings


def visualize_umap(umap_embeddings, labels, feature_type, output_directory, model_no, fixed_label='Original'):
    """
    Apply UMAP and save the plots using Plotly.

    Parameters:
        data (DataFrame): Data with features.
        labels (List): Corresponding labels for the data.
        feature_type (str): The feature type being visualized (for file naming).
        output_directory (str): Directory to save the plots.
    """
    png_output_path = os.path.join(output_directory, f'{model_no}_{feature_type}_umap.png')

    if not os.path.exists(png_output_path):

        # Create a DataFrame for visualization
        visualization_df = pd.DataFrame({
            'UMAP1': umap_embeddings[:, 0],
            'UMAP2': umap_embeddings[:, 1],
            'label': labels
        })

        # Define a color map: "Original" is green, others in warm colors
        unique_labels = visualization_df['label'].unique()
        warm_colors = px.colors.sequential.solar[1:]  # Modify this list to your preferred warm color scheme
        color_map = {label: "green" if label == fixed_label else warm_colors[i % len(warm_colors)]
                    for i, label in enumerate(unique_labels)}

        # Plot using Plotly
        fig = px.scatter(
            visualization_df, 
            x='UMAP1', 
            y='UMAP2', 
            color='label', 
            title=f'UMAP Visualization of {feature_type}',
            labels={'label': 'Label'},
            color_discrete_map=color_map
        )

        # Save the plot as HTML
        # os.makedirs(output_directory, exist_ok=True)
        # html_output_path = os.path.join(output_directory, f'{feature_type}_umap.html')
        # fig.write_html(html_output_path)
        # print(f"Plot saved to {html_output_path}")

        # Save the plot as PNG
        fig.write_image(png_output_path)
        print(f"Plot saved to {png_output_path}")

    else:

        print(f"{png_output_path} already exists!!!")


# Main execution
if __name__ == "__main__":

    base_directory = "/data/Deep_Fake_Data/"
    feat_directory = "Features"
    # data_names = ['DFADD', 'CODEC1', 'CODEC2', 'FF2']
    data_names = ['DFADD']
    
    umap_dir_name = 'umap'
    plot_dir_name = 'plots'

    data_types = ["train"]

    feature_types = ["mockingjay_origin", "mockingjay_100hr", "mockingjay_960hr", "mockingjay_logMelBase_T_AdamW_b32_200k_100hr", "mockingjay_logMelBase_T_AdamW_b32_1m_960hr_drop1", 
                     "mockingjay_logMelLinearLarge_T_AdamW_b32_500k_360hr_drop1",
                     "tera", "tera_100hr", "tera_logMelBase_T_F_M_AdamW_b32_200k_100hr", "tera_logMelBase_T_F_AdamW_b32_1m_960hr_drop1",
                     "audio_albert", "audio_albert_960hr", "audio_albert_logMelBase_T_share_AdamW_b32_1m_960hr_drop1", 
                     "apc", "apc_360hr", "apc_960hr", "vq_apc", "vq_apc_360hr", "vq_apc_960hr", "npc", "npc_360hr", "npc_960hr",
                     "modified_cpc", "decoar", "decoar_layers", "decoar2",
                     "wav2vec", "wav2vec_large", "vq_wav2vec_gumbel", "vq_wav2vec_kmeans", "vq_wav2vec_kmeans_roberta", "wav2vec2_base_960", "wav2vec2_large_960", "wav2vec2_large_ll60k", 
                     "wav2vec2_large_lv60_cv_swbd_fsh", "wav2vec2_conformer_relpos", "wav2vec2_conformer_rope", "wav2vec2_base_s2st_es_voxpopuli", "wav2vec2_base_s2st_en_librilight", 
                     "wav2vec2_conformer_large_s2st_es_voxpopuli", "wav2vec2_conformer_large_s2st_en_librilight",
                     "discretebert", "xlsr_53", "xls_r_300m", "xls_r_1b", "xls_r_2b", 
                     "hubert_base", "hubert_large_ll60k", "mhubert_base_vp_en_es_fr_it3", "espnet_hubert_base_iter0", "espnet_hubert_base_iter1", "cvhubert", "multires_hubert_base",
                     "multires_hubert_large", "multires_hubert_multilingual_base", "multires_hubert_multilingual_large400k", "multires_hubert_multilingual_large600k", "distilhubert_base", "hubert_base_robust_mgr",
                     "wavlablm_ek_40k", "wavlablm_mk_40k", "wavlablm_ms_40k",
                     "unispeech_sat", "unispeech_sat_base", "unispeech_sat_base_plus", "unispeech_sat_large", "wavlm_base", "wavlm_base_plus", "wavlm_large",
                     "data2vec_base_960", "data2vec_large_ll60k",
                     "ast", "ssast_frame_base", "ssast_patch_base", "mae_ast_frame", "mae_ast_patch",
                     "byol_a_2048", "byol_a_1024", "byol_a_512", "byol_s_default", "byol_s_cvt", "byol_s_resnetish34", 
                     "vggish",
                     "passt_base"]  # List of feature types

    # speakers_ls = ["p278", "p376", "p265", "p318", "s5", "p272", "p306", "p239", "p287", "p262", "p288", "p284", "p360", "p251", "p312", "p282"]
    # speakers_ls = ["p282"]

    for data_name in data_names:

        data_path = os.path.join(base_directory, feat_directory, data_name)

        speakers_ls = [ item for item in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, item)) ]

        print(speakers_ls)

        for spkr in speakers_ls:

            for dt in data_types:
                
                plot_directory = os.path.join(base_directory, plot_dir_name, umap_dir_name + '_' + plot_dir_name, data_name, spkr)
                
                if not os.path.exists(plot_directory):
                    os.makedirs(plot_directory)

                for idx, feature_type in enumerate(feature_types):

                    print("Computing umap embeddings and viusalization for speaker {} and feature {}".format(spkr, feature_type))

                    speaker_path = os.path.join(base_directory, feat_directory, data_name, spkr, dt)

                    feat_data_dict = read_pickle_files(speaker_path, feature_type)

                    if len(feat_data_dict['features']) != 0:

                        umap_model_dir = os.path.join(base_directory, umap_dir_name, data_name, spkr)
                        
                        feat_embed = np.array(feat_data_dict['features'])
                        print(feat_embed.shape)
                        umap_embeddings = compute_umap_embeddings(feat_embed, feature_type, num_components=2, umap_model_dir=umap_model_dir, apply_scaler=True, recompute_embeddings=False)

                        visualize_umap(umap_embeddings, feat_data_dict['label'], feature_type, plot_directory, str(idx).zfill(3), )

                        # if not data_df.empty:
                        #     visualize_umap(umap_embeddings, data_df['label'], feature_type, output_directory,str(idx).zfill(3))
                        # else:
                        #     print(f"No data found for feature type: {feature_type}")

                    else:
                        print("Features are not extracted for {}".format(feature_type))
