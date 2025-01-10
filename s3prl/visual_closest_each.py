import os
import pickle
import pandas as pd
import umap
import plotly.express as px
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
import numpy as np

# Function to read pickle files
def read_pickle_files(base_directory, feature_type, speaker_no, train_type):
    data = []
    labels = []

    speaker_path = os.path.join(base_directory, 'COMB_US', speaker_no, train_type)

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

    return pd.DataFrame({'features': data, 'label': labels})

# Function to calculate inter- and intra-cluster distances
def calculate_inter_intra_distances(visualization_df):
    unique_labels = visualization_df['label'].unique()
    cluster_centroids = {}
    intra_cluster_distances = {}

    # Calculate centroid and intra-cluster distances for each label
    for label in unique_labels:
        cluster_points = visualization_df[visualization_df['label'] == label][['UMAP1', 'UMAP2']].values
        centroid = np.mean(cluster_points, axis=0)
        cluster_centroids[label] = centroid

        distances = [euclidean(point, centroid) for point in cluster_points]
        intra_cluster_distances[label] = {
            "mean_intra_distance": np.mean(distances),
            "max_intra_distance": np.max(distances)
        }

    # Calculate inter-cluster distances ONLY from the "Original" label
    inter_cluster_distances = {}
    # Filter out "Original" from the labels to handle every other label
    other_labels = [lbl for lbl in unique_labels if lbl != "Original"]
    
    if "Original" in cluster_centroids:
        for lbl in other_labels:
            dist = euclidean(cluster_centroids["Original"], cluster_centroids[lbl])
            inter_cluster_distances[f"Original_to_{lbl}"] = dist
    else:
        # Optional: handle the scenario where 'Original' isn't present
        print("'Original' label not found in the data.")
    
    return {
        "intra_cluster_distances": intra_cluster_distances,
        "inter_cluster_distances": inter_cluster_distances
    }

def get_custom_color_map(labels, original_label="Original"):
    """
    Returns a dict that maps:
      - original_label -> "green"
      - other labels   -> unique colors from Alphabet (skipping indices 5,6,9).
    """
    # Collect unique labels
    unique_labels = list(set(labels))
    
    # Define the full color palette
    all_colors = px.colors.qualitative.Alphabet  # 26 distinct colors

    # Exclude colors at indices 5, 6, and 9
    excluded_indices = {5, 6, 9}
    filtered_colors = [c for i, c in enumerate(all_colors) if i not in excluded_indices]

    color_map = {}
    color_map[original_label] = "green"

    idx = 0
    for label in unique_labels:
        if label == original_label:
            continue
        color_map[label] = filtered_colors[idx % len(filtered_colors)]
        idx += 1
    
    return color_map
# Example usage in your visualize_umap function:

def visualize_umap(data, labels, feature_type, output_directory, model_no, results_df, speaker_no):
    png_output_path = os.path.join(output_directory, f'{model_no}_{feature_type}_umap.png')

    if not os.path.exists(png_output_path):
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(list(data['features']))

        umap_model = umap.UMAP(n_components=2, random_state=42)
        reduced_data = umap_model.fit_transform(scaled_features)

        visualization_df = pd.DataFrame({
            'UMAP1': reduced_data[:, 0],
            'UMAP2': reduced_data[:, 1],
            'label': labels
        })

        # Calculate distances (intra/inter) if desired
        distances = calculate_inter_intra_distances(visualization_df)
        if distances:
            distance_row = {"Feature_Type": feature_type, "Speaker": speaker_no}

            for label, intra_dist in distances['intra_cluster_distances'].items():
                distance_row[f"Intra_{label}_Mean"] = intra_dist['mean_intra_distance']
                distance_row[f"Intra_{label}_Max"] = intra_dist['max_intra_distance']

            for pair, inter_dist in distances['inter_cluster_distances'].items():
                distance_row[f"Inter_{pair}"] = inter_dist

            results_df.append(distance_row)

        # Build the color map:
        color_map = get_custom_color_map(visualization_df["label"], original_label="Original")

        # Create Plotly scatter
        fig = px.scatter(
            visualization_df,
            x='UMAP1',
            y='UMAP2',
            color='label',
            title=f'UMAP Visualization of {feature_type}',
            labels={'label': 'Label'},
            color_discrete_map=color_map
        )

        fig.write_image(png_output_path)
        print(f"Plot saved to {png_output_path}")
    else:
        print(f"{png_output_path} exists")
# Main execution
if __name__ == "__main__":
    base_directory = "/data/Deep_Fake_Data/Features_no_padding"
#     feature_types = [
#     "mockingjay_origin",
#     "mockingjay_100hr",
#     "mockingjay_960hr",
#     "mockingjay_logMelBase_T_AdamW_b32_200k_100hr",
#     "mockingjay_logMelBase_T_AdamW_b32_1m_960hr_drop1",
#     "mockingjay_logMelLinearLarge_T_AdamW_b32_500k_360hr_drop1",
#     "tera_100hr",
#     "tera_logMelBase_T_F_M_AdamW_b32_200k_100hr",
#     "tera_logMelBase_T_F_AdamW_b32_1m_960hr_drop1",
#     "audio_albert_960hr",
#     "audio_albert_logMelBase_T_share_AdamW_b32_1m_960hr_drop1",
#     "apc_360hr",
#     "apc_960hr",
#     "vq_apc_360hr",
#     "vq_apc_960hr",
#     "npc_360hr",
#     "npc_960hr",
#     "modified_cpc",
#     "decoar_layers",
#     "decoar2",
#     "wav2vec_large",
#     "vq_wav2vec_gumbel",
#     "vq_wav2vec_kmeans",
#     "discretebert",
#     "vq_wav2vec_kmeans_roberta",
#     "wav2vec2_base_960",
#     "wav2vec2_large_960",
#     "wav2vec2_large_ll60k",
#     "wav2vec2_large_lv60_cv_swbd_fsh",
#     "wav2vec2_conformer_relpos",
#     "wav2vec2_conformer_rope",
#     "wav2vec2_base_s2st_es_voxpopuli",
#     "wav2vec2_base_s2st_en_librilight",
#     "wav2vec2_conformer_large_s2st_es_voxpopuli",
#     "wav2vec2_conformer_large_s2st_en_librilight",
#     "xlsr_53",
#     "xls_r_300m",
#     "xls_r_1b",
#     "xls_r_2b",
#     "hubert_base",
#     "hubert_large_ll60k",
#     "mhubert_base_vp_en_es_fr_it3",
#     "espnet_hubert_base_iter0",
#     "espnet_hubert_base_iter1",
#     "cvhubert",
#     "wavlablm_ek_40k",
#     "wavlablm_mk_40k",
#     "wavlablm_ms_40k",
#     "multires_hubert_base",
#     "multires_hubert_large",
#     "multires_hubert_multilingual_base",
#     "multires_hubert_multilingual_large400k",
#     "multires_hubert_multilingual_large600k",
#     "distilhubert_base",
#     "hubert_base_robust_mgr",
#     "unispeech_sat_base",
#     "unispeech_sat_base_plus",
#     "unispeech_sat_large",
#     "wavlm_base",
#     "wavlm_base_plus",
#     "wavlm_large",
#     "data2vec_base_960",
#     "data2vec_large_ll60k",
#     "ast",
#     "ssast_frame_base",
#     "ssast_patch_base",
#     "mae_ast_frame",
#     "mae_ast_patch",
#     "byol_a_2048",
#     "byol_a_1024",
#     "byol_a_512",
#     "byol_s_default",
#     "byol_s_cvt",
#     "byol_s_resnetish34",
#     "vggish",
#     "passt_base"
# ]
    feature_types = ['ssast_patch_base', 'espnet_hubert_base_iter0', 'ast',
       'wav2vec2_base_s2st_es_voxpopuli', 'xlsr_53', 'hubert_large_ll60k',
       'wav2vec2_large_lv60_cv_swbd_fsh', 'wavlm_large']
    speaker_nos = ["p266"]
    results = []

    for speaker_no in speaker_nos:
        train_type = "train"
        output_directory = f"/data/Deep_Fake_Data/plots_v1/umap_plots/COMB_US/{speaker_no}"

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        for idx, feature_type in enumerate(feature_types):
            print(f"Processing feature type: {feature_type}, Speaker: {speaker_no}")

            data_df = read_pickle_files(base_directory, feature_type, speaker_no, train_type)

            if not data_df.empty:
                visualize_umap(data_df, data_df['label'], feature_type, output_directory, str(idx).zfill(3), results, speaker_no)
            else:
                print(f"No data found for feature type: {feature_type}")

    results_df = pd.DataFrame(results)
    results_output_path = os.path.join(base_directory, f"COMB_US.csv")
    results_df.to_csv(results_output_path, index=False)
    print(f"Results saved to {results_output_path}")
