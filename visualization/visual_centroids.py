import os
import pickle
import pandas as pd
import umap
import plotly.express as px
from scipy.spatial.distance import euclidean

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

def calculate_distances_from_fixed_label(visualization_df, fixed_label):
    """
    Calculate distances from a fixed label cluster to all other clusters and their average.

    Parameters:
        visualization_df (DataFrame): DataFrame containing UMAP features and labels.
        fixed_label (str): The fixed label to calculate distances from.

    Returns:
        dict: A dictionary containing distances and the average distance.
    """
    # Calculate centroids for each cluster
    centroids = visualization_df.groupby('label')[['UMAP1', 'UMAP2']].mean()

    # Ensure the fixed label is present
    if fixed_label not in centroids.index:
        print(f"Fixed label '{fixed_label}' not found in the data.")
        return None

    # Calculate distances from the fixed label to all other clusters
    fixed_label_centroid = centroids.loc[fixed_label]
    distances = {
        f"Distance_{fixed_label}_{other_label}": euclidean(fixed_label_centroid, other_centroid)
        for other_label, other_centroid in centroids.iterrows()
        if other_label != fixed_label
    }

    # Calculate average distance
    average_distance = sum(distances.values()) / len(distances) if distances else 0
    distances['Average_Distance'] = average_distance

    return distances, centroids

def visualize_umap(data, labels, feature_type, output_directory, model_no, results_df, speaker_no):
    """
    Apply UMAP and save the plots using Plotly, while calculating distances from a fixed label cluster.

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

    # Calculate distances from fixed label
    fixed_label = "Original"  # Replace with your desired fixed label
    distances, centroids = calculate_distances_from_fixed_label(visualization_df, fixed_label)
    if distances:
        print(f"Feature Type: {feature_type}")
        print(f"Distances from {fixed_label}: {distances}")

        # Add distances to results DataFrame
        distance_row = {"Feature_Type": feature_type, "Speaker": speaker_no}
        distance_row.update(distances)
        results_df.append(distance_row)

    # Plot using Plotly
    fig = px.scatter(
        visualization_df, 
        x='UMAP1', 
        y='UMAP2', 
        color='label', 
        title=f'UMAP Visualization of {feature_type}',
        labels={'label': 'Label'}
    )

    # Add centroids to the plot
    for label, centroid in centroids.iterrows():
        fig.add_scatter(x=[centroid['UMAP1']], y=[centroid['UMAP2']],
                        mode='markers+text',
                        marker=dict(size=10, symbol='x', color='black'),
                        text=label,
                        textposition='top center',
                        name=f'Centroid_{label}')

    # Save the plot as PNG
    png_output_path = os.path.join(output_directory, f'{model_no}_{feature_type}_umap.png')
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
  "byol_s_resnetish34",
  "passt_base"]  # List of feature types
    speaker_nos = [
"p278",
"p376",
"p265",
"p318",
"p272",
"p306",
"p239",
"p287",
"p262",
"p288",
"p284",
"p360",
"p251",
"p312",
"p282",]
    results = []

    for speaker_no in speaker_nos:
        train_type = "train"
        output_directory = f"/data/Deep_Fake_Data/umap_plots_centroids/{speaker_no}"  # Directory to save plots
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        for idx, feature_type in enumerate(feature_types):
            print(f"Processing feature type: {feature_type}",speaker_no)

            data_df = read_pickle_files(base_directory, feature_type, speaker_no, train_type)

            if not data_df.empty:
                visualize_umap(data_df, data_df['label'], feature_type, output_directory, str(idx).zfill(3), results, speaker_no)
            else:
                print(f"No data found for feature type: {feature_type}")

    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)
    results_output_path = os.path.join(base_directory, "umap_distances_summary.csv")
    results_df.to_csv(results_output_path, index=False)
    print(f"Results saved to {results_output_path}")