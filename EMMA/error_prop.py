import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import t
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def calculate_pc_uncertainty(etype_data, tracers, scaler, pca, analytical_dict):
    """
    Propagates tracer uncertainty into PC space for a specific end-member type.
    Uses Genereux (1998) logic + PCA loadings.
    """
    # 1. Calculate Standard Deviation (s) of the end-member samples
    # If n=1, standard deviation is 0, and we rely on analytical uncertainty later
    s = etype_data[tracers].std().fillna(0)
    n = len(etype_data)
    
    # 2. Get Student's t-value for 95% confidence (Genereux Eq 4)
    # Using n-1 degrees of freedom; for n=1, we'll default to a high value or analytical
    t_val = t.ppf(0.975, max(n-1, 1)) if n > 1 else 1.0 

    # 3. Calculate W for each raw tracer (Eq 4 in Genereux)
    # If you have a 'literature' estimate, you'd add it here as a third term
    W_tracers = []
    for tracer in tracers:
        Wa = analytical_dict.get(tracer, 0)
        # Combined uncertainty: sqrt((t*s/sqrt(n))^2 + Wa^2)
        W = np.sqrt((t_val * s[tracer] / np.sqrt(n))**2 + Wa**2)
        # SCALE the uncertainty by the same factor used in PCA
        W_scaled = W / scaler.scale_[tracers.index(tracer)]
        W_tracers.append(W_scaled)
    
    W_tracers = np.array(W_tracers)

    # 4. Project W into PC space using PCA Loadings (Eigenvectors)
    # PC_uncert = sqrt(sum(loading_i^2 * W_scaled_i^2))
    loadings = pca.components_ # Shape (n_components, n_tracers)
    
    pc1_uncert = np.sqrt(np.sum((loadings[0, :]**2) * (W_tracers**2)))
    pc2_uncert = np.sqrt(np.sum((loadings[1, :]**2) * (W_tracers**2)))
    
    return pc1_uncert, pc2_uncert

def plot_event_pca_with_error(
    data, 
    site, 
    start_date, 
    end_date, 
    endmember_ids, 
    analytical_dict,  # <--- MAKE SURE THIS IS HERE
    title="Event-Specific PCA"
):
    # ... (Tracer selection and data subsetting logic remains the same as your code) ...
    
    # [Standard PCA setup]
    scaler = StandardScaler()
    scaled_stream = scaler.fit_transform(subset_stream[tracers])
    pca = PCA(n_components=2)
    stream_pca_result = pca.fit_transform(scaled_stream)
    
    # Project Endmembers
    scaled_endmembers = scaler.transform(subset_endmembers[tracers])
    endmember_pca_result = pca.transform(scaled_endmembers)
    subset_endmembers["PC1"] = endmember_pca_result[:, 0]
    subset_endmembers["PC2"] = endmember_pca_result[:, 1]

    # --- Compute PC Uncertainties using the Genereux-PCA bridge ---
    em_types = subset_endmembers['Type'].unique()
    error_data = []

    for etype in em_types:
        etype_data = subset_endmembers[subset_endmembers['Type'] == etype]
        mean_pc1 = etype_data['PC1'].mean()
        mean_pc2 = etype_data['PC2'].mean()
        
        # Propagate error
        w_pc1, w_pc2 = calculate_pc_uncertainty(etype_data, tracers, scaler, pca, analytical_uncertainties)
        
        error_data.append({
            'Type': etype,
            'PC1_mean': mean_pc1,
            'PC2_mean': mean_pc2,
            'PC1_err': w_pc1,
            'PC2_err': w_pc2
        })
    
    stats_em = pd.DataFrame(error_data)

    # --- Plotting with propagated error bars ---
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Streamwater as background
    ax.scatter(stream_pca_result[:,0], stream_pca_result[:,1], marker='+', c='blue', alpha=0.3, label='Stream')

    # Endmembers with propagated error
    colors = ['#d7191c', '#fdae61', '#abdda4', '#2b83ba', '#2ca25f', '#636363', '#8856a7', '#d95f0e']
    markers = ['o', 's', '^', '*', '<', '>', 'D', 'P']

    for i, row in stats_em.iterrows():
        ax.errorbar(row['PC1_mean'], row['PC2_mean'], 
                    xerr=row['PC1_err'], yerr=row['PC2_err'],
                    fmt=markers[i % len(markers)], color=colors[i % len(colors)],
                    label=f"{row['Type']} (Propagated Error)", markersize=10, capsize=5)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    return stats_em, pca, scaler