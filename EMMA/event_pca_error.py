############################################
# Python module to perform single evnt PCA #
# Megan Duffy - Adair Lab, UVM #############
# last updated 2026-01-19 ##################
############################################

# Event_pca_error.py

# Tracers are set up for Wade, Hungerford, and Potash Brooks
# Error bars are +/- 1 STDV given multiple endmembers of the same type (i.e., three snowmelt lysimeter samples)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull

def plot_event_pca_with_avgEM(
    data,
    site,
    start_date,
    end_date,
    endmember_ids,
    title="Event-Specific PCA"
):
    """
    Generate PCA plot for a specific storm event, following the EMMALAB workflow:
    PCA is fit ONLY on streamwater (mixture) data, then endmembers are projected
    into that PCA space.
    Also plots AvgEMScore: mean PC score for each endmember type.
    """

    # Site-specific tracers
    if site == "Wade":
        tracers = ['Ca_mg_L', 'Si_mg_L', 'Mg_mg_L', 'dD', 'd18O', 'Na_mg_L']
    elif site == "Hungerford":
        tracers = ['Ca_mg_L', 'Cl_mg_L', 'Si_mg_L', 'Na_mg_L', 'Mg_mg_L', 'dD', 'd18O']
    elif site == "Potash":
        tracers = ['Ca_mg_L', 'Cl_mg_L', 'K_mg_L', 'Na_mg_L', 'Mg_mg_L', 'dD', 'd18O']
    else:
        raise ValueError("Site not recognized. Use 'Wade', 'Potash', or 'Hungerford'.")

    # Ensure datetime column is datetime type
    data["Date"] = pd.to_datetime(data["Date"], format="%m/%d/%Y", errors="coerce")

    # Subset streamwater (mixture) in date range and site
    stream = data[
        (data["Site"] == site) &
        (data["Type"].isin(["Grab", "Grab/Isco", "Baseflow", "Isco"])) &
        (data["Date"] >= pd.to_datetime(start_date)) &
        (data["Date"] <= pd.to_datetime(end_date))
    ].copy()

    # Subset endmembers by Sample ID
    endmembers = data[
        (data["Site"] == site) &
        (data["Sample ID"].isin(endmember_ids))
    ].copy()

    # Drop NA for stream tracers
    subset_stream = stream[tracers].dropna().copy()
    subset_stream["Group"] = "Streamwater"
    subset_stream["Type"] = "Streamwater"
    subset_stream["Date"] = stream["Date"]

    # Fill NA for endmembers with mean (per tracer)
    subset_endmembers = endmembers[tracers].copy()
    subset_endmembers = subset_endmembers.fillna(subset_endmembers.mean())
    subset_endmembers["Group"] = "Endmember"
    subset_endmembers["Type"] = endmembers["Type"].values
    subset_endmembers["Date"] = endmembers["Date"].values

    # -----------------------
    # EMMALAB PCA logic
    # -----------------------
    scaler = StandardScaler()
    scaled_stream = scaler.fit_transform(subset_stream[tracers])

    pca = PCA(n_components=2)
    stream_pca_result = pca.fit_transform(scaled_stream)
    subset_stream["PC1"] = stream_pca_result[:, 0]
    subset_stream["PC2"] = stream_pca_result[:, 1]

    scaled_endmembers = scaler.transform(subset_endmembers[tracers])
    endmember_pca_result = pca.transform(scaled_endmembers)
    subset_endmembers["PC1"] = endmember_pca_result[:, 0]
    subset_endmembers["PC2"] = endmember_pca_result[:, 1]

    # Combine
    combined = pd.concat([subset_stream, subset_endmembers], ignore_index=True)

    # -----------------------
    # Compute AvgEMScore + SD
    # -----------------------
    stats_em = (
        subset_endmembers.groupby("Type")[["PC1", "PC2"]]
        .agg(['mean', 'std'])
    )
    stats_em.columns = ['PC1_mean', 'PC1_std', 'PC2_mean', 'PC2_std']
    stats_em = stats_em.reset_index()
    stats_em["Group"] = "AvgEMScore"

    # -----------------------
    # Plotting
    # -----------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    # --- set font sizes globally (scale ~1.5× default) ---
    mpl.rcParams.update({
        "font.size": 18,          # base font size
        "axes.titlesize": 18,     # title
        "axes.labelsize": 18,     # x and y labels
        "xtick.labelsize": 14,    # tick labels
        "ytick.labelsize": 14,
        "legend.fontsize": 18
    })

    # Streamwater points
    sw = combined[combined["Group"] == "Streamwater"]
    ax.scatter(sw["PC1"], sw["PC2"], marker='+', c='blue', alpha=0.5, label='Streamwater')

    # Endmember markers/colors
    endmember_markers = {
        'Rain': 'o', 'Snow': 's', 'Snowmelt lysimeter': '^', 'Precip': '*',
        'Soil water lysimeter dry': '<', 'Soil water lysimeter wet': '>',
        'Groundwater': 'D', 'Baseflow': 'P'
    }
    colors = ['#d7191c', '#fdae61', '#abdda4', '#2b83ba',
              '#2ca25f', '#636363', '#8856a7', '#d95f0e']

    # Plot mean + error bars (instead of individual endmembers)
    for (etype, color) in zip(endmember_markers.keys(), colors):
        em_stat = stats_em[stats_em["Type"] == etype]
        if not em_stat.empty:
            ax.errorbar(
                em_stat["PC1_mean"], em_stat["PC2_mean"],
                xerr=em_stat["PC1_std"], yerr=em_stat["PC2_std"],
                fmt=endmember_markers[etype],  # marker style
                color=color, ecolor=color,
                elinewidth=1.5, capsize=4,
                markersize=12, markeredgecolor='black',
                label=f"{etype} mean ±1 SD"
            )

    # Draw mixing space polygon
    if len(stats_em) >= 3:
        from scipy.spatial import ConvexHull
        import numpy as np

        points = stats_em[["PC1_mean", "PC2_mean"]].values
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        hull_points = np.vstack([hull_points, hull_points[0]])

        ax.plot(hull_points[:,0], hull_points[:,1],
                linestyle='-', color='black', linewidth=1.5,
                label="Mixing space")
        ax.fill(hull_points[:,0], hull_points[:,1],
                facecolor='grey', alpha=0.1)

    # Variance explained
    pc1_var = pca.explained_variance_ratio_[0] * 100
    pc2_var = pca.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f"PC1 ({pc1_var:.1f}%)")
    ax.set_ylabel(f"PC2 ({pc2_var:.1f}%)")

    ax.set_title(title)

    # --- Save plot as 'output/site_title.jpg' ---
    # Sanitize filename: replace spaces and parentheses
    save_event_name = title.replace(" ", "_").replace("(", "").replace(")", "").replace(":", "")
    filename = f"{site}_{save_event_name}.jpg"
    output_path = os.path.join("/home/millieginty/OneDrive/git-repos/LCBP-interannual-EMMAs/", filename)

    fig.savefig(output_path, dpi=300)
    print(f"Saved plot to: {output_path}")
    
    ax.legend(bbox_to_anchor=(1.02, 1.02), loc="upper left")
    plt.tight_layout()
    plt.show()