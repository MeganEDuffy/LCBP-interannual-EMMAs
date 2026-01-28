#########################################
# Python module to perform PCA and EMMA #
# Megan Duffy - Adair Lab, UVM ##########
# last updated 2026-01-19 ###############
#########################################

# Tracers are set up for Wade, Hungerford, and Potash Brooks

# Import libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

# Define event-specfic PCA plot function

def plot_event_pca(
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

    Parameters:
        data (DataFrame): Full dataframe containing stream and endmember samples.
        site (str): Site name ("Wade" or "Hungerford").
        start_date (str or datetime): Start of storm event (e.g., "2023-04-01").
        end_date (str or datetime): End of storm event (e.g., "2023-04-04").
        endmember_ids (list of str): List of Sample IDs to use as endmembers.
        title (str): Title for the plot.
    """

    # Site-specific tracers
    if site == "Wade":
        tracers = ['Ca_mg_L', 'Si_mg_L', 'Mg_mg_L', 'dD', 'd18O', 'Na_mg_L']
    elif site == "Hungerford":
        tracers = ['Ca_mg_L', 'Cl_mg_L', 'Si_mg_L', 'Na_mg_L', 'Mg_mg_L', 'dD', 'd18O']
    elif site == "Potash":
        #tracers = ['Ca_mg_L', 'Cl_mg_L', 'K_mg_L', 'Na_mg_L', 'Mg_mg_L', 'dD', 'd18O']
        tracers = ['Ca_mg_L', 'K_mg_L', 'Na_mg_L', 'Mg_mg_L', 'dD', 'd18O']
    else:
        raise ValueError("Site not recognized. Use 'Wade', 'Hungerford', or 'Potash'.")

    # Ensure datetime column is datetime type
    data['Datetime'] = (data['Date'] + ' ' + data['Time']) # Combine the strings of original inventory Date and Time cols
    data['Datetime'] = pd.to_datetime(data['Datetime'], format="%m/%d/%Y %H:%M", errors="coerce") 
    data = data[data['Datetime'].notna()] # NA dates (we have a couple in the RI23 dataset) not useful - prune 

    # Subset streamwater (mixture) in date range and site
    stream = data[
        (data["Site"] == site) &
        (data["Type"].isin(["Grab", "Grab/Isco", "Baseflow", "Isco"])) &
        (data["Datetime"] >= pd.to_datetime(start_date)) &
        (data["Datetime"] <= pd.to_datetime(end_date))
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
    subset_stream["Datetime"] = stream["Datetime"]

    # Fill NA for endmembers with mean (per tracer)
    subset_endmembers = endmembers[tracers].copy()
    subset_endmembers = subset_endmembers.fillna(subset_endmembers.mean())
    subset_endmembers["Group"] = "Endmember"
    subset_endmembers["Type"] = endmembers["Type"].values
    subset_endmembers["Datetime"] = endmembers["Datetime"].values

    # -----------------------
    # PCA logic 
    # -----------------------

    # 1. Fit scaler ONLY on streamwater using fit_transform
    scaler = StandardScaler()
    scaled_stream = scaler.fit_transform(subset_stream[tracers])

    # 2. PCA ONLY on scaled streamwater using pca.fit_transform
    pca = PCA(n_components=2)
    stream_pca_result = pca.fit_transform(scaled_stream)
    subset_stream["PC1"] = stream_pca_result[:, 0]
    subset_stream["PC2"] = stream_pca_result[:, 1]

    # 3. Project endmembers using same scaler & PCA using and transform and pca.transform
    scaled_endmembers = scaler.transform(subset_endmembers[tracers])
    endmember_pca_result = pca.transform(scaled_endmembers)
    subset_endmembers["PC1"] = endmember_pca_result[:, 0]
    subset_endmembers["PC2"] = endmember_pca_result[:, 1]

    # Combine for plotting
    combined = pd.concat([subset_stream, subset_endmembers], ignore_index=True)

    # -----------------------
    # Plotting
    # -----------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot streamwater points
    sw = combined[combined["Group"] == "Streamwater"]
    ax.scatter(sw["PC1"], sw["PC2"], marker='+', c='blue', alpha=0.5, label='Streamwater')

    # Plot endmembers with distinct markers/colors
    endmember_markers = {
        'Rain': 'o', 'Snow': 's', 'Snowmelt lysimeter': '^', 'Precip': '*',
        'Soil water lysimeter dry': '<', 'Soil water lysimeter wet': '>',
        'Groundwater': 'D', 'Baseflow': 'P'
    }
    colors = ['#d7191c', '#fdae61', '#abdda4', '#2b83ba', '#2ca25f', '#636363', '#8856a7', '#d95f0e']
    for (etype, color) in zip(endmember_markers.keys(), colors):
        em = combined[(combined["Type"] == etype)]
        if not em.empty:
            ax.scatter(em["PC1"], em["PC2"],
                       marker=endmember_markers[etype], c=color, edgecolors='black',
                       alpha=0.8, s=100, label=etype)
            for _, row in em.iterrows():
                if pd.notnull(row["Datetime"]):
                    ax.text(row["PC1"], row["PC2"], row["Datetime"].strftime('%m/%d'),
                            fontsize=12, ha='right')

    # Variance explained for axis labels
    pc1_var = pca.explained_variance_ratio_[0] * 100
    pc2_var = pca.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f"PC1 ({pc1_var:.1f}%)")
    ax.set_ylabel(f"PC2 ({pc2_var:.1f}%)")

    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1.02), loc="upper left")
    plt.tight_layout()
    plt.show()

    # Return info
    return tracers, pca

#########################
# STEP 2: EMMA FUNCTION #
#########################

def run_emma_event(data, site, start_date, end_date, endmember_ids, n_components=2):
    """
    Perform PCA-based End-Member Mixing Analysis (EMMA) for a storm event.
    
    Parameters:
        data (DataFrame): Full dataframe with streamwater and endmember data
        site (str): Site name ("Wade" or "Hungerford")
        start_date (str): Event start date (e.g., '2023-04-01')
        end_date (str): Event end date (e.g., '2023-04-04')
        endmember_ids (list of str): List of sample IDs to use as endmembers
        n_components (int): Number of principal components (default = 2)

    Returns:
        fractions_df (DataFrame): Streamwater samples with source fractions
    """
    
    # Site-specific tracer selection
    if site == "Wade":
        tracers = ['Ca_mg_L', 'Si_mg_L', 'Mg_mg_L', 'dD', 'd18O', 'Na_mg_L']
    elif site == "Hungerford":
        tracers = ['Ca_mg_L', 'Cl_mg_L', 'Si_mg_L', 'Na_mg_L', 'Mg_mg_L', 'dD', 'd18O']
    elif site == "Potash":
        #tracers = ['Ca_mg_L', 'Cl_mg_L', 'K_mg_L', 'Na_mg_L', 'Mg_mg_L', 'dD', 'd18O']
        tracers = ['Ca_mg_L', 'K_mg_L', 'Na_mg_L', 'Mg_mg_L', 'dD', 'd18O']
    else:
        raise ValueError("Site not recognized. Use 'Wade', 'Hungerford', or 'Potash'.")

    # Ensure datetime format
    #data["Date"] = pd.to_datetime(data["Date"], format="%m/%d/%Y", errors="coerce") #OLD

    # Ensure datetime column is datetime type
    data['Datetime'] = (data['Date'] + ' ' + data['Time']) # Combine the strings of original inventory Date and Time cols
    data['Datetime'] = pd.to_datetime(data['Datetime'], format="%m/%d/%Y %H:%M", errors="coerce") 
    data = data[data['Datetime'].notna()] # NA dates (we have a couple in the RI23 dataset) not useful - prune 

    # --- Subset streamwater and endmembers ---
    stream = data[
        (data["Site"] == site) &
        (data["Type"].isin(["Grab", "Grab/Isco", "Baseflow", "Isco"])) &
        (data["Datetime"] >= pd.to_datetime(start_date)) &
        (data["Datetime"] <= pd.to_datetime(end_date))
    ].copy()

    endmembers = data[
        (data["Site"] == site) &
        (data["Sample ID"].isin(endmember_ids))
    ].copy()

    # Clean + prepare
    stream_clean = stream[tracers].dropna().copy()
    stream_clean["Group"] = "Streamwater"
    stream_clean["Datetime"] = stream["Datetime"]

    end_clean = endmembers[tracers].copy()
    end_clean = end_clean.fillna(end_clean.mean())
    end_clean["Group"] = "Endmember"
    end_clean["Datetime"] = endmembers["Datetime"].values
    end_clean["Type"] = endmembers["Type"].values

    combined = pd.concat([stream_clean, end_clean], ignore_index=True)

    # --- PCA on combined data ---
    scaler = StandardScaler()
    scaled = scaler.fit_transform(combined[tracers])

    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled)

    combined["PC1"] = pca_result[:, 0]
    combined["PC2"] = pca_result[:, 1]

    # --- Separate stream and endmembers in PC space ---
    pc_cols = [f"PC{i+1}" for i in range(n_components)]
    stream_pcs = combined[combined["Group"] == "Streamwater"][pc_cols].values
    endmember_pcs = combined[combined["Group"] == "Endmember"][pc_cols].values
    endmember_labels = combined[combined["Group"] == "Endmember"]["Type"].values

    # --- EMMA optimization ---
    def objective(Ii, xi, B):
        xi_pred = np.dot(Ii, B)
        return np.linalg.norm(xi - xi_pred)

    constraints = (
        {'type': 'eq', 'fun': lambda Ii: np.sum(Ii) - 1},
        {'type': 'ineq', 'fun': lambda Ii: Ii},
        {'type': 'ineq', 'fun': lambda Ii: 1 - Ii},
    )

    fractions = []
    for xi in stream_pcs:
        init_guess = np.ones(endmember_pcs.shape[0]) / endmember_pcs.shape[0]
        result = minimize(objective, init_guess, args=(xi, endmember_pcs), constraints=constraints, method='SLSQP')
        fractions.append(result.x if result.success else np.nan)

    fractions = np.vstack(fractions)

    def project_simplex_nonneg(vec):
        vec = np.where(vec < 0, 0, vec)          # set negatives to zero
        s = vec.sum()
        if s == 0:
            return np.ones_like(vec) / len(vec)  # all zero? equal fractions
        return vec / s

    fractions = np.apply_along_axis(project_simplex_nonneg, 1, fractions)

    # --- Assemble output DataFrame ---
    fraction_cols = list(endmember_labels)
    stream_info = stream.reset_index(drop=True)[['Sample ID', 'Datetime', 'Site']]
    fractions_df = pd.concat([stream_info, pd.DataFrame(fractions, columns=fraction_cols)], axis=1)
    fractions_df["Sum_Fractions"] = fractions_df[fraction_cols].sum(axis=1)

    #return fractions_df, etc
    return fractions_df, scaler, pca, end_clean

##########################
# STEP 3. ERROR FUNCTION #
##########################

def predict_tracers_from_fractions(fractions_df, endmembers_df, tracer_cols):
    """
    Use EMMA fractions to predict tracer concentrations in streamwater samples.

    Parameters:
        fractions_df (DataFrame): Output from run_emma_event_fuss (includes fractions)
        endmembers_df (DataFrame): Subset of endmember data used in EMMA
        tracer_cols (list of str): List of tracer column names

    Returns:
        predicted_df (DataFrame): DataFrame with predicted tracer concentrations
    """
    # Unique endmember types used
    endmember_types = [col for col in fractions_df.columns if col in endmembers_df['Type'].unique()]

    # Get endmember mean tracer values by type
    em_means = endmembers_df.groupby("Type")[tracer_cols].mean()

    # Matrix of shape (n_endmembers, n_tracers)
    C = em_means.loc[endmember_types].values

    # Fractions matrix (n_samples, n_endmembers)
    F = fractions_df[endmember_types].values

    # Predicted concentrations (n_samples x n_tracers)
    predicted = np.dot(F, C)

    predicted_df = pd.DataFrame(predicted, columns=[f"{t}_predicted" for t in tracer_cols])
    predicted_df["Sample ID"] = fractions_df["Sample ID"]
    predicted_df["Datetime"] = fractions_df["Datetime"]

    return predicted_df

##############################################
# STEP 4. PLOT PREDICTED VS OBSERVED TRACERS #
##############################################

def plot_observed_vs_predicted(title, predicted_df, stream_df, tracer_cols):
    """
    Plot observed vs predicted tracer concentrations, with R² and RMSE annotations.

    Parameters:
        predicted_df (DataFrame): Output from predict_tracers_from_fractions
        stream_df (DataFrame): Original streamwater data
        tracer_cols (list): List of tracer names
    """
    import matplotlib.pyplot as plt

    # 🔹 Increase global font size for all plot elements
    plt.rcParams.update({"font.size": 16})  

    # Merge observed and predicted data
    merged = pd.merge(predicted_df, stream_df, on=["Sample ID", "Datetime"])

    n = len(tracer_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))

    if n == 1:
        axes = [axes]

    for i, tracer in enumerate(tracer_cols):
        ax = axes[i]

        # Drop rows with missing values for this tracer
        tmp = merged[[tracer, f"{tracer}_predicted"]].dropna()

        # Scatter observed vs predicted
        ax.scatter(tmp[tracer], tmp[f"{tracer}_predicted"], alpha=0.7)

        # 1:1 line
        lims = [tmp.min().min(), tmp.max().max()]
        ax.plot(lims, lims, 'k--')
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        # Regression + metrics
        if len(tmp) > 1:  # need at least 2 points
            X = tmp[[tracer]].values.reshape(-1, 1)
            y = tmp[f"{tracer}_predicted"].values

            # R²
            model = LinearRegression().fit(X, y)
            r2 = model.score(X, y)

            # RMSE
            mse = mean_squared_error(tmp[tracer], tmp[f"{tracer}_predicted"])
            rmse = np.sqrt(mse)

            # Add text box with R² and RMSE
            ax.text(
                0.05, 0.95,
                f"R² = {r2:.2f}\nRMSE = {rmse:.2f}",
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=14,  # slightly larger than base
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
            )

        ax.set_title(tracer, fontsize=18)
        ax.set_xlabel("Observed", fontsize=16)
        ax.set_ylabel("Predicted", fontsize=16)

    plt.suptitle(title, fontsize=20, y=1.05)
    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs("output/EMMA-error-plots", exist_ok=True)

    # Create safe filename from title
    safe_title = "".join(c if c.isalnum() or c in (" ", "_", "-") else "_" for c in title)
    filename = os.path.join("output/EMMA-error-plots", f"{safe_title}.png")

    # Save the figure
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"✅ Figure saved to {filename}")

def plot_emma_fractions_with_hydrograph(
    fractions_df,
    discharge_csv,
    start_date,
    end_date,
    site,
    discharge_time_col="Timestamp (UTC-04:00)",
    discharge_value_col="Value",
    snowpack_csv=None,
    snowpack_time_col=None,
    snowpack_depth_col="snowpack_cm",
    title=None
):
    """
    Plot snowpack depth (top), stream discharge (middle),
    and EMMA source fractions (bottom) for a storm event.
    """

    # ------------------------------------------------------------------
    # Read and prep discharge
    # ------------------------------------------------------------------
    Q = pd.read_csv(discharge_csv, comment="#")
    Q[discharge_time_col] = pd.to_datetime(Q[discharge_time_col])
    Q = Q.rename(columns={discharge_value_col: "Discharge_cms"})

    Q = Q[
        (Q[discharge_time_col] >= pd.to_datetime(start_date)) &
        (Q[discharge_time_col] <= pd.to_datetime(end_date))
    ].copy()

    # ------------------------------------------------------------------
    # Read and prep snowpack (optional)
    # ------------------------------------------------------------------
    if snowpack_csv is not None:
        snow = pd.read_csv(snowpack_csv)
        snow[snowpack_time_col] = pd.to_datetime(snow[snowpack_time_col])

        snow = snow[
            (snow[snowpack_time_col] >= pd.to_datetime(start_date)) &
            (snow[snowpack_time_col] <= pd.to_datetime(end_date))
        ].copy()
    else:
        snow = None

    # ------------------------------------------------------------------
    # Prep EMMA fractions
    # ------------------------------------------------------------------
    frac = fractions_df.copy()
    frac["Datetime"] = pd.to_datetime(frac["Datetime"])

    frac = frac[
        (frac["Datetime"] >= pd.to_datetime(start_date)) &
        (frac["Datetime"] <= pd.to_datetime(end_date))
    ].sort_values("Datetime")

    fraction_cols = [
        c for c in frac.columns
        if c not in ["Sample ID", "Datetime", "Site", "Sum_Fractions"]
    ]

    # ------------------------------------------------------------------
    # Define stack order and colors
    # ------------------------------------------------------------------
    baseflow_keys = ["Baseflow", "Groundwater"]
    soil_keys = ["Soil water lysimeter dry", "Soil water lysimeter wet", "Soil water"]
    melt_keys = ["Snowmelt lysimeter", "Meltwater", "Snowmelt", "Snow"]

    def match_cols(keys):
        return [c for c in fraction_cols if c in keys]

    baseflow_cols = match_cols(baseflow_keys)
    soil_cols = match_cols(soil_keys)
    melt_cols = match_cols(melt_keys)

    stacked_cols = baseflow_cols + soil_cols + melt_cols

    if len(stacked_cols) == 0:
        raise ValueError("No recognizable EMMA fraction columns found.")

    # Gets rid of NaN column stacks so no gaps in fraction plot
    frac = frac.dropna(subset=stacked_cols)


    color_map = {}
    for c in baseflow_cols:
        color_map[c] = "steelblue"
    for c in soil_cols:
        color_map[c] = "firebrick"
    for c in melt_cols:
        color_map[c] = "gold" if c != "Snow" else "yellow"

    stack_colors = [color_map[c] for c in stacked_cols]

    # ------------------------------------------------------------------
    # Plot layout (3 panels)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        3, 1,
        figsize=(13, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 2, 1]}
    )

    ax_snow, ax_q, ax_f = axes

    # ------------------------------------------------------------------
    # Snowpack depth (top)
    # ------------------------------------------------------------------
    if snow is not None:
        ax_snow.plot(
            snow[snowpack_time_col],
            snow[snowpack_depth_col],
            color="navy",
            linewidth=2
        )
        ax_snow.set_ylabel("Snowpack depth (cm)")
        ax_snow.set_title(
            title if title else f"{site}: snowpack, discharge, and EMMA fractions"
        )
        ax_snow.grid(True, alpha=0.3)
    else:
        ax_snow.set_visible(False)

    # ------------------------------------------------------------------
    # Hydrograph (middle)
    # ------------------------------------------------------------------
    ax_q.plot(
        Q[discharge_time_col],
        Q["Discharge_cms"],
        color="black",
        linewidth=2
    )
    ax_q.set_ylabel("Discharge (m³ s⁻¹)")
    ax_q.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # EMMA fractions (bottom)
    # ------------------------------------------------------------------
    ax_f.stackplot(
        frac["Datetime"],
        [frac[c] for c in stacked_cols],
        labels=stacked_cols,
        colors=stack_colors,
        alpha=0.9
    )

    ax_f.set_ylabel("Source fraction")
    ax_f.set_ylim(0, 1)
    ax_f.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.8),
        ncol=3,
        frameon=False
    )
    ax_f.grid(True, alpha=0.3)

    ax_f.set_xlabel("")
    ax_f.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()