#################
# LOAD PACKAGES #
#################

library(tidyverse)
library(ggplot2)

plot_hooper_pca_residuals <- function(chem_data_file,
                                      site_name,
                                      tracer_cols = c("Ca_mg_L", "Mg_mg_L", "Na_mg_L", "Cl_mg_L", 
                                                      "Si_mg_L", "dD", "d18O", "DOC_mg_L"),
                                      dimensions = c(1, 2, 3, 4, 5),
                                      output_dir = NULL,
                                      base_font_size = 10) {
  
  # --- 1. READ AND CLEAN STREAMWATER DATA ---
  raw_df <- read.csv(chem_data_file, stringsAsFactors = FALSE)
  
  stream_df <- raw_df %>%
    filter(Site == site_name) %>%
    mutate(
      Type_clean = case_when(
        Type %in% c("Grab", "Isco", "Grab/Isco", "Grab\\Isco", "Baseflow") ~ "Streamwater",
        TRUE ~ NA_character_
      )
    ) %>%
    filter(Type_clean == "Streamwater")
  
  # Verify which requested tracers exist in the dataset
  available_tracers <- intersect(tracer_cols, names(stream_df))
  missing_tracers <- setdiff(tracer_cols, available_tracers)
  
  if (length(missing_tracers) > 0) {
    message(sprintf("⚠️ Note: The following requested tracers were not found in the dataset: %s", 
                    paste(missing_tracers, collapse = ", ")))
  }
  
  # Extract matrix and drop rows with NAs to ensure valid PCA matrix reconstruction
  plot_data_raw <- stream_df %>%
    select(all_of(available_tracers)) %>%
    drop_na()
  
  n_samples <- nrow(plot_data_raw)
  message(sprintf("ℹ️ Site: %s | Running Hooper PCA residual analysis on n = %d complete streamwater rows across dimensions: %s.", 
                  site_name, n_samples, paste(dimensions, collapse = ", ")))
  
  # --- 2. STANDARDIZE (SCALE) DATA ---
  scaled_mat <- scale(plot_data_raw)
  scaled_df  <- as.data.frame(scaled_mat)
  
  # --- 3. LOOP THROUGH DIMENSIONAL SPACES AND COLLECT DATA ---
  all_plots_df <- list()
  
  for (k in dimensions) {
    if (k >= ncol(scaled_mat)) {
      message(sprintf("⚠️ Skipping %dD space: Dimension cannot exceed or equal total number of tracers (%d).", 
                      k, ncol(scaled_mat)))
      next
    }
    
    # Perform PCA on scaled matrix
    pca_fit <- prcomp(scaled_mat, center = FALSE, scale. = FALSE)
    
    # Extract scores and loadings for the first k dimensions
    scores   <- pca_fit$x[, 1:k, drop = FALSE]
    loadings <- pca_fit$rotation[, 1:k, drop = FALSE]
    
    # Reconstruct standardized data from the k-dimensional subspace
    reconstructed_mat <- scores %*% t(loadings)
    
    # Calculate PCA Residuals: Observed - Reconstructed
    residuals_mat <- scaled_mat - reconstructed_mat
    
    res_df <- as.data.frame(residuals_mat)
    colnames(res_df) <- paste0(available_tracers, "_res")
    
    # Reshape into long format
    obs_long <- scaled_df %>%
      mutate(id = row_number()) %>%
      pivot_longer(cols = all_of(available_tracers), names_to = "Tracer", values_to = "Observed_Scaled")
    
    res_long <- res_df %>%
      mutate(id = row_number()) %>%
      pivot_longer(cols = -id, names_to = "Tracer_Res", values_to = "Residual") %>%
      mutate(Tracer = gsub("_res$", "", Tracer_Res))
    
    plot_df <- left_join(obs_long, res_long, by = c("id", "Tracer")) %>%
      mutate(Dimension = paste0(k, "D"))
    
    all_plots_df[[as.character(k)]] <- plot_df
  }
  
  combined_plot_df <- bind_rows(all_plots_df) %>%
    mutate(Dimension = factor(Dimension, levels = paste0(dimensions, "D")))
  
  # --- 4. PLOT COMBINED GRID (Rows = Dimensions, Columns = Tracers) ---
  p <- ggplot(combined_plot_df, aes(x = Observed_Scaled, y = Residual)) +
    geom_point(alpha = 0.7, color = "darkblue", size = 1.5) +
    geom_hline(yintercept = 0, color = "blue", linewidth = 0.7, linetype = "dashed") +
    facet_grid(Dimension ~ Tracer, scales = "free") +
    theme_bw(base_size = base_font_size) +
    theme(
      panel.background = element_rect(fill = '#FFF5EE', colour = 'black'),
      strip.background = element_rect(fill = "grey90", color = "black"),
      strip.text = element_text(face = "bold")
    ) +
    labs(
      title = paste0(site_name, " Brook: residuals in 1-6D"),
      x = "Observed concentration (standardized z-score)",
      y = "PCA residual (observed - reconstructed)"
    )
  
  # --- 5. SAVE SINGLE COMBINED FIGURE ---
  if (!is.null(output_dir)) {
    dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
    file_path <- file.path(output_dir, paste0(site_name, "_residuals_1to", max(dimensions), "D.jpg"))
    
    n_tracers <- length(available_tracers)
    n_dims    <- length(unique(combined_plot_df$Dimension))
    
    ggsave(filename = file_path, plot = p, width = max(11, n_tracers * 1.6), height = max(7, n_dims * 2.0), dpi = 300)
    #message(paste0("✅ Saved combined residual plot to: ", file_path))
  }
  
  return(p)
}