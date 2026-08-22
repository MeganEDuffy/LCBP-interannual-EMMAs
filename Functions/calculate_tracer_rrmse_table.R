library(tidyverse)

calculate_tracer_rrmse_table <- function(chem_data_file, 
                                          site_name, 
                                          max_dims = 6, 
                                          selected_tracers = NULL,
                                          output_dir = NULL) {
  
  # --- 1. READ AND CLEAN DATA ---
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
  
  # --- 2. SELECT SOLUTE COLUMNS ---
  if (is.null(selected_tracers)) {
    # Default auto-selection (excluding NO2, NO3, PO4)
    solute_cols <- grep("_mg_L$|^dD$|^d18O$", names(stream_df), value = TRUE)
    solute_cols <- solute_cols[!grepl("^(NO2|NO3|PO4)(_|$)", solute_cols, ignore.case = TRUE)]
    valid_solutes <- solute_cols[sapply(stream_df[solute_cols], function(x) sum(!is.na(x)) > 5)]
  } else {
    # Use user-supplied tracers, ensuring they exist in the dataset
    valid_solutes <- intersect(selected_tracers, names(stream_df))
    missing_tracers <- setdiff(selected_tracers, names(stream_df))
    if (length(missing_tracers) > 0) {
      warning(paste("⚠️ The following requested tracers were not found in the dataset and were excluded:", 
                    paste(missing_tracers, collapse = ", ")))
    }
  }
  
  sub_df <- stream_df %>% select(all_of(valid_solutes))
  complete_idx <- complete.cases(sub_df)
  plot_data <- sub_df[complete_idx, ]
  n_samples <- nrow(plot_data)
  
  message(sprintf("ℹ️ Site: %s | Computing RRMSE for dimensions 1 to %d across n = %d complete samples (%d tracers)", 
                  site_name, max_dims, n_samples, length(valid_solutes)))
  
  if (n_samples < (max_dims + 1)) {
    warning("⚠️ Too few complete samples for the requested maximum number of dimensions!")
    return(NULL)
  }
  
  # --- 3. STANDARDIZE DATA (Z-scores for EMMA PCA) ---
  scaled_data <- scale(plot_data)
  means <- attr(scaled_data, "scaled:center")
  sds <- attr(scaled_data, "scaled:scale")
  
  # --- 4. PCA (Eigenvalues/Eigenvectors) ---
  pca_res <- prcomp(scaled_data, center = FALSE, scale. = FALSE)
  
  # --- 5. ITERATE THROUGH DIMENSIONS 1 TO MAX_DIMS ---
  all_results <- list()
  
  for (m in 1:max_dims) {
    loadings <- pca_res$rotation[, 1:m, drop = FALSE]
    scores <- pca_res$x[, 1:m, drop = FALSE]
    
    # Orthogonal projection onto m dimensions
    z_hat <- scores %*% t(loadings)
    
    # Back-scale to original units
    x_hat <- matrix(NA, nrow = nrow(z_hat), ncol = ncol(z_hat))
    colnames(x_hat) <- colnames(plot_data)
    
    for (j in seq_along(valid_solutes)) {
      x_hat[, j] <- z_hat[, j] * sds[j] + means[j]
    }
    
    # Calculate RRMSE (as ratio) for each tracer at dimension m
    for (tracer in valid_solutes) {
      obs <- plot_data[[tracer]]
      proj <- x_hat[, tracer]
      
      obs_mean <- mean(obs, na.rm = TRUE)
      rmse <- sqrt(mean((obs - proj)^2))
      rrmse_val <- rmse / abs(obs_mean)
      
      all_results[[length(all_results) + 1]] <- data.frame(
        Site = site_name,
        Tracer = tracer,
        Dimension = paste0("m_", m),
        RRMSE = rrmse_val
      )
    }
  }
  
  # --- 6. RESHAPE TO WIDE FORMAT (Table 3 Style) ---
  long_df <- bind_rows(all_results)
  
  wide_df <- long_df %>%
    pivot_wider(names_from = Dimension, values_from = RRMSE) %>%
    mutate(N_Samples = n_samples) %>%
    select(Site, Tracer, N_Samples, everything())
  
  # --- 7. SAVE TO CSV IF OUTPUT DIRECTORY IS PROVIDED ---
  if (!is.null(output_dir)) {
    dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
    safe_site_name <- gsub(" ", "_", site_name)
    file_path <- file.path(output_dir, paste0(safe_site_name, "_rrmse_table.csv"))
    write.csv(wide_df, file_path, row.names = FALSE)
    #message(sprintf("✅ RRMSE table successfully saved to: %s", file_path))
  }
  
  return(wide_df)
}