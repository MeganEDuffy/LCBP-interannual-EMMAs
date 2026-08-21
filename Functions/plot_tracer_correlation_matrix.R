#################
# LOAD PACKAGES #
#################

library(tidyverse)
library(ggplot2)

plot_tracer_correlation_matrix <- function(chem_data_file, 
                                           site_name, 
                                           output_dir, 
                                           base_font_size = 12) {
  
  # --- 1. READ AND CLEAN DATA ---
  raw_df <- read.csv(chem_data_file, stringsAsFactors = FALSE)
  
  # Filter for streamwater sample types at the specified site
  stream_df <- raw_df %>%
    filter(Site == site_name) %>%
    mutate(
      Type_clean = case_when(
        Type %in% c("Grab", "Isco", "Grab/Isco", "Grab\\Isco", "Baseflow") ~ "Streamwater",
        TRUE ~ NA_character_
      )
    ) %>%
    filter(Type_clean == "Streamwater")
  
  # --- 2. SELECT SOLUTE COLUMNS (Excluding NO2, NO3, and PO4) ---
  solute_cols <- grep("_mg_L$|^dD$|^d18O$", names(stream_df), value = TRUE)
  solute_cols <- solute_cols[!grepl("^(NO2|NO3|PO4)(_|$)", solute_cols, ignore.case = TRUE)]
  
  valid_solutes <- solute_cols[sapply(stream_df[solute_cols], function(x) sum(!is.na(x)) > 2)]
  
  plot_data <- stream_df %>%
    select(all_of(valid_solutes))
  
  n_samples <- nrow(plot_data)
  message(paste0("ℹ️ Site: ", site_name, " | Computing correlation matrix for n = ", n_samples, " streamwater samples."))
  
  # --- 3. COMPUTE PEARSON CORRELATION & P-VALUES ---
  cor_matrix <- matrix(NA, nrow = length(valid_solutes), ncol = length(valid_solutes))
  pval_matrix <- matrix(NA, nrow = length(valid_solutes), ncol = length(valid_solutes))
  rownames(cor_matrix) <- colnames(cor_matrix) <- valid_solutes
  rownames(pval_matrix) <- colnames(pval_matrix) <- valid_solutes
  
  for (i in seq_along(valid_solutes)) {
    for (j in seq_along(valid_solutes)) {
      x <- plot_data[[valid_solutes[i]]]
      y <- plot_data[[valid_solutes[j]]]
      
      valid_idx <- !is.na(x) & !is.na(y)
      if (sum(valid_idx) > 2) {
        test_res <- cor.test(x[valid_idx], y[valid_idx], method = "pearson")
        cor_matrix[i, j] <- test_res$estimate
        pval_matrix[i, j] <- test_res$p.value
      }
    }
  }
  
  # --- 4. RESHAPE & FILTER FOR LOWER TRIANGLE (Excluding Diagonal & Upper Triangle) ---
  melt_cor <- as.data.frame(as.table(cor_matrix)) %>%
    rename(Var1 = Var1, Var2 = Var2, r_val = Freq)
  
  melt_pval <- as.data.frame(as.table(pval_matrix)) %>%
    rename(Var1 = Var1, Var2 = Var2, p_val = Freq)
  
  cor_df <- left_join(melt_cor, melt_pval, by = c("Var1", "Var2")) %>%
    mutate(
      idx1 = match(Var1, valid_solutes),
      idx2 = match(Var2, valid_solutes)
    ) %>%
    # Keep strictly the lower triangle (drops diagonal 1:1 and upper redundant matrix)
    filter(idx2 < idx1) %>%
    mutate(
      r2_val = r_val^2,
      high_collinear = (r2_val > 0.50 & p_val < 0.01)
    )
  
  # --- 5. BUILD THE CORRELATION MATRIX PLOT ---
  p_mat <- ggplot(cor_df, aes(x = Var1, y = Var2, fill = high_collinear)) +
    geom_tile(color = "white", linewidth = 0.5) +
    scale_fill_manual(
      values = c("TRUE" = "#ff9999", "FALSE" = "whitesmoke"), 
      name = "Collinearity Flag\n(R² > 0.50 & p < 0.01)",
      labels = c("TRUE" = "Exceeds Threshold", "FALSE" = "Below Threshold")
    ) +
    geom_text(aes(label = sprintf("r = %.2f\n(p = %.3f)", r_val, p_val)), size = 5, color = "black", na.rm = TRUE) +
    theme_minimal(base_size = base_font_size) +
    theme(
      axis.title = element_blank(),
      axis.text.x = element_text(angle = 45, hjust = 1, face = "bold"),
      axis.text.y = element_text(face = "bold"),
      panel.grid = element_blank(),
      legend.position = "bottom"
    ) +
    labs(
      title = paste0(site_name, " Brook: Pearson correlation matrix (n = ", n_samples, ")")
    )
  
  # --- 6. SAVE PLOT ---
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  safe_site_name <- gsub(" ", "_", site_name)
  file_path <- file.path(output_dir, paste0(safe_site_name, "_tracer_correlation_matrix.jpg"))
  
  n_tracers <- length(valid_solutes)
  plot_dim <- max(8, n_tracers * 1.2)
  
  ggsave(filename = file_path, plot = p_mat, width = plot_dim, height = plot_dim + 1, dpi = 300)
  
  return(list(plot = p_mat, data = cor_df, n_samples = n_samples))
}