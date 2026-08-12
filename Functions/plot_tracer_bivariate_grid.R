#################
# LOAD PACKAGES #
#################

library(tidyverse)
library(ggplot2)
library(GGally)

plot_tracer_bivariate_grid <- function(chem_data_file, 
                                      site_name, 
                                      output_dir, 
                                      base_font_size = 11) {
  
  # --- 1. READ AND CLEAN DATA ---
  raw_df <- read.csv(chem_data_file, stringsAsFactors = FALSE)
  
  # Filter for the specified site and streamwater sample types
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
  solute_cols <- grep("_mg_L$|^dD$|^d18O$", names(stream_df), value = TRUE)
  
  # Keep columns that have at least 3 valid numerical observations
  valid_solutes <- solute_cols[sapply(stream_df[solute_cols], function(x) sum(!is.na(x)) > 2)]
  
  # Keep data as-is (do NOT use drop_na() here, allowing pairwise deletion)
  plot_data <- stream_df %>%
    select(all_of(valid_solutes))
  
  # Total row count in streamwater samples
  n_samples <- nrow(plot_data)
  
  message(paste0("ℹ️ Site: ", site_name, " | Evaluated ", n_samples, " total streamwater rows with pairwise completeness."))
  
  # --- 3. CUSTOM LOWER TRIANGLE FUNCTION (Pairwise complete stats) ---
  dropdown_stats <- function(data, mapping, ...) {
    p <- ggplot(data = data, mapping = mapping) +
      geom_point(alpha = 0.6, color = "darkblue", size = 1.5, na.rm = TRUE) +
      geom_smooth(method = "lm", formula = y ~ x, color = "firebrick", se = FALSE, linewidth = 0.8, na.rm = TRUE)
    
    # Extract x and y vector data
    x_var <- rlang::as_name(mapping$x)
    y_var <- rlang::as_name(mapping$y)
    x_val <- eval(substitute(data$VAR, list(VAR = as.name(x_var))))
    y_val <- eval(substitute(data$VAR, list(VAR = as.name(y_var))))
    
    # Pairwise complete filtering for stats calculation
    valid_idx <- !is.na(x_val) & !is.na(y_val)
    x_clean <- x_val[valid_idx]
    y_clean <- y_val[valid_idx]
    
    if (length(x_clean) > 2) {
      fit <- lm(y_clean ~ x_clean)
      f_stat <- summary(fit)$fstatistic
      
      if (!is.null(f_stat)) {
        p_val <- pf(f_stat[1], f_stat[2], f_stat[3], lower.tail = FALSE)
        r2 <- summary(fit)$r.squared
        
        p_text <- if (p_val < 0.001) "p < 0.001" else paste0("p = ", sprintf("%.3f", p_val))
        r2_text <- paste0("R² = ", sprintf("%.2f", r2))
        
        p <- p + annotate("text", x = -Inf, y = Inf, label = paste0(r2_text, "\n", p_text),
                          hjust = -0.05, vjust = 1.2, size = 3, fontface = "plain", color = "black")
      }
    }
    return(p)
  }
  
  # --- 4. BUILD THE BIVARIATE GRID ---
  p_grid <- ggpairs(
    plot_data,
    lower = list(continuous = dropdown_stats),
    diag  = list(continuous = wrap("densityDiag", fill = "skyblue3", alpha = 0.6, na.rm = TRUE)),
    upper = list(continuous = wrap("cor", size = 4, fontface = "bold", use = "pairwise.complete.obs"))
  ) +
    theme_bw(base_size = base_font_size) +
    theme(
      panel.background = element_rect(fill = '#FFF5EE', colour = 'black'),
      strip.background = element_rect(fill = "grey90", color = "black"),
      strip.text = element_text(face = "bold", size = base_font_size)
    ) +
    labs(
      title = paste0(site_name, " Brook: Bivariate Solute-Solute Matrix (Max n = ", n_samples, ")")
    )
  
  # --- 5. SAVE PLOT ---
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  safe_site_name <- gsub(" ", "_", site_name)
  file_path <- file.path(output_dir, paste0(safe_site_name, "_tracer_bivariate_grid.jpg"))
  
  n_tracers <- length(valid_solutes)
  plot_dim <- max(8, n_tracers * 1.8)
  
  ggsave(filename = file_path, plot = p_grid, width = plot_dim, height = plot_dim, dpi = 300)
  message(paste0("✅ Bivariate grid successfully saved to: ", file_path))
  
  return(list(plot = p_grid, n_samples = n_samples, valid_tracers = valid_solutes))
}