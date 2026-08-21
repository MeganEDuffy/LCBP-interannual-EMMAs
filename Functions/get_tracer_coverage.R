library(tidyverse)

get_tracer_coverage <- function(chem_data_file, site_name) {
  
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
  
  total_samples <- nrow(stream_df)
  
  if (total_samples == 0) {
    message(paste0("⚠️ No streamwater samples found for site: ", site_name))
    return(NULL)
  }
  
  # --- 2. SELECT SOLUTE COLUMNS (Matching correlation matrix logic) ---
  solute_cols <- grep("_mg_L$|^dD$|^d18O$", names(stream_df), value = TRUE)
  solute_cols <- solute_cols[!grepl("^(NO2|NO3|PO4)(_|$)", solute_cols, ignore.case = TRUE)]
  
  valid_solutes <- solute_cols[sapply(stream_df[solute_cols], function(x) sum(!is.na(x)) > 2)]
  
  # --- 3. CALCULATE COVERAGE PER TRACER ---
  coverage_list <- lapply(valid_solutes, function(tracer) {
    valid_count <- sum(!is.na(stream_df[[tracer]]))
    pct <- (valid_count / total_samples) * 100
    
    data.frame(
      Site = site_name,
      Tracer = tracer,
      Total_Samples = total_samples,
      Valid_Samples = valid_count,
      Coverage_Pct = pct
    )
  })
  
  coverage_df <- bind_rows(coverage_list) %>%
    arrange(desc(Coverage_Pct))
  
  return(coverage_df)
}