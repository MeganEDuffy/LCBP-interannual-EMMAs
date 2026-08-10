#################
# LOAD PACKAGES #
#################

library(tidyverse)
library(dataRetrieval)
library(lubridate)
library(grid)

plot_cumulative_winter_discharge <- function(site_number, 
                                             site_name, 
                                             output_dir, 
                                             panel_label = "a)", # New input parameter for panel label
                                             start_date = "1929-10-01", 
                                             end_date = "2025-01-03") {
  
  # --- 1. RETRIEVE DATA USING MODERN API ---
  qdat_raw <- read_waterdata_daily(
    monitoring_location_id = site_number,
    parameter_code         = "00060",
    time                   = c(start_date, end_date)
  )
  
  # --- 2. CLEAN & PREP DATA ---
  qdat <- qdat_raw %>%
    rename(
      Date  = time,
      Q_cfs = value
    ) %>%
    mutate(
      Date  = as.Date(Date),
      Year  = as.integer(format(Date, "%Y")),
      Qcms  = Q_cfs * 0.028316847 # Convert cfs to cms
    ) %>%
    filter(Date > as.Date("1928-09-30"))
  
  # Total daily discharge (cubic meters / day)
  qdat$tdQcmd <- qdat$Qcms * 86400 
  
  # Day of Water Year (DOWY) function starting Oct 1
  hydro.day.new <- function(x, start.month = 10L) {
    start.yr = year(x) - (month(x) < start.month)
    start.date = make_date(start.yr, start.month, 1L)
    as.integer(x - start.date + 1L)
  }
  
  qdat$DOWY <- hydro.day.new(qdat$Date)
  
  # --- 3. CALCULATE CUMULATIVE WATER YEAR DISCHARGE ---
  qdat <- qdat %>%
    group_by(Year) %>%
    mutate(CumWYQ = cumsum(tdQcmd)) %>%
    ungroup()
  
  # --- 4. SUBSET WINTER ENDPOINT (DOWY 182 = March 31) ---
  d182.win <- qdat %>% filter(DOWY == 182)
  
  # Fit linear model to get R-squared and p-value dynamically
  win_mod <- lm(CumWYQ ~ Year, data = d182.win)
  mod_summary <- summary(win_mod)
  
  r2_val  <- mod_summary$r.squared
  
  # Extract p-value from the F-statistic of the linear model
  f_stat <- mod_summary$fstatistic
  p_val  <- pf(f_stat[1], f_stat[2], f_stat[3], lower.tail = FALSE)
  
  # Format p-value string nicely
  p_text <- if (p_val < 0.001) "p < 0.001" else paste0("p = ", sprintf("%.3f", p_val))
  
  # --- 5. GENERATE PLOT ---
  p <- ggplot(d182.win, aes(x = Year, y = CumWYQ, color = Year)) + 
    geom_point(size = 2, alpha = 0.9) + 
    theme_bw() +
    labs(
      y = "Cumulative winter discharge (cubic meters)", 
      x = "Year"
    ) +
    geom_smooth(method = "lm", color = "black", linetype = "dashed", se = TRUE) + 
    scale_color_viridis_b(option = "viridis") +
    annotation_custom(
      textGrob(site_name, 
               x = unit(0.02, "npc"), y = unit(0.92, "npc"),
               just = c("left", "top"),
               gp = gpar(col = "black", fontsize = 14, fontface = "bold"))
    ) +
    annotation_custom(
      textGrob(sprintf("R2 = %.2f", r2_val), 
               x = unit(0.02, "npc"), y = unit(0.88, "npc"),
               just = c("left", "top"),
               gp = gpar(col = "black", fontsize = 14))
    ) +
    annotation_custom(
      textGrob(p_text, 
               x = unit(0.02, "npc"), y = unit(0.84, "npc"),
               just = c("left", "top"),
               gp = gpar(col = "black", fontsize = 14))
    ) +
    annotation_custom(
      textGrob(panel_label, 
               x = unit(0.02, "npc"), y = unit(0.98, "npc"),
               just = c("left", "top"),
               gp = gpar(col = "black", fontsize = 16, fontface = "bold"))
    ) +
    theme(
      text = element_text(size = 16),
      legend.position = "none"
    )
  
  # --- 6. SAVE PLOT ---
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  safe_name <- gsub(" ", "_", site_name)
  file_path <- file.path(output_dir, paste0(safe_name, "_cumulative_winter_discharge.jpg"))
  
  ggsave(plot = p, width = 5.2, height = 6, dpi = 300, file = file_path)
  
  return(p)
}