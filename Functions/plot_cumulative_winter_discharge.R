#################
# LOAD PACKAGES #
#################

# Use r-usgs environment to run

library(tidyverse)
library(dataRetrieval)
library(lubridate)
library(grid)

plot_cumulative_winter_discharge <- function(site_number, 
                                             site_name, 
                                             panel_label = "a)", 
                                             start_date = "1929-10-01", 
                                             end_date = "2025-01-03",
                                             output_dir = NULL, ...) {
  
  # --- 1. RETRIEVE DATA ---
  qdat_raw <- read_waterdata_daily(
    monitoring_location_id = site_number,
    parameter_code         = "00060",
    time                   = c(start_date, end_date)
  )
  
  # --- 2. CLEAN & PREP DATA ---
  qdat <- qdat_raw %>%
    rename(Date = time, Q_cfs = value) %>%
    mutate(
      Date = as.Date(Date),
      Qcms = Q_cfs * 0.028316847,
      Year = year(Date),
      Month = month(Date),
      Day = day(Date)
    )
  
  # --- 3. FILTER FOR ASTRONOMICAL WINTER (Dec 21 - March 20) ---
  qdat <- qdat %>%
    mutate(
      WinterYear = case_when(
        Month == 12 & Day >= 21 ~ Year + 1,
        Month %in% c(1, 2) ~ Year,
        Month == 3 & Day <= 20 ~ Year,
        TRUE ~ NA_real_
      )
    ) %>%
    filter(!is.na(WinterYear))
  
  # --- 4. CALCULATE CUMULATIVE WINTER DISCHARGE ---
  d_winter <- qdat %>%
    group_by(WinterYear) %>%
    summarise(TotalWinterQ = sum(Qcms * 86400, na.rm = TRUE), .groups = "drop") %>%
    filter(!is.na(WinterYear))
  
  # --- 5. STATS & REGRESSION METRICS ---
  win_mod <- lm(TotalWinterQ ~ WinterYear, data = d_winter)
  mod_summary <- summary(win_mod)
  
  r2_val <- mod_summary$r.squared
  f_stat <- mod_summary$fstatistic
  p_val <- pf(f_stat[1], f_stat[2], f_stat[3], lower.tail = FALSE)
  
  # Extract slope and intercept for 1928 to 2025 calculations
  intercept <- coef(win_mod)[1]
  slope <- coef(win_mod)[2]
  
  pred_1928 <- intercept + slope * 1928
  pred_2025 <- intercept + slope * 2025
  abs_increase <- pred_2025 - pred_1928
  pct_increase <- (abs_increase / pred_1928) * 100
  
  # Create summary stats data frame for the table
  stats_df <- data.frame(
    Site = site_name,
    Site_Number = site_number,
    Abs_Increase_1928_2025_cms_days = abs_increase,
    Pct_Increase_1928_2025 = pct_increase,
    R_squared = r2_val,
    p_value = p_val,
    Slope = slope
  )
  
  p_text <- if (p_val < 0.001) "p < 0.001" else paste0("p = ", sprintf("%.3f", p_val))
  r2_expr <- bquote(R^2 == .(sprintf("%.2f", r2_val)))
  
  # --- 6. GENERATE PLOT ---
  p <- ggplot(d_winter, aes(x = WinterYear, y = TotalWinterQ, color = WinterYear)) + 
    geom_point(size = 2, alpha = 0.9) + 
    theme_bw() +
    labs(
      y = "Cum. winter discharge (cms)", 
      x = "Year"
    ) +
    geom_smooth(method = "lm", color = "black", linetype = "dashed", se = TRUE) + 
    scale_color_viridis_b(option = "viridis", name = "Year") +
    annotation_custom(
      textGrob(site_name, x = unit(0.02, "npc"), y = unit(0.92, "npc"),
               just = c("left", "top"), gp = gpar(col = "black", fontsize = 20))
    ) +
    annotation_custom(
      textGrob(as.expression(r2_expr), x = unit(0.02, "npc"), y = unit(0.87, "npc"),
               just = c("left", "top"), gp = gpar(col = "black", fontsize = 20))
    ) +
    annotation_custom(
      textGrob(p_text, x = unit(0.02, "npc"), y = unit(0.79, "npc"),
               just = c("left", "top"), gp = gpar(col = "black", fontsize = 20))
    ) +
    annotation_custom(
      textGrob(panel_label, x = unit(0.02, "npc"), y = unit(0.98, "npc"),
               just = c("left", "top"), gp = gpar(col = "black", fontsize = 20, fontface = "bold"))
    ) +
    theme(
      text = element_text(size = 16),
      axis.title.y = element_text(margin = margin(r = 15)),
      legend.position = "right"
    )
  
  # Return both the plot and the statistics table data frame
  return(list(plot = p, stats = stats_df))
}