#################
# LOAD PACKAGES #
#################

library(tidyverse)
library(dataRetrieval)
library(lubridate)
library(grid)

plot_cumulative_winter_discharge <- function(site_number, 
                                             site_name, 
                                             panel_label = "a)", 
                                             start_date = "1929-10-01", 
                                             end_date = "2025-01-03") {
  
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
  # Define winter window: Dec 21 to Dec 31 OR Jan 1 to March 20
  # We associate the winter period with the calendar year in which it ends (e.g., Winter 2023 = Dec 2022-Mar 2023)
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
  
  # --- 5. STATS ---
  win_mod <- lm(TotalWinterQ ~ WinterYear, data = d_winter)
  mod_summary <- summary(win_mod)
  r2_val <- mod_summary$r.squared
  f_stat <- mod_summary$fstatistic
  p_val <- pf(f_stat[1], f_stat[2], f_stat[3], lower.tail = FALSE)
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
  
  return(p)
}