######################################################
# R function to plot stream Q, sC, grab sample times #
# Megan Duffy - Adair Lab, UVM #######################
# last updated 2026-06-19 ############################
######################################################

library(tidyverse)
library(dplyr)
library(lubridate)
library(data.table)

plot_samples_q <- function(site_name, 
                           plot_range, 
                           event_bounds, 
                           q_file, 
                           exo_file, 
                           grab_chem_file, 
                           q_lim,
                           sc_lim,
                           output_dir = NULL) {
  
  # --- 1. READ DATA ---
  dat_q    <- read.csv(q_file)
  dat_exo  <- read.csv(exo_file)
  dat_isco <- read.csv(grab_chem_file)
  
  # --- 2. CLEAN CONTINUOUS SENSOR DATA ---
  dat_exo <- dat_exo %>% 
    rename(sC = matches("Sp.Cond")) %>%
    mutate(timestamp = ymd_hms(timestamp, tz = "UTC"))
  
  dat_q <- dat_q %>% 
    rename(timestamp = datetime) %>%
    rename(q_cms = matches("q_cms")) %>% 
    mutate(timestamp = ymd_hms(timestamp, tz = "UTC"))
  
  # Dual-axis join using data.table for speed
  merged_dt <- as.data.table(dat_q)[as.data.table(dat_exo), roll = "nearest", on = .(timestamp)]
  
  # Subset sensor data to the plot window
  merged_dt_sub <- merged_dt %>%
    filter(timestamp >= plot_range[1] & timestamp <= plot_range[2])
  
  # --- 3. CLEAN GRAB/SAMPLE METADATA ---
  dat_isco_clean <- dat_isco %>%
    mutate(timestamp = mdy_hm(paste(Date, Time), tz = "UTC")) %>%
    # Filter for active site and time window
    filter(Site == site_name) %>%
    filter(timestamp >= plot_range[1] & timestamp <= plot_range[2])
  
  if (nrow(dat_isco_clean) == 0) {
    warning("No geochemical samples found within the specified time range for this site.")
  }
  
  # --- 4. PREP SHADING FOR STORM EVENTS ---
  shade_df <- data.frame(
    xmin = as.POSIXct(event_bounds$start, tz = "UTC"),
    xmax = as.POSIXct(event_bounds$end, tz = "UTC"),
    ymin = -Inf, ymax = Inf
  )
  
  # --- 5. PLOTTING LOGIC ---
  coeff_sc <- sc_lim[2] / q_lim[2]
  
  p <- ggplot() +
    # Yellow event shading in the background
    geom_rect(data = shade_df, aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), 
              fill = "yellow", alpha = 0.2) +
    
    # Continuous sensor lines
    geom_line(data = merged_dt_sub, aes(x = timestamp, y = q_cms, color = "Discharge"), linewidth = 0.8) +
    geom_line(data = merged_dt_sub, aes(x = timestamp, y = sC / coeff_sc, color = "Conductivity (sensor)"), linewidth = 0.8) +
    
    # Vertical dashed lines for exact sample collection times, mapped by "Type"
    geom_vline(data = dat_isco_clean, aes(xintercept = timestamp, color = Type), 
               linewidth = 0.7, linetype = "dashed") +
    
    # Labels and Dual Y-Axes scaling
    scale_y_continuous(name = "Q (cms)", limits = q_lim, expand = c(0,0),
                       sec.axis = sec_axis(~.*coeff_sc, name = "sC (uS/cm)")) +
    scale_x_datetime(limits = plot_range, date_labels = "%b %d\n%H:%M") +
    
    # Manual color override to safely blend continuous lines and discrete types
    scale_color_manual(values = c(
      "Discharge" = "blue", 
      "Conductivity (sensor)" = "red",
      "Baseflow" = "darkblue",
      "Grab/Isco" = "darkgreen",
      "Grab" = "forestgreen",
      "Isco" = "emerald",
      "Groundwater" = "purple",
      "Snow" = "cyan",
      "Snowmelt lysimeter" = "orange",
      "Soil water lysimeter dry" = "red",
      "Soil water lysimeter wet" = "red"
    ), aesthetics = "color") +
    
    labs(
      title = paste(site_name, "Brook: ISCO/Grab/end-member collections"),
      subtitle = "Dashed vertical lines indicate sample collection times, yellow shading = ISCO events",
      x = "Date",
      color = "Data Type / Sample Type"
    ) +
    theme_bw() + 
    theme(
      legend.position = "right",
      plot.title = element_text(face = "bold", size = 14)
    )
  
  # --- 6. AUTOMATED EXPORT ---
  if (!is.null(output_dir)) {
    if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
    
    file_name <- paste0(site_name, "_sample_timeline_", format(plot_range[1], "%Y%m%d"), ".jpg")
    out_path  <- file.path(output_dir, file_name)
    
    ggsave(filename = out_path, plot = p, width = 12, height = 6, dpi = 300)
    message("Successfully saved timeline plot to: ", out_path)
  }
  
  return(p)
}