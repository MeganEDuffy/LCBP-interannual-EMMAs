#########################################################
# R function to use continuous EXO  sC for mass balance #
# Megan Duffy - Adair Lab, UVM ##########################
# last updated 2026-06-16 ###############################
#########################################################

#################
# LOAD PACKAGES #
#################

library(tidyverse)
library(dplyr)
library(lubridate)
library(data.table)
library(patchwork)

plot_stream_cmb <- function(site_name, 
                            plot_range, 
                            event_bounds, 
                            q_file, 
                            exo_file, 
                            q_lim,
                            sc_lim,
                            sc_new_val = 10 # Default parameter for rain/melt
) {
  
  # --- 1. READ DATA ---
  dat_q   <- read.csv(q_file)
  dat_exo <- read.csv(exo_file)
  
  # --- 2. CLEAN CONTINUOUS DATA ---
  dat_exo <- dat_exo %>% 
    rename(sC = matches("Sp.Cond")) %>%
    mutate(timestamp = ymd_hms(timestamp, tz = "UTC"))
  
  dat_q <- dat_q %>% 
    rename(timestamp = datetime) %>%
    rename(q_cms = matches("q_cms")) %>% 
    mutate(timestamp = ymd_hms(timestamp, tz = "UTC"))
  
  # Join Q and EXO using data.table
  merged_dt <- as.data.table(dat_q)[as.data.table(dat_exo), roll = "nearest", on = .(timestamp)]
  
  # Filter data to the user-specified plotting window
  merged_dt <- merged_dt %>%
    filter(timestamp >= plot_range[1] & timestamp <= plot_range[2]) %>%
    arrange(timestamp)
  
  # --- 3. CONDUCTIVITY MASS BALANCE MATH ---
  # Dynamically pull the stream's initial sC value within the window as sC_old
  sc_old_val <- merged_dt$sC[1]
  
  cmb_data <- merged_dt %>%
    mutate(
      sC_old = sc_old_val,
      sC_new = sc_new_val,
      # Standard two-component mixing equation
      f_old = (sC - sC_new) / (sC_old - sC_new),
      f_new = 1 - f_old,
      # Handle minor sensor noise by bounding fractions between 0 and 1
      f_old = pmax(0, pmin(1, f_old)),
      f_new = 1 - f_old,
      # Deconvolute hydrograph components (cms)
      q_old = q_cms * f_old,
      q_new = q_cms * f_new
    )
  
  # --- 4. PREP SHADING DATA ---
  shade_df <- data.frame(
    xmin = as.POSIXct(event_bounds$start, tz = "UTC"),
    xmax = as.POSIXct(event_bounds$end, tz = "UTC"),
    ymin = -Inf, ymax = Inf
  )
  
  # --- 5. PLOTTING PANEL LAYOUT ---
  coeff_sc <- sc_lim[2] / q_lim[2]
  
  # Panel 1: Original Stream Sensors (Q & sC Validation)
  p1 <- ggplot() +
    geom_rect(data = shade_df, aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), fill="yellow", alpha=0.2) +
    geom_line(data = cmb_data, aes(x=timestamp, y=q_cms, color="Total Discharge"), linewidth=0.7) +
    geom_line(data = cmb_data, aes(x=timestamp, y=sC/coeff_sc, color="Sp. Conductivity"), linewidth=0.7) +
    scale_y_continuous(name="Q (cms)", limits=q_lim, expand=c(0,0),
                       sec.axis = sec_axis(~.*coeff_sc, name="sC (uS/cm)")) +
    scale_x_datetime(limits = plot_range, date_labels = "") +
    scale_color_manual(values = c("Total Discharge"="black", "Sp. Conductivity"="red")) +
    theme_bw() + theme(legend.position="top", axis.title.x = element_blank(), legend.title = element_blank())
  
  # Panel 2: Separated Hydrograph (Old vs New Flow Rates)
  p2 <- ggplot() +
    geom_rect(data = shade_df, aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), fill="yellow", alpha=0.2) +
    geom_line(data = cmb_data, aes(x=timestamp, y=q_cms, color="Total Q"), linewidth=0.5, linetype="dashed") +
    geom_line(data = cmb_data, aes(x=timestamp, y=q_old, color="Old water"), linewidth=0.8) +
    geom_line(data = cmb_data, aes(x=timestamp, y=q_new, color="New water"), linewidth=0.8) +
    scale_y_continuous(name="Component Q (cms)", limits=q_lim, expand=c(0,0)) +
    scale_x_datetime(limits = plot_range, date_labels = "%b %d") +
    scale_color_manual(values = c("Total Q"="grey50", "Old water"="blue", "New water"="cyan3")) +
    theme_bw() + theme(legend.position="bottom", legend.title = element_blank()) +
    labs(x = "Date")
  
  # Combine Panels Stacked Vertically
  final_cmb_plot <- p1 / p2 + plot_layout(heights = c(1, 1)) +
    plot_annotation(title = paste(site_name, "Brook: continuous conductivity mass balance"))
  
  return(final_cmb_plot)
}


plot_emma_vs_cmb_hydrographs <- function(site_name, 
                                         plot_range, 
                                         q_file, 
                                         exo_file, 
                                         emma_frac_file,
                                         q_lim = c(0, 4),
                                         sc_new_val = 12) {
  
  # --- 1. LOAD & PROCESS CONTINUOUS CMB DATA ---
  dat_q   <- read.csv(q_file)
  dat_exo <- read.csv(exo_file)
  
  dat_exo <- dat_exo %>% 
    rename(sC = matches("Sp.Cond")) %>%
    mutate(timestamp = ymd_hms(timestamp, tz = "UTC"))
  
  dat_q <- dat_q %>% 
    rename(timestamp = datetime) %>%
    rename(q_cms = matches("q_cms")) %>% 
    mutate(timestamp = ymd_hms(timestamp, tz = "UTC"))
  
  merged_dt <- as.data.table(dat_q)[as.data.table(dat_exo), roll = "nearest", on = .(timestamp)]
  
  cmb_data <- merged_dt %>%
    filter(timestamp >= plot_range[1] & timestamp <= plot_range[2]) %>%
    arrange(timestamp)
  
  # CMB Volumetric Math
  #sc_old_val <- cmb_data$sC[1] 
  sc_old_val <- 32
  cmb_data <- cmb_data %>%
    mutate(
      f_old = (sC - sc_new_val) / (sc_old_val - sc_new_val),
      f_old = pmax(0, pmin(1, f_old)),
      f_new = 1 - f_old,
      q_old = q_cms * f_old,
      q_new = q_cms * f_new
    )
  
  # --- 2. LOAD & PROCESS EMMA DATA ---
  # Perform a nearest-join with Q to get instantaneous discharge for each EMMA sample
  emma_raw <- read.csv(emma_frac_file) %>%
    mutate(timestamp = ymd_hms(Datetime, tz = "UTC")) %>%
    filter(timestamp >= plot_range[1] & timestamp <= plot_range[2])
  
  # Match discrete sample times to continuous Q rates
  emma_dt <- as.data.table(dat_q)[as.data.table(emma_raw), roll = "nearest", on = .(timestamp)]
  
  # EMMA Volumetric Math
  emma_data <- emma_dt %>%
    mutate(
      q_gw   = q_cms * Groundwater,
      q_melt = q_cms * Snowmelt.lysimeter,
      q_soil = q_cms * Soil.water.lysimeter
    )
  
  # --- 3. PANEL 1: EMMA VOLUMETRIC HYDROGRAPH (TOP) ---
  p1 <- ggplot() +
    # Total Hydrograph Background
    geom_line(data = cmb_data, aes(x = timestamp, y = q_cms, linetype = "Total Q"), color = "grey50", linewidth = 0.5) +
    # Discrete EMMA Component Points
    geom_point(data = emma_data, aes(x = timestamp, y = q_gw, color = "Groundwater"), shape = 17, size = 3) +
    geom_line(data = emma_data, aes(x = timestamp, y = q_gw, color = "Groundwater"), alpha = 0.4, linewidth = 0.7) +
    
    geom_point(data = emma_data, aes(x = timestamp, y = q_melt, color = "Snowmelt lysimeter"), shape = 16, size = 3) +
    geom_line(data = emma_data, aes(x = timestamp, y = q_melt, color = "Snowmelt lysimeter"), alpha = 0.4, linewidth = 0.7) +
    
    geom_point(data = emma_data, aes(x = timestamp, y = q_soil, color = "Soil water"), shape = 15, size = 3) +
    geom_line(data = emma_data, aes(x = timestamp, y = q_soil, color = "Soil water"), alpha = 0.4, linewidth = 0.7) +
    
    scale_y_continuous(name = "EMMA Q (cms)", limits = q_lim, expand = c(0, 0)) +
    scale_x_datetime(limits = plot_range, date_labels = "") +
    scale_color_manual(values = c("Groundwater" = "blue", "Snowmelt lysimeter" = "gold", "Soil water" = "firebrick")) +
    scale_linetype_manual(values = c("Total Q" = "dashed")) +
    theme_bw() + 
    labs(title = paste(site_name, "Brook: EMMA vs. CMB"), color = "EMMA components", linetype = "") +
    theme(axis.title.x = element_blank(), legend.position = "right")
  
  # --- 4. PANEL 2: CMB VOLUMETRIC HYDROGRAPH (BOTTOM) ---
  p2 <- ggplot() +
    geom_line(data = cmb_data, aes(x = timestamp, y = q_cms, linetype = "Total Q"), color = "grey50", linewidth = 0.5) +
    geom_line(data = cmb_data, aes(x = timestamp, y = q_old, color = "Old water"), linewidth = 0.9) +
    geom_line(data = cmb_data, aes(x = timestamp, y = q_new, color = "New water"), linewidth = 0.9) +
    
    scale_y_continuous(name = "CMB Q (cms)", limits = q_lim, expand = c(0, 0)) +
    scale_x_datetime(name = "Date", limits = plot_range, date_labels = "%b %d\n%H:%M") +
    scale_color_manual(values = c("Old water" = "blue", "New water"="cyan3")) +
    scale_linetype_manual(values = c("Total Q" = "dashed")) +
    theme_bw() + 
    labs(color = "CMB components", linetype = "") +
    theme(legend.position = "right")
  
  # Combine Stacked Panels
  stacked_plot <- p1 / p2 + plot_layout(heights = c(1, 1))
  return(stacked_plot)
}