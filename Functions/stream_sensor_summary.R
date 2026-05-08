
###################################################################
# R function to plot stream sensor data: q, N-NO3, DOC, turbidity #
# All corrected data downloaded as csv files from Aquarius ########
# Megan Duffy - Adair Lab, UVM ####################################
# last updated 2026-05-07 #########################################
###################################################################

#################
# LOAD PACKAGES #
#################

library(tidyverse)
library(lubridate)
library(patchwork)


stream_sensor_summary <- function(site_name, 
                                  plot_range, 
                                  q_file, 
                                  no3_file, 
                                  doc_file, 
                                  turb_file,
                                  q_lim = c(0, 4),
                                  no3_lim = c(0, 1),
                                  doc_lim = c(0, 10),
                                  turb_lim = c(0, 100)) {
  
  # 1. Helper function to read AQUARIUS CSVs (skipping # comments)
  read_aq <- function(path) {
    if (is.null(path) || path == "" || !file.exists(path)) return(NULL)
    
    read.csv(path, comment.char = "#", stringsAsFactors = FALSE) %>%
      # Using ISO 8601 UTC column to avoid local time/DST headaches
      mutate(timestamp = ymd_hms(ISO.8601.UTC, tz = "UTC")) %>%
      select(timestamp, Value)
  }
  
  # 2. Read and Prep Data
  dat_q    <- read_aq(q_file) %>% rename(q_cms = Value)
  dat_no3  <- read_aq(no3_file) %>% rename(no3_mgL = Value)
  dat_doc  <- read_aq(doc_file) %>% rename(doc_mgL = Value)
  dat_turb <- read_aq(turb_file) %>% rename(turb_ntu = Value)
  
  # 3. Create Astronomical Winter Shading Data (Dec 20 - Mar 20)
  # Extract years from the plot range to cover the whole timeline
  years <- year(plot_range[1]):year(plot_range[2])
  winter_shades <- data.frame(yr = years) %>%
    mutate(
      winter_start = as.POSIXct(paste0(yr, "-12-20 00:00:00"), tz = "UTC"),
      winter_end   = as.POSIXct(paste0(yr + 1, "-03-20 23:59:59"), tz = "UTC")
    )
  
  # 4. Plotting Helper Function for consistent panels
  base_plot <- function(data, y_var, y_label, y_lim, color_val) {
    ggplot() +
      # Winter Shading in background
      geom_rect(data = winter_shades, 
                aes(xmin = winter_start, xmax = winter_end, ymin = -Inf, ymax = Inf),
                fill = "lightcyan2", alpha = 0.4) +
      # Sensor Line
      geom_line(data = data, aes(x = timestamp, y = .data[[y_var]]), 
                color = color_val, linewidth = 0.6) +
      scale_x_datetime(limits = plot_range, date_labels = "%b %Y") +
      scale_y_continuous(limits = y_lim) +
      theme_bw() +
      labs(y = y_label) +
      theme(axis.title.x = element_blank())
  }
  
  # 5. Build Panels
  p_q    <- base_plot(dat_q, "q_cms", "Q (cms)", q_lim, "blue") + 
            labs(title = paste(site_name, "Brook: in situ sensor timeseries"))
  
  p_no3  <- base_plot(dat_no3, "no3_mgL", "N-NO3 (mg/L)", no3_lim, "darkgreen")
  
  p_doc  <- base_plot(dat_doc, "doc_mgL", "DOC (mg/L)", doc_lim, "brown")
  
  p_turb <- base_plot(dat_turb, "turb_ntu", "Turb (NTU)", turb_lim, "darkgrey")
  
  # 6. Stack and clean up axes
  # patchwork '&' operator applies the theme change to all plots in the stack
  final_stack <- (p_q / p_no3 / p_doc / p_turb) & 
                 theme(axis.text.x = element_blank())
  
  # Re-enable the x-axis labels only for the bottom plot (Turbidity)
  final_stack[[4]] <- final_stack[[4]] + 
                      theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  return(final_stack)
}