#########################################################
# R function to plot interannual groundwater timeseries #
# Megan Duffy - Adair Lab, UVM ##########################
# last updated 2026-05-07 ###############################
#########################################################

library(tidyverse)
library(lubridate)
library(patchwork)

plot_interannual_groundwater <- function(data_dir, site_name, tracer_list) {
  
  # 1. List and Read Files
  files <- list.files(path = data_dir, pattern = "\\.csv$", full.names = TRUE)
  all_data <- files %>% map_df(~read.csv(.x, stringsAsFactors = FALSE))
  
  # 2. Clean and Filter
  gw_data <- all_data %>%
    filter(Site == site_name) %>%
    filter(Type %in% c("Groundwater", "Baseflow")) %>%
    mutate(timestamp = mdy_hm(paste(Date, Time))) %>%
    filter(!is.na(timestamp)) %>%
    arrange(timestamp)
  
  # 3. Create Winter Shading Data
  # Get the range of years in the dataset
  min_year <- year(min(gw_data$timestamp)) - 1
  max_year <- year(max(gw_data$timestamp)) + 1
  
  # Create a dataframe for the rectangles (Dec 20th to March 20th)
  winter_shades <- data.frame(yr = min_year:max_year) %>%
    mutate(
      winter_start = as.POSIXct(paste0(yr, "-12-20 00:00:00")),
      winter_end   = as.POSIXct(paste0(yr + 1, "-03-20 23:59:59"))
    )
  
  # 4. Generate Plots
  plot_list <- tracer_list %>%
    map(function(tracer) {
      if (tracer %in% names(gw_data) && any(!is.na(gw_data[[tracer]]))) {
        
        ggplot() +
          # Add the Winter Shaded Boxes first so they are in the background
          geom_rect(data = winter_shades, 
                    aes(xmin = winter_start, xmax = winter_end, ymin = -Inf, ymax = Inf),
                    fill = "lightcyan2", alpha = 0.4) +
          # Plot the lines and points
          geom_line(data = gw_data, aes(x = timestamp, y = .data[[tracer]]), 
                    color = "grey80", alpha = 0.5) +
          geom_point(data = gw_data, aes(x = timestamp, y = .data[[tracer]], color = Type), 
                     size = 2) +
          scale_color_manual(values = c("Groundwater" = "darkblue", "Baseflow" = "skyblue")) +
          theme_bw() +
          labs(y = tracer) +
          theme(axis.title.x = element_blank(), legend.position = "none")
        
      } else {
        ggplot() + 
          annotate("text", x = 0.5, y = 0.5, label = paste("No data:", tracer)) +
          theme_void() + theme(panel.border = element_rect(colour = "black", fill=NA))
      }
    })
  
  # 5. Stack
  final_stack <- wrap_plots(plot_list, ncol = 1) +
    plot_annotation(
      title = paste("Interannual groundwater & baseflow timeseries:", site_name),
      subtitle = "Light blue shading indicates astronomical winter (Dec 20 - Mar 20)"
    )
  
  return(final_stack)
}



