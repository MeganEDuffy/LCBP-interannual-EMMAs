#########################################################
# R function to plot stream Q, sC, grab chem for events #
# Also plots endmember tracer geochemistry ##############
# Megan Duffy - Adair Lab, UVM ##########################
# last updated 2026-07-29 ###############################
#########################################################


#################
# LOAD PACKAGES #
#################

library(tidyverse)
library(lubridate)
library(patchwork)

# --- Unicode-safe permil symbol formatting ---
format_tracer_label <- function(tracer) {
  if (tracer == "d18O") {
    return(expression(delta^18 * "O (" * "\u2030" * ")"))
  } else if (tracer %in% c("dD", "d2H")) {
    return(expression(delta * "D (" * "\u2030" * ")"))
  } else {
    lbl <- gsub("_mg_L$", " (mg/L)", tracer)
    lbl <- gsub("_ug_L$", " (µg/L)", lbl)
    lbl <- gsub("_uS_cm$", " (µS/cm)", lbl)
    lbl <- gsub("_", " ", lbl)
    return(lbl)
  }
}

plot_winter_tracer_summary <- function(site_name,
                                        water_year,          # e.g., 2023 for Nov 2022 - Apr 2023
                                        event_bounds = NULL, # Dataframe with start/end columns
                                        q_file,
                                        chem_data_dir,
                                        tracer_list,
                                        q_lim = c(0, 5),
                                        base_font_size = 20,
                                        panel_number) {
  
  # -------------------------------------------------------------------
  # 1. DEFINE SEASONAL TIME WINDOWS
  # -------------------------------------------------------------------
  plot_start <- as.POSIXct(paste0(water_year, "-02-05 00:00:00"), tz = "UTC")
  plot_end   <- as.POSIXct(paste0(water_year, "-04-16 23:59:59"), tz = "UTC")
  
  winter_shade <- data.frame(
    xmin = as.POSIXct(paste0(water_year - 1, "-12-20 00:00:00"), tz = "UTC"),
    xmax = as.POSIXct(paste0(water_year, "-03-20 23:59:59"), tz = "UTC"),
    ymin = -Inf, ymax = Inf
  )
  
  if (!is.null(event_bounds)) {
    event_shades <- event_bounds %>%
      mutate(
        xmin = as.POSIXct(start, tz = "UTC"),
        xmax = as.POSIXct(end, tz = "UTC"),
        ymin = -Inf, ymax = Inf
      )
  } else {
    event_shades <- NULL
  }

  # -------------------------------------------------------------------
  # 2. READ & PREP DISCHARGE (Flexible parsing for dashes/slashes)
  # -------------------------------------------------------------------
  dat_q <- read.csv(q_file, comment.char = "#")
  
  q_time_col <- names(dat_q)[grep("datetime|timestamp|ISO|Date", names(dat_q), ignore.case = TRUE)][1]
  q_val_col  <- names(dat_q)[grep("q_cms|Value", names(dat_q), ignore.case = TRUE)][1]
  
  dat_q <- dat_q %>%
    mutate(
      # Flexible parsing for YMD/MDY with dashes or slashes
      timestamp = parse_date_time(
        .data[[q_time_col]], 
        orders = c("ymd HMS", "ymd HM", "mdy HMS", "mdy HM"), 
        tz = "UTC"
      ),
      q_cms = as.numeric(.data[[q_val_col]])
    ) %>%
    filter(!is.na(timestamp)) %>%
    filter(timestamp >= (plot_start - days(2)) & timestamp <= (plot_end + days(2)))

  # Check in console how many discharge points parsed
  message(paste(" Total discharge rows parsed:", nrow(dat_q)))

  # -------------------------------------------------------------------
  # 3. READ & PREP CHEMISTRY DATA (Relabel Types for simplicity)
  # -------------------------------------------------------------------
  if (dir.exists(chem_data_dir)) {
    chem_files <- list.files(path = chem_data_dir, pattern = "\\.csv$", full.names = TRUE)
    chem_raw   <- chem_files %>% map_df(~read.csv(.x, stringsAsFactors = FALSE))
  } else {
    chem_raw   <- read.csv(chem_data_dir, stringsAsFactors = FALSE)
  }
  
  chem_clean <- chem_raw %>%
    filter(Site == site_name) %>%
    mutate(
      date_str = trimws(as.character(Date)),
      time_str = trimws(as.character(Time)),
      time_str = ifelse(nchar(time_str) == 5, paste0(time_str, ":00"), time_str),
      dt_combined = paste(date_str, time_str)
    ) %>%
    mutate(
      timestamp_local = as.POSIXct(dt_combined, format = "%m/%d/%Y %H:%M:%S", tz = "America/New_York"),
      timestamp = with_tz(timestamp_local, tzone = "UTC")
    ) %>%
    filter(!is.na(timestamp)) %>%
    # RELABEL SAMPLE TYPES
    mutate(
      Type = case_when(
        Type %in% c("Grab", "Grab/Isco", "Isco", "ISCO") ~ "Streamwater",
        Type %in% c("Snowmelt lysimeter", "Snowmelt")    ~ "Meltwater",
        TRUE                                             ~ Type
      )
    )

  # Source formatting aesthetics (Simplified mapped labels)
  source_colors <- c(
    "Streamwater"             = "black",
    "Baseflow"                = "skyblue3",
    "Groundwater"             = "darkblue",
    "Soil water lysimeter"    = "firebrick",
    "Soil water lysimeter dry"= "firebrick",
    "Soil water lysimeter wet"= "firebrick",
    "Meltwater"               = "gold3",
    "Precip"                  = "purple",
    "Snow"                    = "purple"
  )
  
  source_shapes <- c(
    "Streamwater"             = 16, # Solid circle
    "Baseflow"                = 17, # Triangle
    "Groundwater"             = 18, # Diamond
    "Soil water lysimeter"    = 15, # Square
    "Soil water lysimeter dry"= 15,
    "Soil water lysimeter wet"= 15,
    "Meltwater"               = 8,  # Star
    "Precip"                  = 4,   # Cross
    "Snow"                    = 5   # 
  )

  # -------------------------------------------------------------------
  # 4. BUILD TOP PANEL: HYDROGRAPH
  # -------------------------------------------------------------------
  p_q <- ggplot() +
    geom_rect(data = winter_shade, aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), 
              fill = "aliceblue", alpha = 0.6) +
    { if (!is.null(event_shades)) 
        geom_rect(data = event_shades, aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), 
                  fill = "yellow", alpha = 0.35) } +
    geom_line(data = dat_q, aes(x = timestamp, y = q_cms), color = "blue", linewidth = 0.8) +
    scale_y_continuous(
      name = "Q (m³/s)", 
      limits = q_lim, 
      expand = c(0, 0)
    ) +
    coord_cartesian(xlim = c(plot_start, plot_end)) +
    scale_x_datetime(date_labels = "") +
    theme_bw(base_size = base_font_size) + 
    theme(
      axis.title.x = element_blank()
    )

  # -------------------------------------------------------------------
  # 5. BUILD DYNAMIC GEOCHEMISTRY PANELS
  # -------------------------------------------------------------------
  geochem_plots <- tracer_list %>%
    map(function(tracer) {
      if (tracer %in% names(chem_clean) && any(!is.na(chem_clean[[tracer]]))) {
        
        val_min <- min(chem_clean[[tracer]], na.rm = TRUE)
        val_max <- max(chem_clean[[tracer]], na.rm = TRUE)
        val_range <- val_max - val_min
        if (is.na(val_range) || val_range == 0) val_range <- 1

        dat_q_scaled <- dat_q %>%
          mutate(
            q_scaled = val_min + (q_cms / q_lim[2]) * (val_range * 0.55)
          )
        
        y_label <- format_tracer_label(tracer)

        ggplot() +
          geom_rect(data = winter_shade, aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), 
                    fill = "aliceblue", alpha = 0.6) +
          { if (!is.null(event_shades)) 
              geom_rect(data = event_shades, aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), 
                        fill = "yellow", alpha = 0.35) } +
          # Hydrograph background reference line
          geom_line(data = dat_q_scaled, aes(x = timestamp, y = q_scaled), 
                    color = "grey80", alpha = 0.85, linewidth = 0.9) +
          # Chemistry points
          geom_point(data = chem_clean, 
                     aes(x = timestamp, y = .data[[tracer]], color = Type, shape = Type), 
                     size = 3.0, alpha = 0.85) +
          scale_color_manual(values = source_colors, drop = FALSE) +
          scale_shape_manual(values = source_shapes, drop = FALSE) +
          coord_cartesian(xlim = c(plot_start, plot_end)) +
          scale_x_datetime(date_labels = "%b %d") +
          theme_bw(base_size = base_font_size) +
          labs(y = y_label) +
          theme(
            axis.title.x = element_blank(),
            legend.position = "none"
          )
      } else {
        ggplot() + 
          annotate("text", x = 0.5, y = 0.5, label = paste("No data for tracer:", tracer), size = 5) +
          theme_void() + 
          theme(panel.border = element_rect(colour = "black", fill = NA))
      }
    })

  # -------------------------------------------------------------------
  # 6. STACK ALL PANELS
  # -------------------------------------------------------------------
  all_panels <- c(list(p_q), geochem_plots)
  
  final_stack <- wrap_plots(all_panels, ncol = 1, heights = c(1.2, rep(1, length(tracer_list)))) +
    plot_layout(guides = "collect") & 
    theme(
      legend.position = "bottom",
      legend.text = element_text(size = base_font_size),
      legend.title = element_text(size = base_font_size + 1, face = "bold")
    )
  
  final_stack <- final_stack + 
    plot_annotation(
      title = paste(panel_number, ")", site_name, "Brook"),
      theme = theme(
        plot.title = element_text(size = base_font_size + 4, face = "bold")
      )
    )
    
  return(final_stack)
}