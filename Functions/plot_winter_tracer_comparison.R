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

# Helper function to build a single site's vertical stack of tracer panels
build_site_stack <- function(site_name, water_year, event_bounds, q_file, chem_data_dir, 
                           tracer_list, q_lim, base_font_size, panel_label) {
  
  # -------------------------------------------------------------------
  # 1. DEFINE SEASONAL TIME WINDOWS
  # -------------------------------------------------------------------
  plot_start <- as.POSIXct(paste0(water_year, "-01-20 00:00:00"), tz = "UTC")
  plot_end   <- as.POSIXct(paste0(water_year, "-04-16 23:59:59"), tz = "UTC")
  
  winter_shade <- data.frame(
    xmin = as.POSIXct(paste0(water_year - 1, "-12-20 00:00:00"), tz = "UTC"),
    xmax = as.POSIXct(paste0(water_year, "-03-20 23:59:59"), tz = "UTC"),
    ymin = -Inf, ymax = Inf
  )
  
  event_shades <- if (!is.null(event_bounds)) {
    event_bounds %>%
      mutate(
        xmin = as.POSIXct(start, tz = "UTC"),
        xmax = as.POSIXct(end, tz = "UTC"),
        ymin = -Inf, ymax = Inf
      )
  } else {
    NULL
  }

  # -------------------------------------------------------------------
  # 2. READ & PREP DISCHARGE
  # -------------------------------------------------------------------
  dat_q <- read.csv(q_file, comment.char = "#")
  q_time_col <- names(dat_q)[grep("datetime|timestamp|ISO|Date", names(dat_q), ignore.case = TRUE)][1]
  q_val_col  <- names(dat_q)[grep("q_cms|Value", names(dat_q), ignore.case = TRUE)][1]
  
  dat_q <- dat_q %>%
    mutate(
      timestamp = parse_date_time(.data[[q_time_col]], orders = c("ymd HMS", "ymd HM", "mdy HMS", "mdy HM"), tz = "UTC"),
      q_cms = as.numeric(.data[[q_val_col]])
    ) %>%
    filter(!is.na(timestamp)) %>%
    filter(timestamp >= (plot_start - days(2)) & timestamp <= (plot_end + days(2)))

  # -------------------------------------------------------------------
  # 3. READ & PREP CHEMISTRY DATA (Split per site first)
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
    mutate(
      Type = case_when(
        Type %in% c("Grab", "Grab/Isco", "Isco", "ISCO") ~ "Streamwater",
        Type %in% c("Snowmelt lysimeter", "Snowmelt")    ~ "Meltwater",
        TRUE                                             ~ Type
      )
    )

  source_colors <- c(
    "Streamwater"              = "blue",
    "Baseflow"                 = "skyblue3",
    "Groundwater"              = "darkblue",
    "Soil water lysimeter"     = "firebrick",
    "Soil water lysimeter dry" = "firebrick",
    "Soil water lysimeter wet" = "firebrick",
    "Meltwater"                = "darkorange4",
    "Precip"                   = "purple",
    "Snow"                     = "purple"
  )
  
  source_shapes <- c(
    "Streamwater"              = 1,
    "Baseflow"                 = 17,
    "Groundwater"              = 18,
    "Soil water lysimeter"     = 15,
    "Soil water lysimeter dry" = 15,
    "Soil water lysimeter wet" = 15,
    "Meltwater"                = 8,
    "Precip"                   = 4,
    "Snow"                     = 17
  )

  # -------------------------------------------------------------------
  # 4. BUILD GEOCHEMISTRY PANELS (Standard Linear Scales)
  # -------------------------------------------------------------------
  geochem_plots <- tracer_list %>%
    map(function(tracer) {
      
      if (tracer %in% names(chem_clean) && any(!is.na(chem_clean[[tracer]]))) {
        
        valid_vals <- chem_clean[[tracer]][!is.na(chem_clean[[tracer]])]
        
        # Calculate standard linear scaling for background hydrograph trace
        if (length(valid_vals) > 0) {
          val_min <- min(valid_vals, na.rm = TRUE)
          val_max <- max(valid_vals, na.rm = TRUE)
          val_range <- val_max - val_min
          if (is.na(val_range) || val_range == 0) val_range <- 1
          
          dat_q_scaled <- dat_q %>%
            mutate(q_scaled = val_min + (q_cms / q_lim[2]) * (val_range * 0.98))
        } else {
          dat_q_scaled <- dat_q %>% mutate(q_scaled = NA)
        }
        
        y_label <- format_tracer_label(tracer)

        p <- ggplot() +
          geom_rect(data = winter_shade, aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), fill = "aliceblue", alpha = 0.8, inherit.aes = FALSE) +
          { if (!is.null(event_shades)) geom_rect(data = event_shades, aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), fill = "yellow", alpha = 0.35, inherit.aes = FALSE) } +
          geom_line(data = dat_q_scaled, aes(x = timestamp, y = q_scaled), color = "grey80", alpha = 1, linewidth = 0.9) +
          geom_point(data = chem_clean, aes(x = timestamp, y = .data[[tracer]], color = Type, shape = Type), size = 4, alpha = 0.85) +
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
        
        return(p)
      } else {
        ggplot() + 
          annotate("text", x = 0.5, y = 0.5, label = paste("No data for tracer:", tracer), size = 4) +
          theme_void() + 
          theme(panel.border = element_rect(colour = "black", fill = NA))
      }
    })

  # Stack panels vertically for this site
  site_stack <- wrap_plots(geochem_plots, ncol = 1)
  
  return(site_stack)
}

# Main wrapper function to combine two sites side-by-side
plot_winter_tracer_comparison <- function(site1_name, site2_name,
                                          water_year = 2023,
                                          event_bounds_1 = NULL, event_bounds_2 = NULL,
                                          q_file_1, q_file_2,
                                          chem_data_dir,
                                          tracer_list_1, tracer_list_2,
                                          q_lim = c(0, 12),
                                          base_font_size = 14) {
  
  plot1 <- build_site_stack(site1_name, water_year, event_bounds_1, q_file_1, chem_data_dir, 
                            tracer_list_1, q_lim, base_font_size + 4, panel_label = "a)")
  
  plot2 <- build_site_stack(site2_name, water_year, event_bounds_2, q_file_2, chem_data_dir, 
                            tracer_list_2, q_lim, base_font_size + 4, panel_label = "b)")
  
  # Combine side-by-side, add tag labels (a and b), and collect legends at the bottom
  combined_plot <- (plot1 | plot2) +
    plot_layout(guides = "collect") +
    plot_annotation(
      tag_levels = list(c("", "")),
      title = paste0("         a)"        , "  ", site1_name, " Brook                           b)"       , "  ", site2_name, " Brook"),
      theme = theme(
        plot.title = element_text(size = base_font_size + 4, face = "bold", hjust = 0),
        plot.tag = element_text(size = base_font_size + 4, face = "bold"),
        legend.position = "bottom",
        legend.text = element_text(size = base_font_size - 1)
      )
    ) &
    theme(
      legend.position = "bottom",
      legend.text = element_text(size = base_font_size - 1)
    )
  
  return(combined_plot)
}