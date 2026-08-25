#################
# LOAD PACKAGES #
#################

#################
# LOAD PACKAGES #
#################
library(tidyverse)
library(lubridate)
library(data.table)
library(ggplot2)
library(patchwork)

plot_event_emma_with_soils_and_chem <- function(site_name, 
                                                plot_range, 
                                                q_file, 
                                                snow_file,
                                                soil_file_6cm,
                                                soil_file_15cm,
                                                met_file,
                                                emma_frac_file,
                                                no3_file,
                                                doc_file,
                                                q_lim = c(0, 4),
                                                no3_lim = c(0, 2),
                                                doc_lim = c(0, 20),
                                                snow_lim = c(0, 60),      
                                                air_temp_lim = c(-20, 20),
                                                precip_lim = c(0, 50),     # Max limit for precipitation scale mapping
                                                soil_vwc_lim = c(0, 0.5), 
                                                soil_temp_lim = c(-5, 20), 
                                                base_font_size = 14,
                                                event_title = NULL,
                                                show_legend = TRUE,
                                                shading_df = NULL) {
  
  start_date <- as.POSIXct(plot_range[1], tz = "UTC")
  end_date   <- as.POSIXct(plot_range[2], tz = "UTC")
  
  # -------------------------------------------------------------------
  # 1. LOAD & PREP MET DATA (Air Temp & Precip)
  # -------------------------------------------------------------------
  dat_met <- read.csv(met_file, comment.char = "#") %>%
    rename_with(~ "Timestamp", matches("Timestamp|datetime|Date", ignore.case = TRUE)) %>%
    rename_with(~ "Air_Temp", matches("Air_Temp|AirTemp", ignore.case = TRUE)) %>%
    rename_with(~ "Precip_Increm", matches("Precip_Increm|Precipitation|Precip", ignore.case = TRUE)) %>%
    mutate(
      datetime = parse_date_time(Timestamp, orders = c("ymd HMS", "ymd HM", "mdy HM", "mdy HMS", "ymd"), tz = "UTC"),
      Precip_Increm = as.numeric(Precip_Increm)
    ) %>%
    filter(!is.na(datetime), datetime >= start_date & datetime <= end_date)
  
  met_daily <- dat_met %>%
    mutate(date = as.Date(datetime, tz = "UTC")) %>%
    group_by(date) %>%
    summarise(
      Air_Temp_daily = mean(Air_Temp, na.rm = TRUE),
      Precip_daily = sum(Precip_Increm, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(datetime = as.POSIXct(paste(date, "12:00:00"), tz = "UTC"))

  temp_min <- air_temp_lim[1]
  temp_max <- air_temp_lim[2]
  temp_range <- temp_max - temp_min
  snow_max <- snow_lim[2]
  snow_scale <- snow_max / temp_range

  # Scaling factor to map precipitation onto the snow/temp primary axis range (reversed from snow_max down to 0)
  precip_max <- precip_lim[2]
  precip_scale <- snow_max / precip_max

  # -------------------------------------------------------------------
  # 2. LOAD & PREP SNOW DATA
  # -------------------------------------------------------------------
  dat_snow <- read.csv(snow_file, comment.char = "#") %>%
    rename_with(~ "snow_cm", matches("modeled_snow_depth_cm|snowpack_cm|Snowpack Depth cm", ignore.case = TRUE)) %>%
    rename_with(~ "Date_col", matches("Date|DATE|datetime|Timestamp", ignore.case = TRUE)) %>%
    mutate(
      datetime = parse_date_time(Date_col, orders = c("ymd", "mdy", "ymd HMS", "mdy HM"), tz = "UTC"),
      snow_cm = as.numeric(gsub("--", NA, as.character(snow_cm)))
    ) %>%
    filter(!is.na(datetime), datetime >= start_date & datetime <= end_date)

  # -------------------------------------------------------------------
  # 3. LOAD & PREP SOIL DATA (6 cm and 15 cm)
  # -------------------------------------------------------------------
  dat_soil_6cm <- read.csv(soil_file_6cm, comment.char = "#") %>%
    rename_with(~ "Timestamp", matches("Timestamp|dateTimeText|datetime", ignore.case = TRUE)) %>%
    mutate(datetime = parse_date_time(Timestamp, orders = c("ymd HMS", "ymd HM", "mdy HM", "mdy HMS"), tz = "UTC")) %>%
    filter(!is.na(datetime), datetime >= start_date & datetime <= end_date)

  s_temp_col6 <- names(dat_soil_6cm)[grep("Temp|temperature", names(dat_soil_6cm), ignore.case = TRUE)][1]
  s_vwc_col6  <- names(dat_soil_6cm)[grep("VWC|vwc", names(dat_soil_6cm), ignore.case = TRUE)][1]

  load_soil_15cm <- function(sl_file) {
    if (is.null(sl_file) || !file.exists(sl_file)) return(NULL)
    df <- read.csv(sl_file, comment.char = "#", stringsAsFactors = FALSE, check.names = FALSE)
    df <- df[, names(df) != "" & !is.na(names(df)) & !grepl("^\\.\\.\\.", names(df))]
    col_names <- names(df)
    
    t_idx <- grep("Timestamp|datetime|Date", col_names, ignore.case = TRUE)[1]
    if (is.na(t_idx)) t_idx <- 1 
    
    temp_idx <- intersect(grep("15cm", col_names, ignore.case = TRUE), grep("Temp", col_names, ignore.case = TRUE))[1]
    vwc_idx  <- intersect(grep("15cm", col_names, ignore.case = TRUE), grep("VWC|Water|Volumetric", col_names, ignore.case = TRUE))[1]
    
    if (is.na(t_idx) || is.na(temp_idx) || is.na(vwc_idx)) return(NULL)
    
    t_col    <- col_names[t_idx]
    temp_col <- col_names[temp_idx]
    vwc_col  <- col_names[vwc_idx]
    
    df %>%
      mutate(
        datetime = parse_date_time(.data[[t_col]], orders = c("ymd HMS", "ymd HM", "mdy HM", "mdy HMS", "ymd"), tz = "UTC"),
        Soil_Temp = as.numeric(.data[[temp_col]]),
        VWC = as.numeric(.data[[vwc_col]])
      ) %>%
      filter(!is.na(datetime), datetime >= start_date & datetime <= end_date)
  }

  dat_soil_15cm <- load_soil_15cm(soil_file_15cm)

  v_range <- soil_vwc_lim[2] - soil_vwc_lim[1]
  t_range <- soil_temp_lim[2] - soil_temp_lim[1]
  soil_scale <- t_range / v_range 

  # -------------------------------------------------------------------
  # 4. LOAD & PREP DISCHARGE & EMMA FRACTIONS
  # -------------------------------------------------------------------
  dat_q <- read.csv(q_file, comment.char = "#")
  q_time_col <- names(dat_q)[grep("datetime|timestamp|ISO|Date", names(dat_q), ignore.case = TRUE)][1]
  q_val_col  <- names(dat_q)[grep("q_cms|Value", names(dat_q), ignore.case = TRUE)][1]
  
  dat_q <- dat_q %>% 
    rename(timestamp = all_of(q_time_col)) %>%
    rename(q_cms = all_of(q_val_col)) %>% 
    mutate(timestamp = ymd_hms(timestamp, tz = "UTC")) %>%
    filter(timestamp >= start_date & timestamp <= end_date) %>%
    arrange(timestamp)

  emma_raw <- read.csv(emma_frac_file) %>%
    mutate(timestamp = ymd_hms(Datetime, tz = "UTC")) %>%
    filter(timestamp >= start_date & timestamp <= end_date)
  
  emma_dt <- as.data.table(dat_q)[as.data.table(emma_raw), roll = "nearest", on = .(timestamp)]
  
  find_col <- function(df, patterns) {
    col_names <- names(df)
    for (pat in patterns) {
      match_idx <- grep(pat, col_names, ignore.case = TRUE)
      if (length(match_idx) > 0) return(col_names[match_idx[1]])
    }
    return(NULL)
  }
  
  gw_col   <- find_col(emma_dt, c("Groundwater", "Baseflow"))
  melt_col <- find_col(emma_dt, c("Snowmelt", "Meltwater", "Snow"))
  soil_col <- find_col(emma_dt, c("Soil water lysimeter", "Soil", "Soil water"))
  
  if (is.null(gw_col) || is.null(melt_col) || is.null(soil_col)) {
    stop("Could not automatically match EMMA fraction columns. Check column names in: ", emma_frac_file)
  }
  
  emma_data <- emma_dt %>%
    mutate(
      Groundwater   = q_cms * .data[[gw_col]],
      Meltwater = q_cms * .data[[melt_col]],
      `Soil water`  = q_cms * .data[[soil_col]]
    )

  emma_long <- emma_data %>%
    select(timestamp, Groundwater, Meltwater, `Soil water`) %>%
    pivot_longer(cols = c(Groundwater, Meltwater, `Soil water`), names_to = "Component", values_to = "q_component") %>%
    mutate(Component = factor(Component, levels = c("Soil water", "Meltwater", "Groundwater")))

  # -------------------------------------------------------------------
  # 5. LOAD SENSOR HELPER FOR CHEM (DOC & NO3)
  # -------------------------------------------------------------------
  load_sensor <- function(file_path, val_pattern) {
    if (is.null(file_path) || !file.exists(file_path)) return(NULL)
    df <- read.csv(file_path, comment.char = "#", stringsAsFactors = FALSE)
    t_col <- names(df)[grep("ISO\\.8601\\.UTC|datetime|timestamp|Date", names(df), ignore.case = TRUE)][1]
    v_col <- names(df)[grep(val_pattern, names(df), ignore.case = TRUE)][1]
    if (is.na(t_col) || !is.null(v_col) && is.na(v_col)) return(NULL)
    df %>%
      mutate(
        timestamp = ymd_hms(.data[[t_col]], tz = "UTC"),
        value = as.numeric(.data[[v_col]])
      ) %>%
      filter(!is.na(timestamp), timestamp >= start_date, timestamp <= end_date) %>%
      select(timestamp, value)
  }

  dat_no3 <- load_sensor(no3_file, "NO3|Nitrate|Value")
  dat_doc <- load_sensor(doc_file, "doc|Value")

  doc_scale <- doc_lim[2] / no3_lim[2]
  no3_span <- no3_lim[2] - no3_lim[1]
  dat_q_chem_scaled <- dat_q %>%
    mutate(q_scaled = no3_lim[1] + (q_cms / q_lim[2]) * (no3_span * 0.55))

  # -------------------------------------------------------------------
  # 6. BUILD PANELS
  # -------------------------------------------------------------------
  
  # Panel 1: Met, Snowpack, and Reversed Precipitation
  p_snow <- ggplot() +
    { if (!is.null(shading_df)) geom_rect(data = shading_df, aes(xmin = start, xmax = end, ymin = -Inf, ymax = Inf), fill = "yellow", alpha = 0.2, inherit.aes = FALSE) } +
    # Reversed Precipitation bars (hanging from the top: snow_max down to snow_max - value * precip_scale)
    { if ("Precip_Increm" %in% names(dat_met) && nrow(dat_met) > 0) 
        geom_col(data = dat_met, aes(x = datetime, y = snow_max - Precip_Increm * precip_scale, fill = "Precipitation"), width = 3600, alpha = 0.6) } +
    geom_line(data = dat_snow, aes(x = datetime, y = snow_cm, color = "Snow Depth", linetype = "Snow Depth"), linewidth = 1.5) +
    geom_line(data = met_daily, aes(x = datetime, y = (Air_Temp_daily - temp_min) * snow_scale + snow_lim[1], color = "Air Temp", linetype = "Air Temp"), linewidth = 1.5) +
    scale_y_continuous(
      name = "Snow depth (cm)", limits = snow_lim, expand = c(0, 0),
      sec.axis = sec_axis(~ (. - snow_lim[1]) / snow_scale + temp_min, name = "Air Temp (°C)")
    ) +
    scale_x_datetime(limits = c(start_date, end_date), date_labels = "") +
    scale_color_manual(values = c("Snow Depth" = "blue4", "Air Temp" = "firebrick")) +
    scale_fill_manual(values = c("Precipitation" = "dodgerblue")) +
    scale_linetype_manual(values = c("Snow Depth" = "solid", "Air Temp" = "dotted")) +
    theme_bw(base_size = base_font_size) +
    labs(title = event_title, color = NULL, fill = NULL, linetype = NULL) +
    theme(
      axis.title.x = element_blank(), 
      axis.text.x = element_blank(), 
      plot.title = element_text(face = "bold"),
      axis.title.y.right = element_text(color = "firebrick"),
      axis.title.y.left = element_text(color = "blue4"),
      legend.position = "none"
    )

  # Panel 2: Stacked Volumetric Hydrograph
  p_emma <- ggplot() +
    { if (!is.null(shading_df)) geom_rect(data = shading_df, aes(xmin = start, xmax = end, ymin = -Inf, ymax = Inf), fill = "yellow", alpha = 0.2, inherit.aes = FALSE) } +
    geom_area(data = emma_long, aes(x = timestamp, y = q_component, fill = Component), position = "stack", alpha = 0.85) +
    geom_line(data = dat_q, aes(x = timestamp, y = q_cms, linetype = "Total Q"), color = "black", linewidth = 1.1) +
    scale_y_continuous(name = "Stream q (cms) & EMMA", limits = q_lim, expand = c(0, 0)) +
    scale_x_datetime(limits = c(start_date, end_date), date_labels = "") +
    scale_fill_manual(values = c("Groundwater" = "deepskyblue3", "Meltwater" = "gold3", "Soil water" = "firebrick")) +
    scale_linetype_manual(values = c("Total Q" = "solid")) +
    guides(fill = guide_legend(ncol = 3), linetype = guide_legend(ncol = 1)) +
    theme_bw(base_size = base_font_size) + 
    labs(x = "", fill = "Components", linetype = "") +
    theme(axis.title.x = element_blank(), axis.text.x = element_blank(), legend.position = "bottom")

  # Panel 3: Stream Chemistry
  p_chem <- ggplot() +
    { if (!is.null(shading_df)) geom_rect(data = shading_df, aes(xmin = start, xmax = end, ymin = -Inf, ymax = Inf), fill = "yellow", alpha = 0.2, inherit.aes = FALSE) } +
    geom_line(data = dat_q_chem_scaled, aes(x = timestamp, y = q_scaled), color = "grey80", alpha = 0.85, linewidth = 1.5) +
    { if (!is.null(dat_no3) && nrow(dat_no3) > 0) geom_line(data = dat_no3, aes(x = timestamp, y = value, color = "NO3"), linewidth = 1.5) } +
    { if (!is.null(dat_doc) && nrow(dat_doc) > 0) geom_line(data = dat_doc, aes(x = timestamp, y = value / doc_scale, color = "DOC"), linewidth = 1.5) } +
    scale_y_continuous(
      name = "Stream nitrate (mg/L)", limits = no3_lim, expand = c(0, 0),
      sec.axis = sec_axis(~ . * doc_scale, name = "Stream DOC (mg/L)")
    ) +
    scale_x_datetime(limits = c(start_date, end_date), date_labels = "") +
    scale_color_manual(values = c("NO3" = "darkgreen", "DOC" = "saddlebrown")) +
    guides(color = guide_legend(ncol = 2)) +
    theme_bw(base_size = base_font_size) +
    labs(x = "", color = NULL) +
    theme(
      axis.title.x = element_blank(),
      axis.text.x = element_blank(),
      axis.title.y.left = element_text(color = "darkgreen"),
      axis.title.y.right = element_text(color = "saddlebrown"),
      legend.position = "bottom"
    )

  # Panel 4: Consolidated Soil Temperature and Moisture
  p_soil <- ggplot() +
    { if (!is.null(shading_df)) geom_rect(data = shading_df, aes(xmin = start, xmax = end, ymin = -Inf, ymax = Inf), fill = "yellow", alpha = 0.2, inherit.aes = FALSE) } +
    geom_smooth(data = dat_soil_6cm, aes(x = datetime, y = .data[[s_vwc_col6]]), color = "dodgerblue3", se = FALSE, linewidth = 1.4, method = "loess", span = 0.15) +
    { if (!is.null(dat_soil_15cm)) geom_smooth(data = dat_soil_15cm, aes(x = datetime, y = VWC), color = "dodgerblue4", se = FALSE, linewidth = 2, method = "loess", span = 0.15) } +
    geom_smooth(data = dat_soil_6cm, aes(x = datetime, y = (.data[[s_temp_col6]] - soil_temp_lim[1]) / soil_scale + soil_vwc_lim[1]), color = "darkorange3", linetype = "dotted", se = FALSE, linewidth = 1.4, method = "loess", span = 0.15) +
    { if (!is.null(dat_soil_15cm)) geom_smooth(data = dat_soil_15cm, aes(x = datetime, y = (Soil_Temp - soil_temp_lim[1]) / soil_scale + soil_vwc_lim[1]), color = "chocolate4", linetype = "dotted", se = FALSE, linewidth = 2, method = "loess", span = 0.15) } +
    scale_y_continuous(
      name = "Soil VWC", limits = soil_vwc_lim,
      sec.axis = sec_axis(~ (. - soil_vwc_lim[1]) * soil_scale + soil_temp_lim[1], name = "Soil temp (°C)")
    ) +
    scale_x_datetime(limits = c(start_date, end_date), date_labels = "%b %d") +
    theme_bw(base_size = base_font_size) +
    labs(x = "Date") +
    theme(
      axis.title.y.left = element_text(color = "dodgerblue3"),
      axis.title.y.right = element_text(color = "darkorange3")
    )

  leg_pos <- if (show_legend) "bottom" else "none"

  stacked_column <- (p_snow / p_emma / p_chem / p_soil) + 
    plot_layout(heights = c(0.9, 1.2, 1.1, 1)) &
    theme(legend.position = leg_pos)

  return(stacked_column)
}

plot_multi_event_grid <- function(site_name, 
                                q_file, 
                                snow_file,
                                soil_file_6cm,
                                soil_file_15cm,
                                met_file,
                                no3_file,
                                doc_file,
                                event_windows_df,
                                emma_files,
                                event_titles,
                                q_lim = c(0, 3.5),
                                no3_lim = c(0, 2),
                                doc_lim = c(0, 20),
                                snow_lim = c(0, 60),      
                                air_temp_lim = c(-20, 20),
                                precip_lim = c(0, 50),
                                soil_vwc_lim = c(0, 0.5),    
                                soil_temp_lim = c(-5, 20),  
                                base_font_size = 11,
                                shading_df = NULL) {
  
  cols <- list()
  
  for (i in seq_len(nrow(event_windows_df))) {
    r_start <- event_windows_df$start[i]
    r_end   <- event_windows_df$end[i]
    e_file  <- emma_files[i]
    e_title <- event_titles[i]
    
    is_first <- (i == 1)
    
    col_plot <- plot_event_emma_with_soils_and_chem(
      site_name      = site_name,
      plot_range     = c(r_start, r_end),
      q_file         = q_file,
      snow_file      = snow_file,
      soil_file_6cm  = soil_file_6cm,
      soil_file_15cm = soil_file_15cm,
      met_file       = met_file,
      emma_frac_file = e_file,
      no3_file       = no3_file,
      doc_file       = doc_file,
      q_lim          = q_lim,
      no3_lim        = no3_lim,
      doc_lim        = doc_lim,
      snow_lim       = snow_lim,        
      air_temp_lim   = air_temp_lim,    
      precip_lim     = precip_lim,
      soil_vwc_lim   = soil_vwc_lim,    
      soil_temp_lim  = soil_temp_lim,  
      base_font_size = base_font_size,
      event_title    = e_title,
      show_legend    = is_first,
      shading_df     = shading_df
    )
    
    cols[[i]] <- col_plot
  }
  
  grid_plot <- (cols[[1]] | cols[[2]] | cols[[3]]) +
    plot_layout(guides = "collect") +
    plot_annotation(
      tag_levels = 'a',
      theme = theme(
        legend.position = "right",
        legend.box = "vertical",
        plot.tag = element_text(size = base_font_size + 2, face = "bold")
      )
    ) &
    theme(legend.position = "right")
  
  return(grid_plot)
}