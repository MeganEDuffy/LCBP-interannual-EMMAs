#################
# LOAD PACKAGES #
#################
library(tidyverse)
library(lubridate)
library(data.table)
library(ggplot2)
library(patchwork)

plot_single_event_emma <- function(site_name, 
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
                                   snow_lim = c(0, 50),      
                                   air_temp_lim = c(-15, 20),
                                   precip_lim = c(0, 10),
                                   soil_vwc_lim = c(0, 0.5), 
                                   soil_temp_lim = c(-5, 20), 
                                   base_font_size = 14,
                                   event_title = NULL,
                                   shading_df = NULL) {
  
  start_date <- as.POSIXct(plot_range[1], tz = "UTC")
  end_date   <- as.POSIXct(plot_range[2], tz = "UTC")
  
  # 1. LOAD & PREP MET DATA
  dat_met <- read.csv(met_file, comment.char = "#") %>%
    rename_with(~ "Timestamp", matches("Timestamp|datetime|Date", ignore.case = TRUE)) %>%
    rename_with(~ "Air_Temp", matches("Air_Temp|AirTemp", ignore.case = TRUE)) %>%
    rename_with(~ "Precip", matches("Precip|Precip_Increm", ignore.case = TRUE)) %>%
    mutate(
      datetime = parse_date_time(Timestamp, orders = c("ymd HMS", "ymd HM", "mdy HM", "mdy HMS", "ymd"), tz = "UTC"),
      Precip = as.numeric(Precip)
    ) %>%
    filter(!is.na(datetime), datetime >= start_date & datetime <= end_date)
  
  met_daily <- dat_met %>%
    mutate(date = as.Date(datetime, tz = "UTC")) %>%
    group_by(date) %>%
    summarise(Air_Temp_daily = mean(Air_Temp, na.rm = TRUE), .groups = "drop") %>%
    mutate(datetime = as.POSIXct(paste(date, "12:00:00"), tz = "UTC"))

  temp_min <- air_temp_lim[1]
  temp_max <- air_temp_lim[2]
  temp_range <- temp_max - temp_min
  
  snow_min <- snow_lim[1]
  snow_max <- snow_lim[2]
  snow_range <- snow_max - snow_min

  temp_to_snow_scale <- snow_range / temp_range
  precip_max <- precip_lim[2]
  precip_height_fraction <- 0.4 
  precip_pixel_span <- snow_range * precip_height_fraction
  precip_scale <- precip_pixel_span / precip_max

  # 2. LOAD & PREP SNOW DATA
  dat_snow <- read.csv(snow_file, comment.char = "#") %>%
    rename_with(~ "snow_cm", matches("modeled_snow_depth_cm|snowpack_cm|Snowpack Depth cm", ignore.case = TRUE)) %>%
    rename_with(~ "Date_col", matches("Date|DATE|datetime|Timestamp", ignore.case = TRUE)) %>%
    mutate(
      datetime = parse_date_time(Date_col, orders = c("ymd", "mdy", "ymd HMS", "mdy HM"), tz = "UTC"),
      snow_cm = as.numeric(gsub("--", NA, as.character(snow_cm)))
    ) %>%
    filter(!is.na(datetime), datetime >= start_date & datetime <= end_date)

  # 3. LOAD & PREP SOIL DATA
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

  # 4. LOAD & PREP DISCHARGE & EMMA FRACTIONS
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

  # 5. LOAD SENSOR HELPER FOR CHEM
  load_sensor <- function(file_path, val_pattern) {
    if (is.null(file_path) || !file.exists(file_path)) return(NULL)
    df <- read.csv(file_path, comment.char = "#", stringsAsFactors = FALSE)
    t_col <- names(df)[grep("ISO\\.8601\\.UTC|datetime|timestamp|Date", names(df), ignore.case = TRUE)][1]
    v_col <- names(df)[grep(val_pattern, names(df), ignore.case = TRUE)][1]
    if (is.na(t_col) || (!is.null(v_col) && is.na(v_col))) return(NULL)
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

  shading_layer <- NULL
  if (!is.null(shading_df) && nrow(shading_df) > 0) {
    shading_layer <- geom_rect(
      data = shading_df,
      aes(xmin = start, xmax = end, ymin = -Inf, ymax = Inf),
      fill = "grey", alpha = 0.20, inherit.aes = FALSE
    )
  }

  # --- PANEL 1: MET & SNOW ---
  p_snow <- ggplot() +
    shading_layer +
    { if ("Precip" %in% names(dat_met) && nrow(dat_met) > 0) 
        geom_linerange(data = dat_met, aes(x = datetime, ymin = snow_max, ymax = snow_max - Precip * precip_scale, color = "Total scaled precipitation"), linewidth = 1.2) } +
    geom_line(data = dat_snow, aes(x = datetime, y = snow_cm, color = "Snow Depth", linetype = "Snow Depth"), linewidth = 1.2) +
    geom_line(data = met_daily, aes(x = datetime, y = (Air_Temp_daily - temp_min) * temp_to_snow_scale + snow_min, color = "Air Temp", linetype = "Air Temp"), linewidth = 1.0) +
    scale_y_continuous(
      name = "Snow depth (cm)", limits = snow_lim, expand = c(0, 0),
      sec.axis = sec_axis(~ (. - snow_min) / temp_to_snow_scale + temp_min, name = "Air Temp (°C)")
    ) +
    scale_x_datetime(limits = c(start_date, end_date), date_labels = "") +
    scale_color_manual(values = c("Snow Depth" = "blue4", "Air Temp" = "firebrick", "Total scaled precipitation" = "dodgerblue")) +
    scale_linetype_manual(values = c("Snow Depth" = "solid", "Air Temp" = "dotted")) +
    theme_minimal(base_size = base_font_size) +
    labs(title = event_title, color = NULL, linetype = NULL) +
    theme(
      axis.title.x = element_blank(), 
      axis.text.x = element_blank(), 
      plot.title = element_text(face = "bold"),
      axis.title.y.left = element_text(color = "blue4"),
      axis.title.y.right = element_text(color = "firebrick"),
      legend.position = "right"
    )

  # --- PANEL 2: EMMA HYDROGRAPH ---
  p_emma <- ggplot() +
    shading_layer +
    geom_col(data = emma_long, aes(x = timestamp, y = q_component, fill = Component), position = "stack", width = 14400, alpha = 0.85) +
    geom_line(data = dat_q, aes(x = timestamp, y = q_cms, linetype = "Total Q"), color = "black", linewidth = 1.1) +
    scale_y_continuous(name = "Stream q (cms) & EMMA", limits = q_lim, expand = c(0, 0)) +
    scale_x_datetime(limits = c(start_date, end_date), date_labels = "") +
    scale_fill_manual(values = c("Groundwater" = "deepskyblue3", "Meltwater" = "gold3", "Soil water" = "firebrick")) +
    scale_linetype_manual(values = c("Total Q" = "solid")) +
    theme_minimal(base_size = base_font_size) + 
    labs(x = "", fill = "EMMA mixing fractions", linetype = NULL) +
    theme(axis.title.x = element_blank(), axis.text.x = element_blank(), legend.position = "right")

  # --- PANEL 3: STREAM CHEMISTRY ---
  p_chem <- ggplot() +
    shading_layer +
    geom_line(data = dat_q_chem_scaled, aes(x = timestamp, y = q_scaled, color = "Scaled Q"), alpha = 0.85, linewidth = 1.5) +
    { if (!is.null(dat_no3) && nrow(dat_no3) > 0) geom_line(data = dat_no3, aes(x = timestamp, y = value, color = "NO3-N"), linewidth = 1.5) } +
    { if (!is.null(dat_doc) && nrow(dat_doc) > 0) geom_line(data = dat_doc, aes(x = timestamp, y = value / doc_scale, color = "DOC"), linewidth = 1.5) } +
    scale_y_continuous(
      name = expression("Stream " ~ NO[3]*"-N (mg/L)"), limits = no3_lim, expand = c(0, 0),
      sec.axis = sec_axis(~ . * doc_scale, name = "Stream DOC (mg/L)")
    ) +
    scale_x_datetime(limits = c(start_date, end_date), date_labels = "") +
    scale_color_manual(
      values = c("NO3-N" = "darkgreen", "DOC" = "saddlebrown", "Scaled Q" = "grey80"),
      breaks = c("NO3-N", "DOC", "Scaled Q"),
      labels = c(expression(NO[3]*"-N"), "DOC", "Scaled Q")
    ) +
    theme_minimal(base_size = base_font_size) +
    labs(x = "", color = NULL) +
    theme(
      axis.title.x = element_blank(),
      axis.text.x = element_blank(),
      axis.title.y.left = element_text(color = "darkgreen"),
      axis.title.y.right = element_text(color = "saddlebrown"),
      legend.position = "right"
    )

  # --- PANEL 4: SOILS ---
  p_soil <- ggplot() +
    shading_layer +
    geom_smooth(data = dat_soil_6cm, aes(x = datetime, y = .data[[s_vwc_col6]], color = "VWC (6 cm)", linetype = "VWC (6 cm)"), se = FALSE, linewidth = 1.4, method = "loess", span = 0.15) +
    { if (!is.null(dat_soil_15cm)) geom_smooth(data = dat_soil_15cm, aes(x = datetime, y = VWC, color = "VWC (15 cm)", linetype = "VWC (15 cm)"), se = FALSE, linewidth = 2, method = "loess", span = 0.15) } +
    geom_smooth(data = dat_soil_6cm, aes(x = datetime, y = (.data[[s_temp_col6]] - soil_temp_lim[1]) / soil_scale + soil_vwc_lim[1], color = "Temp (6 cm)", linetype = "Temp (6 cm)"), se = FALSE, linewidth = 1.4, method = "loess", span = 0.15) +
    { if (!is.null(dat_soil_15cm)) geom_smooth(data = dat_soil_15cm, aes(x = datetime, y = (Soil_Temp - soil_temp_lim[1]) / soil_scale + soil_vwc_lim[1], color = "Temp (15 cm)", linetype = "Temp (15 cm)"), se = FALSE, linewidth = 2, method = "loess", span = 0.15) } +
    scale_y_continuous(
      name = "Soil VWC", limits = soil_vwc_lim, expand = c(0, 0),
      sec.axis = sec_axis(~ (. - soil_vwc_lim[1]) * soil_scale + soil_temp_lim[1], name = "Soil temp (°C)")
    ) +
    scale_x_datetime(limits = c(start_date, end_date), date_labels = "%b %d") +
    scale_color_manual(values = c(
      "VWC (6 cm)" = "dodgerblue3", "VWC (15 cm)" = "dodgerblue4",
      "Temp (6 cm)" = "darkorange3", "Temp (15 cm)" = "chocolate4"
    )) +
    scale_linetype_manual(values = c(
      "VWC (6 cm)" = "solid", "VWC (15 cm)" = "solid",
      "Temp (6 cm)" = "dotted", "Temp (15 cm)" = "dotted"
    )) +
    theme_minimal(base_size = base_font_size) +
    labs(x = "Date", color = NULL, linetype = NULL) +
    theme(
      axis.title.y.left = element_text(color = "dodgerblue3"),
      axis.title.y.right = element_text(color = "darkorange3"),
      legend.position = "right"
    )

  # Stack vertically into a 1x4 column layout, collecting guides and squaring up alignment
  single_column_grid <- (p_snow / p_emma / p_chem / p_soil) + 
    plot_layout(heights = c(0.9, 1.2, 1.1, 1), guides = "collect") &
    theme(
      legend.title = element_text(face = "bold", size = base_font_size - 1),
      legend.box.just = "left",
      legend.margin = margin(35, 35, 35, 35)
    )

  return(single_column_grid)
}