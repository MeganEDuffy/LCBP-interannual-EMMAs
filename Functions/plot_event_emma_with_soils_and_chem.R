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
                                                soil_file,
                                                met_file,
                                                emma_frac_file,
                                                no3_file,
                                                doc_file,
                                                q_lim = c(0, 4),
                                                no3_lim = c(0, 2),
                                                doc_lim = c(0, 20),
                                                base_font_size = 14,
                                                event_title = NULL) {
  
  start_date <- as.POSIXct(plot_range[1], tz = "UTC")
  end_date   <- as.POSIXct(plot_range[2], tz = "UTC")
  
  # -------------------------------------------------------------------
  # 1. LOAD & PREP MET DATA (Air Temp)
  # -------------------------------------------------------------------
  dat_met <- read.csv(met_file, comment.char = "#") %>%
    rename_with(~ "Timestamp", matches("Timestamp|datetime|Date", ignore.case = TRUE)) %>%
    rename_with(~ "Air_Temp", matches("Air_Temp|AirTemp", ignore.case = TRUE)) %>%
    mutate(datetime = parse_date_time(Timestamp, orders = c("ymd HMS", "ymd HM", "mdy HM", "mdy HMS"), tz = "UTC")) %>%
    filter(!is.na(datetime), datetime >= start_date & datetime <= end_date)
  
  met_daily <- dat_met %>%
    mutate(date = as.Date(datetime, tz = "UTC")) %>%
    group_by(date) %>%
    summarise(Air_Temp_daily = mean(Air_Temp, na.rm = TRUE), .groups = "drop") %>%
    mutate(datetime = as.POSIXct(paste(date, "12:00:00"), tz = "UTC"))

  temp_min <- -30
  temp_max <- 30
  temp_range <- temp_max - temp_min

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

  max_snow <- max(dat_snow$snow_cm, na.rm = TRUE)
  if (!is.finite(max_snow) || max_snow == 0) max_snow <- 50

  # -------------------------------------------------------------------
  # 3. LOAD & PREP SOIL DATA
  # -------------------------------------------------------------------
  dat_soil <- read.csv(soil_file, comment.char = "#") %>%
    rename_with(~ "Timestamp", matches("Timestamp|dateTimeText|datetime", ignore.case = TRUE)) %>%
    mutate(datetime = parse_date_time(Timestamp, orders = c("ymd HMS", "ymd HM", "mdy HM", "mdy HMS"), tz = "UTC")) %>%
    filter(!is.na(datetime), datetime >= start_date & datetime <= end_date)

  s_temp_col <- names(dat_soil)[grep("Temp|temperature", names(dat_soil), ignore.case = TRUE)][1]
  s_vwc_col  <- names(dat_soil)[grep("VWC|vwc", names(dat_soil), ignore.case = TRUE)][1]

  max_vwc <- max(dat_soil[[s_vwc_col]], na.rm = TRUE)
  max_temp <- max(abs(dat_soil[[s_temp_col]]), na.rm = TRUE)
  if (!is.finite(max_vwc) || max_vwc == 0) max_vwc <- 0.5
  if (!is.finite(max_temp) || max_temp == 0) max_temp <- 20
  soil_scale <- max_temp / max_vwc 

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
  
  gw_col   <- names(emma_dt)[grep("Groundwater", names(emma_dt), ignore.case = TRUE)][1]
  melt_col <- names(emma_dt)[grep("Snowmelt", names(emma_dt), ignore.case = TRUE)][1]
  soil_col <- names(emma_dt)[grep("Soil", names(emma_dt), ignore.case = TRUE)][1]
  
  emma_data <- emma_dt %>%
    mutate(
      q_gw   = q_cms * .data[[gw_col]],
      q_melt = q_cms * .data[[melt_col]],
      q_soil = q_cms * .data[[soil_col]]
    )

  # -------------------------------------------------------------------
  # 5. LOAD SENSOR HELPER FOR CHEM (DOC & NO3)
  # -------------------------------------------------------------------
  load_sensor <- function(file_path, val_pattern) {
    if (is.null(file_path) || !file.exists(file_path)) return(NULL)
    df <- read.csv(file_path, comment.char = "#", stringsAsFactors = FALSE)
    t_col <- names(df)[grep("ISO\\.8601\\.UTC|datetime|timestamp|Date", names(df), ignore.case = TRUE)][1]
    v_col <- names(df)[grep(val_pattern, names(df), ignore.case = TRUE)][1]
    if (is.na(t_col) || is.na(v_col)) return(NULL)
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

  # Scale factor to display DOC and NO3 cleanly on a dual secondary axis
  doc_scale <- doc_lim[2] / no3_lim[2]

  # -------------------------------------------------------------------
  # 6. BUILD PANELS
  # -------------------------------------------------------------------
  
  # Panel 1: Snowpack Depth & Air Temperature
  p_snow <- ggplot() +
    geom_line(data = dat_snow, aes(x = datetime, y = snow_cm, color = "Snow Depth"), linewidth = 1) +
    geom_line(data = met_daily, aes(x = datetime, y = max_snow - (Air_Temp_daily - temp_min) * (max_snow / temp_range), color = "Air Temp"), linewidth = 0.8) +
    scale_y_continuous(
      name = "Snow (cm)", limits = c(0, max_snow),
      sec.axis = sec_axis(~ temp_min + (max_snow - .) * (temp_range / max_snow), name = "Air Temp (°C)")
    ) +
    scale_x_datetime(limits = c(start_date, end_date), date_labels = "") +
    scale_color_manual(values = c("Snow Depth" = "blue4", "Air Temp" = "firebrick")) +
    theme_bw(base_size = base_font_size) +
    labs(title = event_title, color = NULL) +
    theme(
      axis.title.x = element_blank(), 
      axis.text.x = element_blank(), 
      plot.title = element_text(face = "bold"),
      axis.title.y.right = element_text(color = "firebrick"),
      axis.title.y.left = element_text(color = "blue4"),
      legend.position = "none"
    )

  # Panel 2: Soil Temperature and Moisture
  p_soil <- ggplot(dat_soil, aes(x = datetime)) +
    geom_smooth(aes(y = .data[[s_vwc_col]]), color = "dodgerblue3", se = FALSE, linewidth = 0.9, method = "loess", span = 0.15) +
    geom_smooth(aes(y = .data[[s_temp_col]] / soil_scale), color = "darkorange3", linetype = "dotted", se = FALSE, linewidth = 0.9, method = "loess", span = 0.15) +
    scale_y_continuous(
      name = "VWC", limits = c(0, max_vwc),
      sec.axis = sec_axis(~ . * soil_scale, name = "Temp (°C)")
    ) +
    scale_x_datetime(limits = c(start_date, end_date), date_labels = "") +
    theme_bw(base_size = base_font_size) +
    theme(
      axis.title.x = element_blank(), 
      axis.text.x = element_blank(),
      axis.title.y.left = element_text(color = "dodgerblue3"),
      axis.title.y.right = element_text(color = "darkorange3")
    )

  # Panel 3: EMMA Volumetric Hydrograph
  p_emma <- ggplot() +
    geom_line(data = dat_q, aes(x = timestamp, y = q_cms, linetype = "Total Q"), color = "grey50", linewidth = 0.5) +
    geom_point(data = emma_data, aes(x = timestamp, y = q_gw, color = "Groundwater"), shape = 17, size = 2) +
    geom_line(data = emma_data, aes(x = timestamp, y = q_gw, color = "Groundwater"), alpha = 0.4, linewidth = 0.6) +
    geom_point(data = emma_data, aes(x = timestamp, y = q_melt, color = "Meltwater"), shape = 16, size = 2) +
    geom_line(data = emma_data, aes(x = timestamp, y = q_melt, color = "Meltwater"), alpha = 0.4, linewidth = 0.6) +
    geom_point(data = emma_data, aes(x = timestamp, y = q_soil, color = "Soil water"), shape = 15, size = 2) +
    geom_line(data = emma_data, aes(x = timestamp, y = q_soil, color = "Soil water"), alpha = 0.4, linewidth = 0.6) +
    scale_y_continuous(name = "Q (cms)", limits = q_lim, expand = c(0, 0)) +
    scale_x_datetime(limits = c(start_date, end_date), date_labels = "") +
    scale_color_manual(values = c("Groundwater" = "blue", "Meltwater" = "gold3", "Soil water" = "firebrick")) +
    scale_linetype_manual(values = c("Total Q" = "dashed")) +
    theme_bw(base_size = base_font_size) + 
    labs(x = "", color = "Components", linetype = "") +
    theme(axis.title.x = element_blank(), axis.text.x = element_blank(), legend.position = "bottom")

# Panel 4: Stream Chemistry (DOC & NO3 Dual Axis)
  p_chem <- ggplot() +
    { if (!is.null(dat_no3) && nrow(dat_no3) > 0) geom_line(data = dat_no3, aes(x = timestamp, y = value, color = "NO3"), linewidth = 0.8) } +
    { if (!is.null(dat_doc) && nrow(dat_doc) > 0) geom_line(data = dat_doc, aes(x = timestamp, y = value / doc_scale, color = "DOC"), linewidth = 0.8) } +
    scale_y_continuous(
      name = "NO3 (mg/L)", 
      limits = no3_lim, 
      expand = c(0, 0),
      sec.axis = sec_axis(~ . * doc_scale, name = "DOC (mg/L)") # Removed limits argument here
    ) +
    scale_x_datetime(limits = c(start_date, end_date), date_labels = "%b %d") +
    scale_color_manual(values = c("NO3" = "darkgreen", "DOC" = "saddlebrown")) +
    theme_bw(base_size = base_font_size) +
    labs(x = "Date", color = NULL) +
    theme(
      axis.title.y.left = element_text(color = "darkgreen"),
      axis.title.y.right = element_text(color = "saddlebrown"),
      legend.position = "bottom"
    )
  # -------------------------------------------------------------------
  # 7. STACK VERTICALLY WITH PATCHWORK
  # -------------------------------------------------------------------
  stacked_column <- (p_snow / p_soil / p_emma / p_chem) + 
    plot_layout(heights = c(0.9, 1, 1.2, 1.1), guides = "collect") &
    theme(legend.position = "bottom")

  return(stacked_column)
}

