#################
# LOAD PACKAGES #
#################

library(tidyverse)
library(lubridate)
library(cowplot)

plot_compare_winter_soils <- function(site1_name, met_file1, snow_file1, soil_file1, q_file_1,
                                      site2_name, met_file2, snow_file2, soil_file2, q_file_2,
                                      resin_file,
                                      site1_samp_dates = NULL, 
                                      site2_samp_dates = NULL, 
                                      plot_range = c("2022-06-01", "2023-04-30"),
                                      base_font_size = 14) {
  
  # -------------------------------------------------------------------
  # 1. TIME BOUNDS & DATE PREP
  # -------------------------------------------------------------------
  start_date <- as.POSIXct(plot_range[1], tz = "UTC")
  end_date   <- as.POSIXct(plot_range[2], tz = "UTC")
  
  # -------------------------------------------------------------------
  # 2. INTERNAL HELPERS FOR DATA LOADING
  # -------------------------------------------------------------------
  load_met <- function(m_file) {
    dat <- read.csv(m_file, comment.char = "#") %>%
      rename_with(~ "Timestamp", matches("Timestamp|datetime|Date", ignore.case = TRUE)) %>%
      rename_with(~ "Precip", matches("Precip|Precip_Increm", ignore.case = TRUE)) %>%
      rename_with(~ "Air_Temp", matches("Air_Temp|AirTemp", ignore.case = TRUE)) %>%
      mutate(datetime = parse_date_time(Timestamp, orders = c("ymd HMS", "ymd HM", "mdy HM", "mdy HMS"), tz = "UTC")) %>%
      filter(!is.na(datetime), datetime >= start_date & datetime <= end_date)
    
    dat_daily <- dat %>%
      mutate(date = as.Date(datetime, tz = "UTC")) %>%
      group_by(date) %>%
      summarise(Air_Temp_daily = mean(Air_Temp, na.rm = TRUE), .groups = "drop") %>%
      mutate(datetime = as.POSIXct(paste(date, "12:00:00"), tz = "UTC"))
    
    return(list(raw = dat, daily = dat_daily))
  }
  
  load_snow <- function(sn_file) {
    read.csv(sn_file, comment.char = "#") %>%
      rename_with(~ "snow_cm", matches("modeled_snow_depth_cm|snowpack_cm|Snowpack Depth cm", ignore.case = TRUE)) %>%
      rename_with(~ "Date_col", matches("Date|DATE|datetime|Timestamp", ignore.case = TRUE)) %>%
      mutate(
        datetime = parse_date_time(Date_col, orders = c("ymd", "mdy", "ymd HMS", "mdy HM"), tz = "UTC"),
        snow_cm = as.numeric(gsub("--", NA, as.character(snow_cm)))
      ) %>%
      filter(!is.na(datetime), datetime >= start_date & datetime <= end_date)
  }
  
  load_soil <- function(sl_file) {
    read.csv(sl_file, comment.char = "#") %>%
      rename_with(~ "Timestamp", matches("Timestamp|dateTimeText|datetime", ignore.case = TRUE)) %>%
      mutate(datetime = parse_date_time(Timestamp, orders = c("ymd HMS", "ymd HM", "mdy HM", "mdy HMS"), tz = "UTC")) %>%
      filter(!is.na(datetime), datetime >= start_date & datetime <= end_date)
  }
  
  load_discharge <- function(q_file) {
    dat_q <- read.csv(q_file, comment.char = "#")
    q_time_col <- names(dat_q)[grep("datetime|timestamp|ISO|Date", names(dat_q), ignore.case = TRUE)][1]
    q_val_col  <- names(dat_q)[grep("q_cms|Value", names(dat_q), ignore.case = TRUE)][1]
    
    dat_q %>% 
      rename(timestamp = all_of(q_time_col), q_cms = all_of(q_val_col)) %>% 
      mutate(
        datetime = parse_date_time(timestamp, orders = c("ymd HMS", "ymd HM", "mdy HMS", "mdy HM"), tz = "UTC"),
        q_cms = as.numeric(q_cms)
      ) %>%
      filter(!is.na(datetime), datetime >= start_date & datetime <= end_date)
  }
  
  # -------------------------------------------------------------------
  # 3. LOAD ALL DATA
  # -------------------------------------------------------------------
  met1   <- load_met(met_file1)
  snow1  <- load_snow(snow_file1)
  soil1  <- load_soil(soil_file1)
  q1     <- load_discharge(q_file_1)
  
  met2   <- load_met(met_file2)
  snow2  <- load_snow(snow_file2)
  soil2  <- load_soil(soil_file2)
  q2     <- load_discharge(q_file_2)
  
  dat_resin <- read.csv(resin_file, comment.char = "#") %>%
    mutate(
      Date_str = paste(Month, "1", Year),
      Date = parse_date_time(Date_str, orders = "b d Y", tz = "UTC"),
      ug_cm2_month = as.numeric(gsub(",", "", as.character(ug_cm2_month)))
    ) %>%
    filter(!is.na(Date), Date >= start_date & Date <= end_date) %>%
    mutate(month_label = factor(format(Date, "%b"), levels = month.abb[c(6:12, 1:5)]))
  
  # -------------------------------------------------------------------
  # 4. CALCULATE GLOBAL AXIS LIMITS (For 1-to-1 Comparability)
  # -------------------------------------------------------------------
  max_p <- max(c(met1$raw$Precip, met2$raw$Precip), na.rm = TRUE)
  if (max_p == 0 || !is.finite(max_p)) max_p <- 10
  
  temp_min <- -30
  temp_max <- 30
  temp_range <- temp_max - temp_min
  
  max_snow <- max(c(snow1$snow_cm, snow2$snow_cm), na.rm = TRUE)
  if (max_snow == 0 || !is.finite(max_snow)) max_snow <- 50
  
  # Fixed upper limit for discharge secondary axis
  fixed_max_q <- 15
  
  s_temp_col1 <- names(soil1)[grep("Temp|temperature", names(soil1), ignore.case = TRUE)][1]
  s_vwc_col1  <- names(soil1)[grep("VWC|vwc", names(soil1), ignore.case = TRUE)][1]
  
  s_temp_col2 <- names(soil2)[grep("Temp|temperature", names(soil2), ignore.case = TRUE)][1]
  s_vwc_col2  <- names(soil2)[grep("VWC|vwc", names(soil2), ignore.case = TRUE)][1]
  
  max_vwc <- max(c(soil1[[s_vwc_col1]], soil2[[s_vwc_col2]]), na.rm = TRUE)
  max_temp <- max(c(abs(soil1[[s_temp_col1]]), abs(soil2[[s_temp_col2]])), na.rm = TRUE)
  
  if (max_vwc == 0 || !is.finite(max_vwc)) max_vwc <- 0.5
  if (max_temp == 0 || !is.finite(max_temp)) max_temp <- 20
  
  soil_scale <- max_temp / max_vwc 

  # -------------------------------------------------------------------
  # 5. INTERNAL PLOTTING HELPER
  # -------------------------------------------------------------------
  build_site_panels <- function(site_n, dat_m, dat_m_daily, dat_sn, dat_sl, dat_q, v_col, t_col, samp_dates) {
    
    shading_layer <- NULL
    if (!is.null(samp_dates) && nrow(samp_dates) > 0) {
      shading_layer <- geom_rect(
        data = samp_dates,
        aes(xmin = start, xmax = end, ymin = -Inf, ymax = Inf),
        fill = "gold", alpha = 0.25, inherit.aes = FALSE
      )
    }

    # Panel A: Met
    p_met <- ggplot() +
      shading_layer + 
      geom_col(data = dat_m, aes(x = datetime, y = Precip), fill = "#2F4F4F", alpha = 0.7, width = 86400) +
      geom_line(data = dat_m_daily, aes(x = datetime, y = max_p - (Air_Temp_daily - temp_min) * (max_p / temp_range)), 
                color = "firebrick", linewidth = 0.8) +
      scale_y_reverse(
        name = "Precip (mm)", limits = c(max_p, 0),
        sec.axis = sec_axis(~ temp_min + (max_p - .) * (temp_range / max_p), name = "Air Temp (°C)")
      ) +
      scale_x_datetime(limits = c(start_date, end_date), date_labels = "") +
      labs(title = paste(site_n, "- Total precip & daily air temp")) +
      theme_minimal(base_size = base_font_size) +
      theme(
        axis.title.x = element_blank(), axis.text.x = element_blank(),
        axis.title.y.right = element_text(color = "firebrick"),
        axis.title.y.left = element_text(color = "#2F4F4F")
      )
    
    # Panel B: Snowpack Depth & Scaled Discharge Secondary Axis (Capped at 20 cms)
    p_snow <- ggplot() +
      shading_layer +
      geom_line(data = dat_q, aes(x = datetime, y = (q_cms / fixed_max_q) * max_snow), color = "grey", alpha = 0.75, linewidth = 0.8) +
      geom_line(data = dat_sn, aes(x = datetime, y = snow_cm), color = "blue4", linewidth = 1) +
      scale_y_continuous(
        name = "Snow depth (cm)", 
        limits = c(0, max_snow),
        sec.axis = sec_axis(~ . * (fixed_max_q / max_snow), name = "Discharge (cms)")
      ) +
      scale_x_datetime(limits = c(start_date, end_date), date_labels = "") +
      labs(title = paste(site_n, "- Snowpack depth & discharge")) +
      theme_minimal(base_size = base_font_size) +
      theme(
        axis.title.x = element_blank(), 
        axis.text.x = element_blank(),
        axis.title.y.left = element_text(color = "blue4"),
        axis.title.y.right = element_text(color = "black")
      )
    
    # Panel C: Soil
    p_soil <- ggplot(dat_sl, aes(x = datetime)) +
      shading_layer +
      geom_smooth(aes(y = .data[[v_col]]), color = "dodgerblue3", se = FALSE, linewidth = 1, method = "loess", span = 0.15) +
      geom_smooth(aes(y = .data[[t_col]] / soil_scale), color = "darkorange3", linetype = "dotted", se = FALSE, linewidth = 1, method = "loess", span = 0.15) +
      scale_y_continuous(
        #name = "VWC (m³/m³)", limits = c(0, max_vwc),
        name = "VWC (0-1)", limits = c(0, max_vwc),
        sec.axis = sec_axis(~ . * soil_scale, name = "Soil Temp (°C)")
      ) +
      scale_x_datetime(limits = c(start_date, end_date), date_labels = "%b %Y") +
      labs(title = paste(site_n, "- Soil moisture & temp, 6 cm"), x = "Date") +
      theme_minimal(base_size = base_font_size) +
      theme(
        axis.title.y.left = element_text(color = "dodgerblue3"),
        axis.title.y.right = element_text(color = "darkorange3")
      )
    
    make_boxplot <- function(species_name, y_limit) {
      dat_sub <- dat_resin %>% filter(Site == site_n & Species == species_name)
      ggplot(dat_sub, aes(x = month_label, y = ug_cm2_month)) +
        geom_boxplot(outlier.shape = NA, fill = "gray90", color = "black") +
        scale_x_discrete(drop = FALSE) +
        ylim(0, y_limit) + 
        labs(title = paste(site_n, "-", species_name), x = NULL, y = "µg/cm²/month") +
        theme_minimal(base_size = base_font_size) +
        theme(axis.text.x = element_text(angle = 30, hjust = 1))
    }
    
    p_nh4 <- make_boxplot("Ammonium", 30)
    p_no3 <- make_boxplot("Nitrate", 50)
    p_po4 <- make_boxplot("Phosphate", 15)
    
    return(list(met = p_met, snow = p_snow, soil = p_soil, nh4 = p_nh4, no3 = p_no3, po4 = p_po4))
  }
  
  # -------------------------------------------------------------------
  # 6. GENERATE PANELS & STITCH WITH COWPLOT
  # -------------------------------------------------------------------
  plots1 <- build_site_panels(site1_name, met1$raw, met1$daily, snow1, soil1, q1, s_vwc_col1, s_temp_col1, site1_samp_dates)
  plots2 <- build_site_panels(site2_name, met2$raw, met2$daily, snow2, soil2, q2, s_vwc_col2, s_temp_col2, site2_samp_dates)
  
  left_top <- plot_grid(plots1$met, plots1$snow, plots1$soil, ncol = 1, align = "v", rel_heights = c(1, 1, 1.2), labels = c("a)", "b)", "c)"), label_x = -0.02)
  left_bot <- plot_grid(plots1$nh4, plots1$no3, plots1$po4, ncol = 1, align = "v", labels = c("d)", "e)", "f)"), label_x = -0.02)
  left_col <- plot_grid(left_top, left_bot, ncol = 1, rel_heights = c(2.2, 1.8))
  
  right_top <- plot_grid(plots2$met, plots2$snow, plots2$soil, ncol = 1, align = "v", rel_heights = c(1, 1, 1.2), labels = c("g)", "h)", "i)"), label_x = -0.02)
  right_bot <- plot_grid(plots2$nh4, plots2$no3, plots2$po4, ncol = 1, align = "v", labels = c("j)", "k)", "l)"), label_x = -0.02)
  right_col <- plot_grid(right_top, right_bot, ncol = 1, rel_heights = c(2.2, 1.8))
  
  final_composite <- plot_grid(left_col, right_col, ncol = 2)
  
  padded_final <- ggdraw(final_composite) +
    theme(plot.margin = margin(t = 10, r = 15, b = 10, l = 25, unit = "pt"))
  
  return(padded_final)
}