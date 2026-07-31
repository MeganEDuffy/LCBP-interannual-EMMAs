
#########################################################
# R function to plot resin nutrients, met & soil sensors#
# Also plots endmember tracer geochemistry ##############
# Megan Duffy - Adair Lab, UVM ##########################
# last updated 2026-07-30 ###############################
#########################################################


#################
# LOAD PACKAGES #
#################

library(tidyverse)
library(lubridate)
library(cowplot)

plot_winter_soils_and_nutrients <- function(site_name,
                                            met_file,
                                            snow_file,
                                            soil_file,
                                            resin_file,
                                            plot_range = c("2022-06-01", "2023-04-30"),
                                            base_font_size = 14) {
  
  # -------------------------------------------------------------------
  # 1. TIME BOUNDS & DATE PREP
  # -------------------------------------------------------------------
  start_date <- as.POSIXct(plot_range[1], tz = "UTC")
  end_date   <- as.POSIXct(plot_range[2], tz = "UTC")
  
  # -------------------------------------------------------------------
  # 2. LOAD & CLEAN MET DATA (Precipitation & Air Temp)
  # -------------------------------------------------------------------
  dat_met <- read.csv(met_file, comment.char = "#") %>%
    rename_with(~ "Timestamp", matches("Timestamp|datetime|Date", ignore.case = TRUE)) %>%
    rename_with(~ "Precip", matches("Precip|Precip_Increm", ignore.case = TRUE)) %>%
    rename_with(~ "Air_Temp", matches("Air_Temp|AirTemp", ignore.case = TRUE)) %>%
    mutate(
      datetime = parse_date_time(Timestamp, orders = c("ymd HMS", "ymd HM", "mdy HM", "mdy HMS"), tz = "UTC")
    ) %>%
    filter(!is.na(datetime), datetime >= start_date & datetime <= end_date)

  # Calculate daily average air temperatures for smoothing
  dat_met_daily <- dat_met %>%
    mutate(date = as.Date(datetime, tz = "UTC")) %>%
    group_by(date) %>%
    summarise(Air_Temp_daily = mean(Air_Temp, na.rm = TRUE), .groups = "drop") %>%
    mutate(datetime = as.POSIXct(paste(date, "12:00:00"), tz = "UTC"))

  # -------------------------------------------------------------------
  # 3. LOAD & CLEAN SNOWPACK DATA
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
  # 4. LOAD & CLEAN SOIL SENSOR DATA (Temp & VWC)
  # -------------------------------------------------------------------
  dat_soil <- read.csv(soil_file, comment.char = "#") %>%
    rename_with(~ "Timestamp", matches("Timestamp|dateTimeText|datetime", ignore.case = TRUE)) %>%
    mutate(
      datetime = parse_date_time(Timestamp, orders = c("ymd HMS", "ymd HM", "mdy HM"), tz = "UTC")
    ) %>%
    filter(!is.na(datetime), datetime >= start_date & datetime <= end_date)

  soil_temp_col <- names(dat_soil)[grep("Temp|temperature", names(dat_soil), ignore.case = TRUE)][1]
  soil_vwc_col  <- names(dat_soil)[grep("VWC|vwc", names(dat_soil), ignore.case = TRUE)][1]

  # -------------------------------------------------------------------
  # 5. LOAD & CLEAN RESIN NUTRIENT DATA
  # -------------------------------------------------------------------
  dat_resin <- read.csv(resin_file, comment.char = "#") %>%
    filter(Site == site_name) %>%
    mutate(
      Date_str = paste(Month, "1", Year),
      Date = parse_date_time(Date_str, orders = "b d Y", tz = "UTC"),
      ug_cm2_month = as.numeric(gsub(",", "", as.character(ug_cm2_month)))
    ) %>%
    filter(!is.na(Date), Date >= start_date & Date <= end_date) %>%
    mutate(
      month_label = factor(format(Date, "%b"), levels = month.abb[c(6:12, 1:5)])
    )

  # -------------------------------------------------------------------
  # 6. BUILD CONTINUOUS SENSOR PANELS (A-C)
  # -------------------------------------------------------------------
  
  # Panel A: Dual Axis (Precip Inverted Bar + Daily Air Temp Normal Line)
  max_p <- max(dat_met$Precip, na.rm = TRUE)
  if (max_p == 0 || !is.finite(max_p)) max_p <- 10
  
  # Define the anticipated min and max range for your air temperature to set the secondary axis bounds
  temp_min <- -30
  temp_max <- 30
  temp_range <- temp_max - temp_min
  
  p_met <- ggplot() +
    geom_col(data = dat_met, aes(x = datetime, y = Precip), fill = "#2F4F4F", alpha = 0.7, width = 86400) +
    # Map Air Temp onto the reversed primary axis (warm temps approach 0, cold temps approach max_p)
    geom_line(data = dat_met_daily, 
              aes(x = datetime, y = max_p - (Air_Temp_daily - temp_min) * (max_p / temp_range)), 
              color = "firebrick", linewidth = 0.8) +
    scale_y_reverse(
      name = "Precip (mm)",
      limits = c(max_p, 0),
      # Map the reversed axis back to normal temperature readings for the labels
      sec.axis = sec_axis(~ temp_min + (max_p - .) * (temp_range / max_p), name = "Air Temp (°C)")
    ) +
    scale_x_datetime(limits = c(start_date, end_date), date_labels = "") +
    labs(title = "Precipitation & Daily Air Temperature") +
    theme_minimal(base_size = base_font_size) +
    theme(
      axis.title.x = element_blank(), 
      axis.text.x = element_blank(),
      axis.title.y.right = element_text(color = "firebrick"),
      axis.title.y.left = element_text(color = "#2F4F4F")
    )

  # Panel B: Snowpack Depth
  p_snow <- ggplot(dat_snow, aes(x = datetime, y = snow_cm)) +
    geom_line(color = "blue4", linewidth = 1) +
    scale_x_datetime(limits = c(start_date, end_date), date_labels = "") +
    labs(title = "Snowpack Depth (cm)", y = "Depth (cm)") +
    theme_minimal(base_size = base_font_size) +
    theme(axis.title.x = element_blank(), axis.text.x = element_blank())

  # Panel C: Combined Soil Moisture (VWC) and Soil Temperature
  max_vwc <- max(dat_soil[[soil_vwc_col]], na.rm = TRUE)
  max_temp <- max(abs(dat_soil[[soil_temp_col]]), na.rm = TRUE)
  if (max_vwc == 0 || !is.finite(max_vwc)) max_vwc <- 0.5
  if (max_temp == 0 || !is.finite(max_temp)) max_temp <- 20
  
  soil_scale <- max_temp / max_vwc 
  
  p_soil <- ggplot(dat_soil, aes(x = datetime)) +
    geom_smooth(aes(y = .data[[soil_vwc_col]]), color = "dodgerblue3", se = FALSE, linewidth = 1, method = "loess", span = 0.15) +
    # Changed color to a deeper shade: "darkorange3"
    geom_smooth(aes(y = .data[[soil_temp_col]] / soil_scale), color = "darkorange3", linetype = "dotted", se = FALSE, linewidth = 1, method = "loess", span = 0.15) +
    scale_y_continuous(
      name = "VWC (m³/m³)",
      sec.axis = sec_axis(~ . * soil_scale, name = "Soil Temp (°C)")
    ) +
    scale_x_datetime(limits = c(start_date, end_date), date_labels = "%b %Y") +
    labs(title = "Soil Moisture & Temperature", x = "Date") +
    theme_minimal(base_size = base_font_size) +
    theme(
      axis.title.y.left = element_text(color = "dodgerblue3"),
      axis.title.y.right = element_text(color = "darkorange3") # Matched right axis label color to line
    )

  # -------------------------------------------------------------------
  # 7. BUILD RESIN NUTRIENT BOXPLOT PANELS (D-F)
  # -------------------------------------------------------------------
  make_nutrient_boxplot <- function(species_name, y_max) {
    dat_sub <- dat_resin %>% filter(Species == species_name)
    
    ggplot(dat_sub, aes(x = month_label, y = ug_cm2_month)) +
      geom_boxplot(outlier.shape = NA, fill = "gray90", color = "black") +
      scale_x_discrete(drop = FALSE) +
      ylim(0, y_max) +
      labs(
        title = paste("Available", species_name),
        x = NULL,
        y = "µg/cm²/month"
      ) +
      theme_minimal(base_size = base_font_size) +
      theme(axis.text.x = element_text(angle = 30, hjust = 1))
  }

  p_nh4 <- make_nutrient_boxplot("Ammonium", y_max = 30)
  p_no3 <- make_nutrient_boxplot("Nitrate",  y_max = 50)
  p_po4 <- make_nutrient_boxplot("Phosphate", y_max = 15)

  # -------------------------------------------------------------------
  # 8. STACK & ALIGN ALL PANELS WITH COWPLOT
  # -------------------------------------------------------------------
  top_stack <- plot_grid(
    p_met, p_snow, p_soil,
    ncol = 1,
    align = "v",
    rel_heights = c(1, 1, 1.2),
    labels = c("A", "B", "C"),
    label_x = -0.02
  )

  bottom_stack <- plot_grid(
    p_nh4, p_no3, p_po4,
    ncol = 1,
    align = "v",
    labels = c("D", "E", "F"),
    label_x = -0.02
  )

  final_composite <- plot_grid(
    top_stack,
    bottom_stack,
    ncol = 1,
    rel_heights = c(2.2, 1.8)
  )

  padded_final <- ggdraw(final_composite) +
    theme(plot.margin = margin(t = 10, r = 15, b = 10, l = 25, unit = "pt"))

  return(padded_final)
}