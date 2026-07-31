#################
# LOAD PACKAGES #
#################

library(tidyverse)
library(lubridate)
library(cowplot)

plot_compare_stream_sensors <- function(site1_name, s1_q_file, s1_no3_file, s1_doc_file, s1_tp_file, s1_turb_file, site1_samp_dates,
                                        site2_name, s2_q_file, s2_no3_file, s2_doc_file, s2_tp_file, s2_turb_file, site2_samp_dates,
                                        plot_range = c("2022-10-01", "2023-05-01"),
                                        q_lim = c(0, 15), 
                                        s1_no3_lim = c(0, 12), s2_no3_lim = c(0, 12), 
                                        doc_lim = c(0, 35), 
                                        s1_tp_lim = c(0, 1), s2_tp_lim = c(0, 1), 
                                        turb_lim = c(0, 1000),
                                        base_font_size = 14) {
  
  # -------------------------------------------------------------------
  # 1. TIME BOUNDS & SHADING PREP
  # -------------------------------------------------------------------
  start_date <- as.POSIXct(plot_range[1], tz = "UTC")
  end_date   <- as.POSIXct(plot_range[2], tz = "UTC")
  
  # Astronomical Winter (Northern Hemisphere 2022-2023)
  winter_shade <- data.frame(
    xmin = as.POSIXct("2022-12-21 00:00:00", tz = "UTC"),
    xmax = as.POSIXct("2023-03-20 23:59:59", tz = "UTC"),
    ymin = -Inf, ymax = Inf
  )
  
  # -------------------------------------------------------------------
  # 2. DATA LOADER HELPER (Updated to use ymd_hms)
  # -------------------------------------------------------------------
  load_sensor <- function(file_path, val_pattern) {
    if (is.null(file_path) || !file.exists(file_path)) return(NULL)
    
    df <- read.csv(file_path, comment.char = "#", stringsAsFactors = FALSE)
    
    t_col <- names(df)[grep("ISO\\.8601\\.UTC|datetime|timestamp|Date", names(df), ignore.case = TRUE)][1]
    v_col <- names(df)[grep(val_pattern, names(df), ignore.case = TRUE)][1]
    
    if (is.na(t_col) || is.na(v_col)) {
      message("Could not locate time or value column in: ", file_path)
      return(NULL)
    }
    
    df %>%
      mutate(
        timestamp = ymd_hms(.data[[t_col]], tz = "UTC"),
        value = as.numeric(.data[[v_col]])
      ) %>%
      filter(!is.na(timestamp), timestamp >= start_date, timestamp <= end_date) %>%
      select(timestamp, value)
  }
  
  # -------------------------------------------------------------------
  # 3. LOAD SITE DATA
  # -------------------------------------------------------------------
  # Site 1 Data
  s1_q    <- load_sensor(s1_q_file, "q_cms_hb|Discharge|Value")
  s1_no3  <- load_sensor(s1_no3_file, "NO3|Nitrate|Value")
  s1_doc  <- load_sensor(s1_doc_file, "doc|Value")
  s1_tp   <- load_sensor(s1_tp_file, "TP|phosphorus|Value")
  s1_turb <- load_sensor(s1_turb_file, "turb|Value")
  
  # Site 2 Data
  s2_q    <- load_sensor(s2_q_file, "q_cms_wb|Discharge|Value")
  s2_no3  <- load_sensor(s2_no3_file, "NO3|Nitrate|Value")
  s2_doc  <- load_sensor(s2_doc_file, "doc|Value")
  s2_tp   <- load_sensor(s2_tp_file, "TP|phosphorus|Value")
  s2_turb <- load_sensor(s2_turb_file, "turb|Value")
  
  # -------------------------------------------------------------------
  # 4. PANEL BUILDER HELPER
  # -------------------------------------------------------------------
  build_panel <- function(df, y_label, y_lims, l_color, samp_dates, is_bottom = FALSE, title = NULL) {
    
    p <- ggplot() +
      geom_rect(data = winter_shade, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax), 
                fill = "#ADD8E6", alpha = 0.3, inherit.aes = FALSE)
      
      if (!is.null(samp_dates) && nrow(samp_dates) > 0) {
        p <- p + geom_rect(data = samp_dates, aes(xmin = start, xmax = end, ymin = -Inf, ymax = Inf), 
                           fill = "yellow", alpha = 0.35, inherit.aes = FALSE)
      }
      
      if (!is.null(df) && nrow(df) > 0) {
        p <- p + geom_line(data = df, aes(x = timestamp, y = value), color = l_color, linewidth = 0.8, na.rm = TRUE)
      }
      
      p <- p + 
        scale_y_continuous(name = y_label, limits = y_lims) +
        scale_x_datetime(limits = c(start_date, end_date), date_labels = if(is_bottom) "%b %Y" else "") +
        theme_minimal(base_size = base_font_size) +
        theme(axis.title.x = element_blank())
      
      if (!is_bottom) p <- p + theme(axis.text.x = element_blank())
      if (!is.null(title)) p <- p + labs(title = title)
      
    return(p)
  }
  
  # -------------------------------------------------------------------
  # 5. BUILD INDIVIDUAL PANELS (Using Site-Specific Limits where needed)
  # -------------------------------------------------------------------
  # Site 1 Panels
  p_q1    <- build_panel(s1_q, "Q (m³/s)", q_lim, "blue", site1_samp_dates, title = paste(site1_name, "Brook"))
  p_no31  <- build_panel(s1_no3, "NO3 (mg/L)", s1_no3_lim, "darkgreen", site1_samp_dates)
  p_doc1  <- build_panel(s1_doc, "DOC (mg/L)", doc_lim, "saddlebrown", site1_samp_dates)
  p_tp1   <- build_panel(s1_tp, "TP (mg/L)", s1_tp_lim, "purple", site1_samp_dates)
  p_turb1 <- build_panel(s1_turb, "Turbidity (NTU)", turb_lim, "darkorange3", site1_samp_dates, is_bottom = TRUE)
  
  # Site 2 Panels
  p_q2    <- build_panel(s2_q, "Q (m³/s)", q_lim, "blue", site2_samp_dates, title = paste(site2_name, "Brook"))
  p_no32  <- build_panel(s2_no3, "NO3 (mg/L)", s2_no3_lim, "darkgreen", site2_samp_dates)
  p_doc2  <- build_panel(s2_doc, "DOC (mg/L)", doc_lim, "saddlebrown", site2_samp_dates)
  p_tp2   <- build_panel(s2_tp, "TP (mg/L)", s2_tp_lim, "purple", site2_samp_dates)
  p_turb2 <- build_panel(s2_turb, "Turbidity (NTU)", turb_lim, "darkorange3", site2_samp_dates, is_bottom = TRUE)
  
  # -------------------------------------------------------------------
  # 6. STITCH WITH COWPLOT (Hardcoded label_size = 20)
  # -------------------------------------------------------------------
  left_col <- plot_grid(p_q1, p_no31, p_doc1, p_tp1, p_turb1, ncol = 1, align = "v", 
                        labels = c("a)", "b)", "c)", "d)", "e)"), label_x = -0.02, label_size = 20)
  
  right_col <- plot_grid(p_q2, p_no32, p_doc2, p_tp2, p_turb2, ncol = 1, align = "v", 
                         labels = c("f)", "g)", "h)", "i)", "j)"), label_x = -0.02, label_size = 20)
  
  final_composite <- plot_grid(left_col, right_col, ncol = 2)
  
  padded_final <- ggdraw(final_composite) +
    theme(plot.margin = margin(t = 10, r = 15, b = 10, l = 25, unit = "pt"))
  
  return(padded_final)
}