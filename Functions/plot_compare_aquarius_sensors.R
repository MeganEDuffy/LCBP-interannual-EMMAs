#################
# LOAD PACKAGES #
#################

library(tidyverse)
library(lubridate)
library(cowplot)

plot_compare_aquarius_sensors <- function(site1_name, s1_q_file, s1_doc_file, s1_turb_file, site1_samp_dates,
                                          site2_name, s2_q_file, s2_doc_file, s2_turb_file, site2_samp_dates,
                                          plot_range = c("2022-10-01", "2023-05-01"),
                                          q_lim = c(0, 15), doc_lim = c(0, 35), turb_lim = c(0, 1000),
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
  # 2. EXACT AQUARIUS LOADER HELPER (WITH DIAGNOSTICS)
  # -------------------------------------------------------------------
  read_aq <- function(path) {
    if (is.null(path) || path == "" || !file.exists(path)) {
      message("\n[!] File not found or path empty: ", path)
      return(NULL)
    }
    
    message("\n--- Processing: ", basename(path), " ---")
    
    # 1. Read the raw data
    df <- read.csv(path, comment.char = "#", stringsAsFactors = FALSE)
    message("1. Initial rows read: ", nrow(df))
    message("   Columns found: ", paste(names(df), collapse = ", "))
    
    # Check if the expected columns even exist after reading
    if (!"ISO.8601.UTC" %in% names(df) || !"Value" %in% names(df)) {
      message("   [!] ERROR: Cannot find 'ISO.8601.UTC' or 'Value' columns! Check the column names above.")
      return(NULL)
    }
    
    # 2. Parse the dates and numeric values
    df_processed <- df %>%
      mutate(
        timestamp = ymd_hms(ISO.8601.UTC, tz = "UTC"),
        value = as.numeric(Value)
      )
    
    message("2. Rows with successfully parsed timestamps: ", sum(!is.na(df_processed$timestamp)))
    message("   Rows with successfully parsed numeric values: ", sum(!is.na(df_processed$value)))
    
    # 3. Filter by your target date range
    df_filtered <- df_processed %>%
      filter(!is.na(timestamp), timestamp >= start_date, timestamp <= end_date)
    
    message("3. Rows remaining AFTER filtering for plot_range (", start_date, " to ", end_date, "): ", nrow(df_filtered))
    
    # 4. Final Sanity Check on the filtered data
    if(nrow(df_filtered) > 0) {
      message("   Date Range: ", min(df_filtered$timestamp), " to ", max(df_filtered$timestamp))
      message("   Value Range: ", min(df_filtered$value, na.rm=TRUE), " to ", max(df_filtered$value, na.rm=TRUE))
    } else {
      message("   [!] WARNING: Zero rows remaining to plot.")
    }
    
    return(df_filtered %>% select(timestamp, value))
  }
  
  # -------------------------------------------------------------------
  # 3. LOAD SITE DATA
  # -------------------------------------------------------------------
  # Site 1 Data
  s1_q    <- read_aq(s1_q_file)
  s1_doc  <- read_aq(s1_doc_file)
  s1_turb <- read_aq(s1_turb_file)
  
  # Site 2 Data
  s2_q    <- read_aq(s2_q_file)
  s2_doc  <- read_aq(s2_doc_file)
  s2_turb <- read_aq(s2_turb_file)
  
  # -------------------------------------------------------------------
  # 4. PANEL BUILDER HELPER
  # -------------------------------------------------------------------
  build_panel <- function(df, y_label, y_lims, l_color, samp_dates, is_bottom = FALSE, title = NULL) {
    
    p <- ggplot() +
      # 1. Winter Shading
      geom_rect(data = winter_shade, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax), 
                fill = "#ADD8E6", alpha = 0.3, inherit.aes = FALSE)
      
      # 2. Sample Event Shading
      if (!is.null(samp_dates) && nrow(samp_dates) > 0) {
        p <- p + geom_rect(data = samp_dates, aes(xmin = start, xmax = end, ymin = -Inf, ymax = Inf), 
                           fill = "yellow", alpha = 0.35, inherit.aes = FALSE)
      }
      
      # 3. Sensor Data Line
      if (!is.null(df) && nrow(df) > 0) {
        p <- p + geom_line(data = df, aes(x = timestamp, y = value), color = l_color, linewidth = 0.8, na.rm = TRUE)
      }
      
      # Formatting
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
  # 5. BUILD INDIVIDUAL PANELS
  # -------------------------------------------------------------------
  # Site 1 Panels
  p_q1    <- build_panel(s1_q, "Q (m³/s)", q_lim, "blue", site1_samp_dates, title = paste(site1_name, "Brook"))
  p_doc1  <- build_panel(s1_doc, "DOC (mg/L)", doc_lim, "saddlebrown", site1_samp_dates)
  p_turb1 <- build_panel(s1_turb, "Turbidity (NTU)", turb_lim, "darkorange3", site1_samp_dates, is_bottom = TRUE)
  
  # Site 2 Panels
  p_q2    <- build_panel(s2_q, "Q (m³/s)", q_lim, "blue", site2_samp_dates, title = paste(site2_name, "Brook"))
  p_doc2  <- build_panel(s2_doc, "DOC (mg/L)", doc_lim, "saddlebrown", site2_samp_dates)
  p_turb2 <- build_panel(s2_turb, "Turbidity (NTU)", turb_lim, "darkorange3", site2_samp_dates, is_bottom = TRUE)
  
  # -------------------------------------------------------------------
  # 6. STITCH WITH COWPLOT
  # -------------------------------------------------------------------
  left_col <- plot_grid(p_q1, p_doc1, p_turb1, ncol = 1, align = "v", 
                        labels = c("a)", "b)", "c)"), label_x = -0.02)
  
  right_col <- plot_grid(p_q2, p_doc2, p_turb2, ncol = 1, align = "v", 
                         labels = c("d)", "e)", "f)"), label_x = -0.02)
  
  final_composite <- plot_grid(left_col, right_col, ncol = 2)
  
  padded_final <- ggdraw(final_composite) +
    theme(plot.margin = margin(t = 10, r = 15, b = 10, l = 25, unit = "pt"))
  
  return(padded_final)
}