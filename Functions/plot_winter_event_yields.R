#################
# LOAD PACKAGES #
#################

library(tidyverse)
library(ggplot2)
library(patchwork)

plot_winter_event_yields <- function(yield_file, base_font_size = 12) {
  
  # --- 1. READ & CLEAN DATA ---
  df <- read.csv(yield_file, check.names = FALSE)
  
  cat_col  <- names(df)[grep("Catchment", names(df), ignore.case = TRUE)][1]
  type_col <- names(df)[grep("Event type", names(df), ignore.case = TRUE)][1]
  
  no3_col  <- names(df)[grep("Nitrate yield", names(df), ignore.case = TRUE)][1]
  tp_col   <- names(df)[grep("Total phosphorus yield", names(df), ignore.case = TRUE)][1]
  doc_col  <- names(df)[grep("DOC yield", names(df), ignore.case = TRUE)][1]
  
  df_clean <- df %>%
    rename(
      Catchment  = all_of(cat_col),
      Event.type = all_of(type_col),
      NO3_norm   = all_of(no3_col),
      TP_norm    = all_of(tp_col),
      DOC_norm   = all_of(doc_col)
    ) %>%
    mutate(
      # Chronological order: ROS -> Thermal -> Final spring melt
      Event.type = factor(Event.type, levels = c("ROS", "Thermal", "Final spring melt")),
      Catchment  = as.factor(Catchment)
    )
  
  # Custom color palette mapped to event types
  event_colors <- c(
    "ROS"               = "blue", 
    "Thermal"           = "red", 
    "Final spring melt" = "green4"
  )
  
  # --- 2. CREATE INDIVIDUAL NUTRIENT PLOTS ---
  
  # Nitrate Yield Panel
  p_no3 <- ggplot(df_clean, aes(x = Event.type, y = NO3_norm, fill = Event.type)) +
    geom_col(position = position_dodge(), width = 0.65, alpha = 0.85, color = "black") +
    facet_wrap(~ Catchment, ncol = 1) +
    scale_fill_manual(values = event_colors) +
    labs(
      title = "a) Nitrate",
      x = NULL,
      y = expression(Normalized~Yield~(kg~NO[3]-N~km^-2~mm^-1))
    ) +
    theme_minimal(base_size = base_font_size) +
    theme(
      panel.background = element_rect(fill = '#FFF5EE', colour = 'black'),
      axis.text.x = element_text(angle = 15, hjust = 1),
      legend.position = "none"
    )
  
  # Total Phosphorus Yield Panel
  p_tp <- ggplot(df_clean, aes(x = Event.type, y = TP_norm, fill = Event.type)) +
    geom_col(position = position_dodge(), width = 0.65, alpha = 0.85, color = "black") +
    facet_wrap(~ Catchment, ncol = 1) +
    scale_fill_manual(values = event_colors) +
    labs(
      title = "b) Total phosphorus",
      x = NULL,
      y = expression(Normalized~Yield~(kg~P~km^-2~mm^-1))
    ) +
    theme_minimal(base_size = base_font_size) +
    theme(
      panel.background = element_rect(fill = '#FFF5EE', colour = 'black'),
      axis.text.x = element_text(angle = 15, hjust = 1),
      legend.position = "none"
    )
  
  # DOC Yield Panel
  p_doc <- ggplot(df_clean, aes(x = Event.type, y = DOC_norm, fill = Event.type)) +
    geom_col(position = position_dodge(), width = 0.65, alpha = 0.85, color = "black") +
    facet_wrap(~ Catchment, ncol = 1) +
    scale_fill_manual(values = event_colors) +
    labs(
      title = "c) DOC",
      x = NULL,
      y = expression(Normalized~Yield~(kg~DOC~km^-2~mm^-1))
    ) +
    theme_minimal(base_size = base_font_size) +
    theme(
      panel.background = element_rect(fill = '#FFF5EE', colour = 'black'),
      axis.text.x = element_text(angle = 15, hjust = 1),
      legend.position = "right"
    )
  
  # --- 3. COMBINE USING PATCHWORK ---
  combined_plot <- (p_no3 | p_tp | p_doc) +
    plot_layout(guides = "collect") &
    theme(legend.position = "bottom")
  
  return(combined_plot)
}