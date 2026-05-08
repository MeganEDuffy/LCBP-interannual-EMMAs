
#########################################################
# R function to plot stream Q, sC, grab chem for events #
# Megan Duffy - Adair Lab, UVM ##########################
# last updated 2026-05-01 ###############################
#########################################################

#################
# LOAD PACKAGES #
#################

library(tidyverse)
library(viridis)
library(dplyr)
library(lubridate)
library(caTools)  # for numerical integration
library(data.table) # for nearest join of q and ISCO data
library(patchwork)

plot_stream_summary <- function(site_name, 
                                plot_range, 
                                event_bounds, 
                                q_file, 
                                exo_file, 
                                grab_chem_file, 
                                doc_file, 
                                ic_sc_file,
                                q_lim,
                                sc_lim,
                                chem_lim
                                ) {
  
  # 1. READ DATA
  dat_q    <- read.csv(q_file)
  dat_exo  <- read.csv(exo_file)
  dat_isco <- read.csv(grab_chem_file)
  # dat_doc  <- read.csv(doc_file) # Ready for when you add DOC layers
  dat_ic   <- read.csv(ic_sc_file)
  
  # 2. CLEAN CONTINUOUS DATA
  # Prep EXO
  dat_exo <- dat_exo %>% 
    rename(sC = matches("Sp.Cond")) %>%
    mutate(timestamp = ymd_hms(timestamp, tz = "UTC"))
  
  # Prep Q - Using more flexible column selection
  dat_q <- dat_q %>% 
  # This finds the column that contains 'q_cms' regardless of suffix
    rename(timestamp = datetime) %>%
    rename(q_cms = matches("q_cms")) %>% 
    mutate(timestamp = ymd_hms(timestamp, tz = "UTC"))
  
  # Join Q and EXO using data.table for speed
  merged_dt <- as.data.table(dat_q)[as.data.table(dat_exo), roll = "nearest", on = .(timestamp)]

  # Diagnostic for merge
  print(paste("Rows in Q:", nrow(dat_q)))
  print(paste("Rows in EXO:", nrow(dat_exo)))
  print(paste("Rows in Joined Data:", nrow(merged_dt)))
  print(head(merged_dt))
  
  # 3. CLEAN GRAB DATA
  dat_isco <- dat_isco %>%
    mutate(timestamp = mdy_hm(paste(Date, Time))) %>%
    filter(Site == site_name, Type %in% c("Baseflow", "Grab/Isco"))
  
  # 4. PREP SHADING
  shade_df <- data.frame(
    xmin = as.POSIXct(event_bounds$start),
    xmax = as.POSIXct(event_bounds$end),
    ymin = -Inf, ymax = Inf
  )
  
  # 5. PLOTTING
  # Calculate the scaling coefficient dynamically 
  # so the right axis (sC) always matches the left axis (Q)
  coeff_sc <- sc_lim[2] / q_lim[2]

  # --- Filter and Prep IC Lab Data ---
  # Filter for Wade (already handled by site_name) and prep shapes
  dat_ic_clean <- dat_ic %>%
    mutate(timestamp = ymd_hms(timestamp)) %>%
    filter(Site == site_name) %>%
    mutate(MarkerGroup = ifelse(Sample.Type %in% c("Grab/Isco", "Grab", "Isco", "Baseflow"), 
                                "Stream/Base", "Other"))


  # P1: Sensors
  p1 <- ggplot() +
    geom_rect(data = shade_df, aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), fill="yellow", alpha=0.2) +
    geom_line(data = merged_dt, aes(x=timestamp, y=q_cms, color="Discharge"), linewidth=0.7) +
    geom_line(data = merged_dt, aes(x=timestamp, y=sC/coeff_sc, color="Conductivity (Sensor)"), linewidth=0.7) +
    
    # ADDED: Lab IC points with conditional shapes
    geom_point(data = dat_ic_clean, 
               aes(x=timestamp, y=sC_total_uS/coeff_sc, shape=MarkerGroup), 
               color="black", size=2.5) +
    
    scale_y_continuous(name="Q (cms)", 
                       limits=q_lim, 
                       expand=c(0,0),
                       sec.axis = sec_axis(~.*coeff_sc, name="sC (uS/cm)")) +
    scale_x_datetime(limits = plot_range, date_labels = "") +
    
    # Define the shapes: 16 is a solid circle, 15 is a solid square
    scale_shape_manual(values = c("Stream/Base" = 16, "Other" = 15), guide = "none") +
    
    scale_color_manual(values = c("Discharge"="blue", "Conductivity (Sensor)"="red")) +
    theme_bw() + theme(legend.position="none", axis.title.x=element_blank())
  
  # P2: Isotopes
  p2 <- ggplot() +
    geom_rect(data = shade_df, aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), fill="yellow", alpha=0.2) +
    geom_line(data = merged_dt, aes(x=timestamp, y=(q_cms*5)-100), color="grey85", alpha=0.6) +
    geom_point(data = dat_isco, aes(x=timestamp, y=dD, shape="dD"), color="darkgreen", size=2.5) +
    geom_point(data = dat_isco, aes(x=timestamp, y=d18O*7, shape="d18O"), color="darkorange", size=2.5) +
    scale_y_continuous(name=expression(delta*D), limits=c(-100, -60),
                       sec.axis = sec_axis(~./7, name=expression(delta*18*O))) +
    scale_x_datetime(limits = plot_range, date_labels = "") +
    theme_bw() + theme(legend.position="none", axis.title.x=element_blank())
  
  # P3: Geochem
  p3 <- ggplot() +
    geom_rect(data = shade_df, aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), fill="yellow", alpha=0.2) +
    geom_line(data = merged_dt, aes(x=timestamp, y=q_cms), color="grey85", alpha=0.6) +
    geom_point(data = dat_isco, aes(x=timestamp, y=Ca_mg_L, color="Calcium"), size=2.5) +
    geom_point(data = dat_isco, aes(x=timestamp, y=Na_mg_L, color="Sodium"), size=2.5) +
    geom_point(data = dat_isco, aes(x=timestamp, y=Mg_mg_L*5, color="Magnesium"), size=2.5) +
    scale_y_continuous(name="Ca & Na (mg/L)", 
                       limits=chem_lim,
                       sec.axis = sec_axis(~./5, name="Mg (mg/L)")) +
    scale_x_datetime(limits = plot_range, date_labels = "%b %d") +
    scale_color_manual(values = c("Calcium"="darkblue", "Sodium"="darkcyan", "Magnesium"="purple")) +
    theme_bw() + labs(x="Date", color="Tracer") + theme(legend.position="bottom")
  
  # STACK
  final_plot <- p1 / p2 / p3 + plot_layout(heights = c(1, 1, 1))
  return(final_plot)
}