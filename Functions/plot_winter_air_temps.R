#########################################################
# R function to calc and plot winter ave air temps ######
# Also plots endmember tracer geochemistry ##############
# Megan Duffy - Adair Lab, UVM ##########################
# last updated 2026-07-30 ###############################
#########################################################


#################
# LOAD PACKAGES #
#################

# Import libraries
library(fs)
library(dplyr)
library(readr)
library(lubridate)

# Set directories
home_dir <- Sys.getenv("HOME")
repo_dir <- file.path(home_dir, "OneDrive/git-repos/LCBP-interannual-EMMAs")
# source(file.path(repo_dir, "Functions/plot_winter_air_temps.R"))
data_dir <- file.path(repo_dir, "Data/") 

# 1. Define the Function
calculate_met_stats <- function(site_name, start_datetime, end_datetime, met_data_path) {
  
  # Read in the CSV file
  # show_col_types = FALSE suppresses the verbose column type message
  df <- read_csv(met_data_path, show_col_types = FALSE)
  
  # Parse the input datetime strings (MM-DD-YYYY HH:MM:SS)
  start_dt <- mdy_hms(start_datetime)
  end_dt <- mdy_hms(end_datetime)
  
  # Parse the dataframe's Timestamp column (YYYY-MM-DD HH:MM:SS)
  df <- df %>%
    mutate(Timestamp = ymd_hms(Timestamp))
  
  # Filter for the target datetime window and calculate statistics
  results <- df %>%
    filter(Timestamp >= start_dt & Timestamp <= end_dt) %>%
    summarise(
      Site = site_name,
      Start_Time = start_dt,
      End_Time = end_dt,
      Mean_Air_Temp = mean(Air_Temp, na.rm = TRUE),
      Mean_Precip = mean(Precip_Increm, na.rm = TRUE),
      Total_Precip = sum(Precip_Increm, na.rm = TRUE),
      Observations = n() # Counts the number of rows in the window
    )
  
  return(results)
}



