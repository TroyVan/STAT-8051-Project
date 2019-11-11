library(readr)
library(dplyr)

# Same code to load the database as in "Loading datasets (2019-11-10 edition).R"

policies <- read_csv("policies edited 2019-11-10.csv", col_types = cols(
  Agent_cd = col_character(),
  CAT_zone = col_factor(levels = c("1", "2", "3", "4", "5")),
  Cov_package_type = col_factor(levels = c("Low", "Medium", "High")),
  Home_policy_ind = col_logical(),
  Prior_carrier_grp = col_factor(levels = c("Carrier_1", "Carrier_2", "Carrier_3", "Carrier_4", "Carrier_5", "Carrier_6", "Carrier_7", "Carrier_8", "Other")),
  Quote_dt = col_date(format = "%Y-%m-%d"),
  convert_ind = col_logical(),
  county_name = col_character(),
  credit_score = col_integer(),
  discount = col_logical(),
  num_leased_veh = col_integer(),
  num_loaned_veh = col_integer(),
  num_owned_veh = col_integer(),
  number_drivers = col_integer(),
  primary_parking = col_factor(levels = c("home/driveway", "parking garage", "street", "unknown")),
  quoted_amt = col_integer(),
  split = col_factor(levels = c("Test", "Train")),
  state_id = col_factor(levels = c("AL", "CT", "FL", "GA", "MN", "NJ", "NY", "WI")),
  total_number_veh = col_integer(),
  zip = col_character()
))

drivers <- read_csv("drivers edited 2019-11-10.csv", col_types = cols(
  age = col_integer(),
  gender = col_factor(levels = c("F", "M")),
  high_education_ind = col_logical(),
  living_status = col_factor(levels = c("dependent",  "own", "rent")),
  policy_id = col_character(),
  safty_rating = col_integer()
))

vehicles <- read_csv("vehicles edited 2019-11-10.csv", col_types = cols(
  age = col_integer(),
  car_no = col_character(),
  color = col_factor(levels = c("black", "blue", "gray", "red", "silver", "white", "other")),
  ownership_type = col_factor(levels = c("leased", "loaned", "owned")),
  policy_id = col_character()
))

for(i in 1:dim(policies)[1]){
  policies$matching_drivers[i] = length(which(policies$policy_id[i] == drivers$policy_id))
  policies$matching_loaned_vehicles[i] = length(which(policies$policy_id[i] == vehicles$policy_id & vehicles$ownership_type == "loaned"))
  policies$matching_owned_vehicles[i] = length(which(policies$policy_id[i] == vehicles$policy_id & vehicles$ownership_type == "owned"))
  policies$matching_leased_vehicles[i] = length(which(policies$policy_id[i] == vehicles$policy_id & vehicles$ownership_type == "leased"))
  if(i %% 1000 == 0) print(i)
}
policies$total_matching_vehicles = policies$matching_loaned_vehicles + policies$matching_owned_vehicles + policies$matching_leased_vehicles

policies$diff_drivers = policies$matching_drivers - policies$number_drivers
policies$diff_loaned_vehicles = policies$matching_loaned_vehicles - policies$num_loaned_veh
policies$diff_owned_vehicles = policies$matching_owned_vehicles - policies$num_owned_veh
policies$diff_leased_vehicles = policies$matching_leased_vehicles - policies$num_leased_veh
policies$diff_total_vehicles = policies$total_matching_vehicles - policies$total_number_veh

# These should be all zeroes

count(policies, diff_drivers)
count(policies, diff_loaned_vehicles)
count(policies, diff_owned_vehicles)
count(policies, diff_leased_vehicles)
count(policies, diff_total_vehicles)