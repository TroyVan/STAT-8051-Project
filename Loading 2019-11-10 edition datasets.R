library(readr)

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

# Split train and test policies

policies_train = policies[policies$split == "Train",]
policies_test = policies[policies$split == "Test",]
policies_train$split = NULL
policies_test$split = NULL
policies_test$convert_ind = NULL

# Split train and test drivers

for(i in 1:dim(drivers)[1]){
  drivers$matching_train_policies[i] = length(which(drivers$policy_id[i] == policies_train$policy_id))
  drivers$matching_test_policies[i] = length(which(drivers$policy_id[i] == policies_test$policy_id))
}
remove(i)

# Verify that each driver is matched to either a test policy or a train policy
drivers$total_matching_policies = drivers$matching_train_policies + drivers$matching_test_policies
library(dplyr)
count(drivers, matching_train_policies) # Should be all 0's and 1's
count(drivers, matching_test_policies) # Should be all 0's and 1's
count(drivers, total_matching_policies) # Should be all 1's
drivers$total_matching_policies = NULL

drivers_train = drivers[drivers$matching_train_policies == 1,]
drivers_test = drivers[drivers$matching_test_policies == 1,]
drivers$matching_train_policies = NULL
drivers$matching_test_policies = NULL
drivers_train$matching_train_policies = NULL
drivers_train$matching_test_policies = NULL
drivers_test$matching_train_policies = NULL
drivers_test$matching_test_policies = NULL

# Split train and test vehicles

for(i in 1:dim(vehicles)[1]){
  vehicles$matching_train_policies[i] = length(which(vehicles$policy_id[i] == policies_train$policy_id))
  vehicles$matching_test_policies[i] = length(which(vehicles$policy_id[i] == policies_test$policy_id))
}
remove(i)

# Verify that each vehicle is matched to either a test policy or a train policy
vehicles$total_matching_policies = vehicles$matching_train_policies + vehicles$matching_test_policies
library(dplyr)
count(vehicles, matching_train_policies) # Should be all 0's and 1's
count(vehicles, matching_test_policies) # Should be all 0's and 1's
count(vehicles, total_matching_policies) # Should be all 1's
vehicles$total_matching_policies = NULL

vehicles_train = vehicles[vehicles$matching_train_policies == 1,]
vehicles_test = vehicles[vehicles$matching_test_policies == 1,]
vehicles$matching_train_policies = NULL
vehicles$matching_test_policies = NULL
vehicles_train$matching_train_policies = NULL
vehicles_train$matching_test_policies = NULL
vehicles_test$matching_train_policies = NULL
vehicles_test$matching_test_policies = NULL

# Drop original databases (optional)
remove(drivers)
remove(policies)
remove(vehicles)