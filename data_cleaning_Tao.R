library(readr)
library(tidyverse)

# import and code data -------------------------------------------------------------

policy <- read_csv("policies.csv")
driver <- read_csv('drivers.csv')
vehicle <- read_csv('vehicles.csv')

policy$quoted_amt <- as.numeric(gsub('[$,]', '', policies$quoted_amt)) # transfer dollar to number

policy <- policy %>%
  select(-X1) %>%
  mutate(discount = if_else(discount == 'Yes', 1, 0),
         Home_policy_ind = if_else(Home_policy_ind == 'Y', 1, 0),
         Carrier1 = if_else(Prior_carrier_grp == 'Carrier_1', 1, 0, missing = 0),
         Carrier2 = if_else(Prior_carrier_grp == 'Carrier_2', 1, 0, missing = 0),
         Carrier3 = if_else(Prior_carrier_grp == 'Carrier_3', 1, 0, missing = 0),
         Carrier4 = if_else(Prior_carrier_grp == 'Carrier_4', 1, 0, missing = 0),
         Carrier5 = if_else(Prior_carrier_grp == 'Carrier_5', 1, 0, missing = 0),
         Carrier6 = if_else(Prior_carrier_grp == 'Carrier_6', 1, 0, missing = 0),
         Carrier7 = if_else(Prior_carrier_grp == 'Carrier_7', 1, 0, missing = 0),
         Carrier8 = if_else(Prior_carrier_grp == 'Carrier_8', 1, 0, missing = 0),
         Package_high = if_else(Cov_package_type == 'High', 1, 0),
         Package_medium = if_else(Cov_package_type == 'Medium', 1, 0),
         Package_low = if_else(Cov_package_type == 'Low', 1, 0),
         Catzone1 = if_else(CAT_zone == 1, 1, 0),
         Catzone2 = if_else(CAT_zone == 2, 1, 0),
         Catzone3 = if_else(CAT_zone == 3, 1, 0),
         Catzone4 = if_else(CAT_zone == 4, 1, 0),
         Catzone5 = if_else(CAT_zone == 5, 1, 0),
         Parking_home = if_else(primary_parking == 'home/driveway', 1, 0),
         Parking_garage = if_else(primary_parking == 'parking garage', 1, 0),
         Parking_street = if_else(primary_parking == 'street', 1, 0))

driver <- driver %>%
  select(-X1) %>%
  mutate(gender = if_else(gender == 'M', 1, 0, missing = 0),
         liv_own = if_else(living_status == 'own', 1, 0, missing = 0),
         liv_dep = if_else(living_status == 'dependent', 1, 0, missing = 0),
         liv_rent = if_else(living_status == 'rent', 1, 0, missing = 0)) %>%
  group_by(policy_id) %>%
  summarise(mean_gender = mean(gender),
            act_num_drivers = n(),
            num_liv_own = sum(liv_own),
            num_liv_dep = sum(liv_dep),
            num_liv_rent = sum(liv_rent),
            mean_age = mean(age),
            mean_safe = mean(safty_rating, na.rm = T),
            mean_edu = mean(high_education_ind, na.rm = T))

driver$mean_safe[is.na(driver$mean_safe)] <- NA
driver$mean_edu[is.na(driver$mean_edu)] <- NA

vehicle <- vehicle %>%
  select(-X1) %>%
  mutate(lease = if_else(ownership_type == 'leased', 1, 0, missing = 0),
         loan = if_else(ownership_type == 'loaned', 1, 0, missing = 0),
         own = if_else(ownership_type == 'owned', 1, 0, missing = 0)) %>%
  group_by(policy_id) %>%
  summarise(act_num_leased_veh = sum(lease),
           act_num_loaned_veh = sum(loan),
           act_num_owned_veh = sum(own),
           mean_car_age = mean(age, na.rm = T))

vehicle$mean_car_age[is.na(vehicle$mean_car_age)] <- NA

merge_policy <- policy %>%
  left_join(driver, by = 'policy_id') %>%
  left_join(vehicle, by = 'policy_id')

train <- filter(merge_policy, split == 'Train')
test <- filter(merge_policy, split ==  'Test')

# create 10 folds into variable cv_index
set.seed(8051)
policy_one <- filter(train, convert_ind == 1)
policy_zero <- filter(train, convert_ind == 0)
policy_one <- policy_one[sample(nrow(policy_one)), ]
policy_zero <- policy_zero[sample(nrow(policy_zero)), ]
policy_one$cv_index <- cut(seq(1, nrow(policy_one)), breaks = 10, labels = F)
policy_zero$cv_index <- cut(seq(1, nrow(policy_zero)), breaks = 10, labels = F)

train <- rbind(policy_one, policy_zero)

save(train, test,  file = 'tao.RData')
