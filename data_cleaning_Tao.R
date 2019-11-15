library(readr)
library(tidyverse)

# import and code data -------------------------------------------------------------

policy <- read_csv("policies.csv")
driver <- read_csv('drivers.csv')
vehicle <- read_csv('vehicles.csv')

policy$quoted_amt <- as.numeric(gsub('[$,]', '', policy$quoted_amt)) # transfer dollar to number

policy_clean <- policy %>%
  select(-X1) %>%
  mutate(policy_NA = rowSums(is.na(.))) %>%
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
         Parking_street = if_else(primary_parking == 'street', 1, 0)) %>%
  mutate(count = 1) %>%
  spread(state_id, count, fill = 0) %>%
  select(-Prior_carrier_grp, -Cov_package_type, -CAT_zone,
         -primary_parking, -Quote_dt, -zip, -county_name, -Agent_cd)

driver_clean <- driver %>%
  select(-X1) %>%
  mutate(driver_NA = rowSums(is.na(.))) %>%
  mutate(gender_male = if_else(gender == 'M', 1, 0),
         gender_female = if_else(gender == 'F', 1, 0),
         liv_own = if_else(living_status == 'own', 1, 0, missing = 0),
         liv_dep = if_else(living_status == 'dependent', 1, 0, missing = 0),
         liv_rent = if_else(living_status == 'rent', 1, 0, missing = 0),
         age_young = if_else(age < 18, 1, 0),
         age_adult = if_else(age >= 25 & age < 65, 1, 0),
         age_old = if_else(age >= 65, 1, 0)) %>%
  group_by(policy_id) %>%
  summarise(num_male = sum(gender_male),
            num_female = sum(gender_female),
            act_num_drivers = n(),
            num_liv_own = sum(liv_own),
            num_liv_dep = sum(liv_dep),
            num_liv_rent = sum(liv_rent),
            mean_safe = mean(safty_rating, na.rm = T),
            mean_edu = mean(high_education_ind, na.rm = T),
            num_young = sum(age_young),
            num_adult = sum(age_adult),
            num_old = sum(age_old),
            mean_age = mean(age),
            driver_NA = mean(driver_NA))

driver_clean$mean_safe[is.na(driver_clean$mean_safe)] <- NA
driver_clean$mean_edu[is.na(driver_clean$mean_edu)] <- NA

veh_brand <- vehicle %>%
  select(policy_id, make_model) %>%
  separate(make_model, c('brand', 'model'), sep = ' : ') %>%
  group_by(policy_id, brand) %>%
  summarise(count = n()) %>%
  spread(brand, count, fill = 0)

vehicle_clean <- vehicle %>%
  select(-X1, -make_model) %>%
  mutate(veh_NA = rowSums(is.na(.))) %>%
  mutate(lease = if_else(ownership_type == 'leased', 1, 0, missing = 0),
         loan = if_else(ownership_type == 'loaned', 1, 0, missing = 0),
         own = if_else(ownership_type == 'owned', 1, 0, missing = 0)) %>%
  group_by(policy_id) %>%
  summarise(act_num_leased_veh = sum(lease),
           act_num_loaned_veh = sum(loan),
           act_num_owned_veh = sum(own),
           act_num_veh = n(),
           mean_car_age = mean(age, na.rm = T),
           veh_NA = mean(veh_NA)) %>%
  left_join(veh_brand, by = 'policy_id')

vehicle_clean$mean_car_age[is.na(vehicle_clean$mean_car_age)] <- NA

merge_policy <- policy_clean %>%
  left_join(driver_clean, by = 'policy_id') %>%
  left_join(vehicle_clean, by = 'policy_id') %>%
  mutate(num_NA = policy_NA + veh_NA + driver_NA) %>%
  select(-policy_NA, -veh_NA, -driver_NA)

train <- merge_policy %>%
  filter(split == 'Train') %>%
  select(-split, -policy_id) %>%
  select(convert_ind, everything())

test <- merge_policy %>%
  filter(split == 'Test') %>%
  select(-split, -convert_ind) %>%
  select(policy_id, everything())

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

