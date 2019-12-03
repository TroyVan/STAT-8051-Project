library(readr)
library(tidyverse)
library(lubridate)

# import and code data -------------------------------------------------------------

policy <- read_csv("policies.csv")
driver <- read_csv('drivers.csv')
vehicle <- read_csv('vehicles.csv')

policy$quoted_amt <- as.numeric(gsub('[$,]', '', policy$quoted_amt)) # transfer dollar to number

policy_clean <- policy %>%
  select(-X1) %>%
  mutate(policy_NA = rowSums(is.na(.))) %>%
  mutate(
    year = year(Quote_dt),
    month = month(Quote_dt),
    wday = wday(Quote_dt),
    mday = day(Quote_dt)
  ) %>%
  mutate(
    Prior_carrier_grp = as.integer(factor(Prior_carrier_grp)),
    Cov_package_type = as.integer(factor(Cov_package_type)),
    discount = if_else(discount == 'Yes', 1, 0),
    primary_parking = as.integer(factor(primary_parking)),
    CAT_zone = as.integer(factor(CAT_zone)),
    Home_policy_ind = if_else(Home_policy_ind == 'Y', 1, 0),
    zip = as.integer(factor(zip)),
    state_id = as.integer(factor(state_id)),
    county_name = as.integer(factor(county_name)),
    Agent_cd = as.integer(factor(Agent_cd))
  ) %>%
  mutate(
    avg_driver_amt = quoted_amt/number_drivers
  ) %>%
  select(-Quote_dt, -(num_loaned_veh:total_number_veh))


driver_clean <- driver %>%
  select(-X1) %>%
  mutate(driver_NA = rowSums(is.na(.))) %>%
  mutate(
    liv_own = if_else(living_status == 'own', 1, 0, missing = 0),
    liv_dep = if_else(living_status == 'dependent', 1, 0, missing = 0),
    liv_rent = if_else(living_status == 'rent', 1, 0, missing = 0)
  ) %>%
  mutate(gender_male = if_else(gender == 'M', 1, 0)) %>%
  group_by(policy_id) %>%
  summarise(num_male = sum(gender_male),
            mean_safe = mean(safty_rating, na.rm = T),
            mean_edu = mean(high_education_ind, na.rm = T),
            mean_age = mean(age),
            max_age = max(age),
            min_age = min(age),
            driver_NA = mean(driver_NA),
            num_liv_own = sum(liv_own),
            num_liv_dep = sum(liv_dep),
            num_liv_rent = sum(liv_rent))

driver_clean$mean_safe[is.na(driver_clean$mean_safe)] <- NA
driver_clean$mean_edu[is.na(driver_clean$mean_edu)] <- NA

vehicle_clean <- vehicle %>%
  select(-X1) %>%
  mutate(veh_NA = rowSums(is.na(.))) %>%
  mutate(lease = if_else(ownership_type == 'leased', 1, 0, missing = 0),
         loan = if_else(ownership_type == 'loaned', 1, 0, missing = 0),
         own = if_else(ownership_type == 'owned', 1, 0, missing = 0)) %>%
  mutate(
    color = as.integer(factor(color)),
    make_model = as.integer(factor(make_model))
  ) %>%
  group_by(policy_id) %>%
  summarise(
    act_num_leased_veh = sum(lease),
    act_num_loaned_veh = sum(loan),
    act_num_owned_veh = sum(own),
    mean_car_age = mean(age, na.rm = T),
    max_car_age = max(age, na.rm = T),
    min_car_age = min(age, na.rm = T),
    num_car = n(),
    mean_color = mean(color, na.rm = T),
    mean_model = mean(make_model),
    veh_NA = mean(veh_NA)
  )

vehicle_clean$mean_car_age[is.na(vehicle_clean$mean_car_age)] <- NA
vehicle_clean$max_car_age[is.infinite(vehicle_clean$max_car_age)] <- NA
vehicle_clean$min_car_age[is.infinite(vehicle_clean$min_car_age)] <- NA
vehicle_clean$mean_color[is.na(vehicle_clean$mean_color)] <- NA

merge_policy <- policy_clean %>%
  left_join(driver_clean, by = 'policy_id') %>%
  left_join(vehicle_clean, by = 'policy_id') %>%
  mutate(num_NA = policy_NA + veh_NA + driver_NA) %>%
  replace(., is.na(.), -1) %>%
  select(-policy_NA, -veh_NA, -driver_NA) %>%
  mutate(
    avg_car_amt = quoted_amt/num_car
  )

# merge_policy <- merge_policy %>%
#   mutate(
#     i1 = mean_age - quoted_amt,
#     i2 = mean_edu - year,
#     i3 = CAT_zone - mean_age,
#     i4 = quoted_amt - year,
#     i5 = mean_age - zip,
#     i6 = mean_edu - quoted_amt,
#     i7 = quoted_amt - zip,
#     i8 = num_car - number_drivers,
#     i9 = credit_score - zip,
#     i10 = CAT_zone - quoted_amt,
#     i11 = mean_age - mean_edu,
#     i12 = credit_score - mean_age,
#     i13 = Cov_package_type - mean_age,
#     i14 = credit_score - quoted_amt,
#     i15 = mean_model - zip,
#     i16 = CAT_zone - mean_edu,
#     i17 = Agent_cd - zip,
#     i18 = CAT_zone - zip,
#     i19 = Agent_cd - mean_model,
#     i20 = mean_age - mean_safe,
#     i21 = year - zip,
#     i22 = Agent_cd - credit_score,
#     i23 = mean_age - number_drivers,
#     i24 = Prior_carrier_grp - quoted_amt,
#     i25 = Agent_cd - quoted_amt,
#     i26 = credit_score - mean_model,
#     i27 = mean_model - quoted_amt,
#     i28 = county_name - zip,
#     i29 = mean_safe - zip,
#     i30 = mean_age - mean_model
#   ) %>%
#   mutate(
#     m1 = mean_edu - quoted_amt -year,
#     m2 = CAT_zone - mean_age - quoted_amt,
#     m3 = mean_age - quoted_amt - zip,
#     m4 = mean_age - mean_edu - quoted_amt,
#     m5 = mean_age - mean_edu - year,
#     m6 = CAT_zone - mean_age - zip,
#     m7 = quoted_amt - year - zip,
#     m8 = mean_edu - year - zip,
#     m9 = CAT_zone - mean_age - mean_edu,
#     m10 = CAT_zone - num_liv_dep - zip,
#     m11 = mean_edu - quoted_amt - zip,
#     m12 = credit_score - mean_age - mean_edu,
#     m13 = discount - mean_edu - year,
#     m14 = CAT_zone - mean_edu - quoted_amt,
#     m15 = CAT_zone - mean_age - num_liv_dep,
#     m16 = mean_edu - min_age - year,
#     m17 = discount - quoted_amt - year,
#     m18 = credit_score - mean_age - zip,
#     m19 = mean_age - quoted_amt - year,
#     m20 = num_car - number_drivers - quoted_amt,
#     m21 = CAT_zone - mean_edu - year,
#     m22 = Prior_carrier_grp - mean_age - number_drivers,
#     m23 = number_drivers - quoted_amt - year,
#     m24 = credit_score - mean_edu - year,
#     m25 = mean_edu - number_drivers - year,
#     m26 = discount - mean_age - mean_edu,
#     m27 = Cov_package_type - mean_age - mean_edu,
#     m28 = Agent_cd - county_name - credit_score,
#     m29 = Agent_cd - credit_score - zip,
#     m30 = Agent_cd - mean_edu - year
#   )

train <- merge_policy %>%
  filter(split == 'Train') %>%
  select(-split, -policy_id) %>%
  select(convert_ind, everything())

test <- merge_policy %>%
  filter(split == 'Test') %>%
  select(-split, -convert_ind) %>%
  select(policy_id, everything())

save(train, test,  file = 'tao.RData')
load('tao.RData')

write.csv(train, 'train.csv', row.names = F, na = '')
write.csv(test, 'test.csv', row.names = F, na = '')
