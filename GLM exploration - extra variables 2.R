library(pROC)
library(tidyverse)

# Now with vehicle make and color information

# Load in "2019-11-28 edition datasets.RData"

# ---How to do cross-validation---
# The quotes have been randomly sorted into 10 groups of equal size. For each group i:
# - Fit a GLM with observations NOT in group i
# - Use that model to predict the probabilities of conversion for observations in group i
# Then, compute the AUC with the generated probabilities of conversion
glmcv = function(formula) {
  for(i in 1:10){
    fitting_data = policies_train[policies_train$group != i,]
    mod = glm(formula, binomial, fitting_data)
    policies_train$pred_prob[policies_train$group == i] = predict(mod, policies_train[policies_train$group == i,], type = "response")
  }
  
  # If no predicted probablity, use mean conversion rate as default
  policies_train$pred_prob[is.na(policies_train$pred_prob)] = mean(policies_train$convert_ind)
  
  auc(response = policies_train$convert_ind, predictor = policies_train$pred_prob)
}

mode = function(x){
  tab = table(x)
  maxcount = max(tab)
  if(maxcount == -Inf) return(NA) # No non-NA value
  modes = names(tab)[tab == maxcount]
  if(length(modes) > 1) return("a_multimodal")
  else return(modes[1])
}

# Create extra variables

for(i in 1:nrow(policies_train)){
  matching_drivers = drivers_train[which(policies_train$policy_id[i] == drivers_train$policy_id),]
  
  policies_train$prop_male[i] = mean(matching_drivers$gender == "M")
  
  if(sum(matching_drivers$living_status == "own", na.rm = TRUE) > 0) policies_train$living_status[i] = "own"
  else if(sum(matching_drivers$living_status == "rent", na.rm = TRUE) > 0) policies_train$living_status[i] = "rent"
  else if(sum(matching_drivers$living_status == "dependent", na.rm = TRUE) > 0) policies_train$living_status[i] = "dependent"
  else policies_train$living_status[i] = NA
  
  policies_train$min_age[i] = min(matching_drivers$age)
  policies_train$max_age[i] = max(matching_drivers$age)
  policies_train$avg_age[i] = mean(matching_drivers$age)
  
  policies_train$min_safety_rating[i] = min(matching_drivers$safty_rating, na.rm = TRUE)
  policies_train$max_safety_rating[i] = max(matching_drivers$safty_rating, na.rm = TRUE)
  policies_train$avg_safety_rating[i] = mean(matching_drivers$safty_rating, na.rm = TRUE)
  
  policies_train$prop_high_education[i] = mean(matching_drivers$high_education_ind, na.rm = TRUE)
  
  matching_vehicles = vehicles_train[which(policies_train$policy_id[i] == vehicles_train$policy_id),]
  matching_vehicles$make = sub(" : .*", "", matching_vehicles$make_model)
  
  policies_train$min_vehicle_age[i] = min(matching_vehicles$age, na.rm = TRUE)
  policies_train$max_vehicle_age[i] = max(matching_vehicles$age, na.rm = TRUE)
  policies_train$avg_vehicle_age[i] = mean(matching_vehicles$age, na.rm = TRUE)
  
  policies_train$mode_color[i] = mode(matching_vehicles$color)
  policies_train$mode_make[i] = mode(matching_vehicles$make)
  
  if(i %% 1000 == 0) print(i)
}

remove(i)
remove(matching_drivers)
remove(matching_vehicles)

policies_train$year = as.factor(lubridate::year(policies_train$Quote_dt))
policies_train$month = as.factor(lubridate::month(policies_train$Quote_dt))
policies_train$wday = as.factor(lubridate::wday(policies_train$Quote_dt))

policies_train$min_safety_rating[which(is.infinite(policies_train$min_safety_rating))] = NA
policies_train$max_safety_rating[which(is.infinite(policies_train$max_safety_rating))] = NA
policies_train$avg_safety_rating[which(is.nan(policies_train$avg_safety_rating))] = NA

policies_train$prop_high_education[which(is.nan(policies_train$prop_high_education))] = NA

policies_train$min_vehicle_age[which(is.infinite(policies_train$min_vehicle_age))] = NA
policies_train$max_vehicle_age[which(is.infinite(policies_train$max_vehicle_age))] = NA
policies_train$avg_vehicle_age[which(is.nan(policies_train$avg_vehicle_age))] = NA

policies_train$living_status = as.factor(policies_train$living_status)
policies_train$mode_color = as.factor(policies_train$mode_color)
policies_train$mode_make = as.factor(policies_train$mode_make)

# Strike factor variables with many levels and remove missing values to make step() happy
policies_train_noNA = policies_train
policies_train_noNA$Quote_dt = NULL
policies_train_noNA$zip = NULL
policies_train_noNA$county_name = NULL
policies_train_noNA$Agent_cd = NULL
policies_train_noNA$policy_id = NULL
policies_train_noNA = na.omit(policies_train_noNA)

summary(step(glm(convert_ind ~ ., binomial, policies_train_noNA)))
# Resulting formula: convert_ind ~ discount + state_id + quoted_amt + 
# Prior_carrier_grp + credit_score + Cov_package_type + CAT_zone + 
# number_drivers + primary_parking + matching_leased_vehicles + 
# living_status + min_age + max_age + avg_age + prop_high_education + 
# max_vehicle_age + avg_vehicle_age + mode_make + year

glmcv(convert_ind ~ discount + state_id + quoted_amt + 
        Prior_carrier_grp + credit_score + Cov_package_type + CAT_zone + 
        number_drivers + primary_parking + matching_leased_vehicles + 
        living_status + min_age + max_age + avg_age + prop_high_education + 
        max_vehicle_age + avg_vehicle_age + mode_make + year)
# Cross-Validation AUC = .6463

mod = glm(convert_ind ~ discount + state_id + quoted_amt + 
            Prior_carrier_grp + credit_score + Cov_package_type + CAT_zone + 
            number_drivers + primary_parking + matching_leased_vehicles + 
            living_status + min_age + max_age + avg_age + prop_high_education + 
            max_vehicle_age + avg_vehicle_age + mode_make + year,
          binomial, policies_train)

# Create extra variables

for(i in 1:nrow(policies_test)){
  matching_drivers = drivers_test[which(policies_test$policy_id[i] == drivers_test$policy_id),]
  
  policies_test$prop_male[i] = mean(matching_drivers$gender == "M")
  
  if(sum(matching_drivers$living_status == "own", na.rm = TRUE) > 0) policies_test$living_status[i] = "own"
  else if(sum(matching_drivers$living_status == "rent", na.rm = TRUE) > 0) policies_test$living_status[i] = "rent"
  else if(sum(matching_drivers$living_status == "dependent", na.rm = TRUE) > 0) policies_test$living_status[i] = "dependent"
  else policies_test$living_status[i] = NA
  
  policies_test$min_age[i] = min(matching_drivers$age)
  policies_test$max_age[i] = max(matching_drivers$age)
  policies_test$avg_age[i] = mean(matching_drivers$age)
  
  policies_test$min_safety_rating[i] = min(matching_drivers$safty_rating, na.rm = TRUE)
  policies_test$max_safety_rating[i] = max(matching_drivers$safty_rating, na.rm = TRUE)
  policies_test$avg_safety_rating[i] = mean(matching_drivers$safty_rating, na.rm = TRUE)
  
  policies_test$prop_high_education[i] = mean(matching_drivers$high_education_ind, na.rm = TRUE)
  
  matching_vehicles = vehicles_test[which(policies_test$policy_id[i] == vehicles_test$policy_id),]
  matching_vehicles$make = sub(" : .*", "", matching_vehicles$make_model)
  
  policies_test$min_vehicle_age[i] = min(matching_vehicles$age, na.rm = TRUE)
  policies_test$max_vehicle_age[i] = max(matching_vehicles$age, na.rm = TRUE)
  policies_test$avg_vehicle_age[i] = mean(matching_vehicles$age, na.rm = TRUE)
  
  policies_test$mode_color[i] = mode(matching_vehicles$color)
  policies_test$mode_make[i] = mode(matching_vehicles$make)
  
  if(i %% 1000 == 0) print(i)
}

remove(i)
remove(matching_drivers)
remove(matching_vehicles)

policies_test$year = as.factor(lubridate::year(policies_test$Quote_dt))
policies_test$month = as.factor(lubridate::month(policies_test$Quote_dt))
policies_test$wday = as.factor(lubridate::wday(policies_test$Quote_dt))

policies_test$min_safety_rating[which(is.infinite(policies_test$min_safety_rating))] = NA
policies_test$max_safety_rating[which(is.infinite(policies_test$max_safety_rating))] = NA
policies_test$avg_safety_rating[which(is.nan(policies_test$avg_safety_rating))] = NA

policies_test$prop_high_education[which(is.nan(policies_test$prop_high_education))] = NA

policies_test$min_vehicle_age[which(is.infinite(policies_test$min_vehicle_age))] = NA
policies_test$max_vehicle_age[which(is.infinite(policies_test$max_vehicle_age))] = NA
policies_test$avg_vehicle_age[which(is.nan(policies_test$avg_vehicle_age))] = NA

policies_test$living_status = as.factor(policies_test$living_status)
policies_test$mode_color = as.factor(policies_test$mode_color)
policies_test$mode_make = as.factor(policies_test$mode_make)

test_predictions = data.frame(
  policy_id = policies_test$policy_id,
  conv_prob = predict(mod, policies_test, type = "response")
)
# If no predicted probablity, use mean conversion rate as default
test_predictions$conv_prob[is.na(test_predictions$conv_prob)] = mean(policies_train$convert_ind)
write.csv(test_predictions, "Troy's GLM exploration test predictions 2019-12-03.csv", row.names = FALSE)
# Test score = .63819

# Resulting database saved as "GLM with more extra variables 2019-12-03.RData"