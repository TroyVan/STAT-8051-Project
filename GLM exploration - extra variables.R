# Create extra variables

for(i in 1:dim(policies_train)[1]){
  policies_train$matching_loaned_vehicles[i] = length(which(policies_train$policy_id[i] == vehicles_train$policy_id & vehicles_train$ownership_type == "loaned"))
  policies_train$matching_owned_vehicles[i] = length(which(policies_train$policy_id[i] == vehicles_train$policy_id & vehicles_train$ownership_type == "owned"))
  policies_train$matching_leased_vehicles[i] = length(which(policies_train$policy_id[i] == vehicles_train$policy_id & vehicles_train$ownership_type == "leased"))
  if(i %% 1000 == 0) print(i)
}
policies_train$total_matching_vehicles = policies_train$matching_loaned_vehicles + policies_train$matching_owned_vehicles + policies_train$matching_leased_vehicles

for(i in 1:dim(policies_train)[1]){
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
  
  policies_train$min_vehicle_age[i] = min(matching_vehicles$age, na.rm = TRUE)
  policies_train$max_vehicle_age[i] = max(matching_vehicles$age, na.rm = TRUE)
  policies_train$avg_vehicle_age[i] = mean(matching_vehicles$age, na.rm = TRUE)
  
  if(i %% 1000 == 0) print(i)
}
policies_train$min_safety_rating[which(is.infinite(policies_train$min_safety_rating))] = NA
policies_train$max_safety_rating[which(is.infinite(policies_train$max_safety_rating))] = NA
policies_train$avg_safety_rating[which(is.nan(policies_train$avg_safety_rating))] = NA

policies_train$prop_high_education[which(is.nan(policies_train$prop_high_education))] = NA

policies_train$min_vehicle_age[which(is.infinite(policies_train$min_vehicle_age))] = NA
policies_train$max_vehicle_age[which(is.infinite(policies_train$max_vehicle_age))] = NA
policies_train$avg_vehicle_age[which(is.nan(policies_train$avg_vehicle_age))] = NA

policies_train$living_status = as.factor(policies_train$living_status)

library(pROC)

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

glmcv(convert_ind ~ discount + Home_policy_ind + state_id + quoted_amt + Prior_carrier_grp +
        credit_score + Cov_package_type + CAT_zone + number_drivers + num_loaned_veh +
        num_owned_veh + num_leased_veh + total_number_veh + primary_parking +
        matching_loaned_vehicles + matching_owned_vehicles + matching_leased_vehicles +
        total_matching_vehicles + prop_male + living_status +
        min_age + max_age + avg_age + min_safety_rating + max_safety_rating + avg_safety_rating +
        prop_high_education + min_vehicle_age + max_vehicle_age + avg_vehicle_age)
# Cross-Validation AUC = .6451, a new record

# Remove missing values to make step() happy
policies_train_noNA = policies_train
policies_train_noNA$Quote_dt = NULL
policies_train_noNA$zip = NULL
policies_train_noNA$county_name = NULL
policies_train_noNA$Agent_cd = NULL
policies_train_noNA = na.omit(policies_train_noNA)

summary(step(glm(convert_ind ~ discount + Home_policy_ind + state_id + quoted_amt + Prior_carrier_grp +
                   credit_score + Cov_package_type + CAT_zone + number_drivers + num_loaned_veh +
                   num_owned_veh + num_leased_veh + total_number_veh + primary_parking +
                   matching_loaned_vehicles + matching_owned_vehicles + matching_leased_vehicles +
                   total_matching_vehicles + prop_male + living_status +
                   min_age + max_age + avg_age + min_safety_rating + max_safety_rating + avg_safety_rating +
                   prop_high_education + min_vehicle_age + max_vehicle_age + avg_vehicle_age,
                 binomial, policies_train_noNA)))
# Dropped Home_policy_ind, num_loaned_veh, num_owned_veh, num_leased_veh, total_number_veh, matching_loaned_vehicles, total_matching_vehicles,
# prop_male, min_safety_rating, max_safety_rating, avg_safety_rating, min_vehicle_age

glmcv(convert_ind ~ discount + state_id + quoted_amt + 
        Prior_carrier_grp + credit_score + Cov_package_type + CAT_zone + 
        number_drivers + primary_parking + matching_owned_vehicles + 
        matching_leased_vehicles + living_status + min_age + 
        max_age + avg_age + prop_high_education + max_vehicle_age + 
        avg_vehicle_age)
# Cross-Validation AUC = .6461, a new record

# Create extra variables

for(i in 1:dim(policies_test)[1]){
  policies_test$matching_loaned_vehicles[i] = length(which(policies_test$policy_id[i] == vehicles_test$policy_id & vehicles_test$ownership_type == "loaned"))
  policies_test$matching_owned_vehicles[i] = length(which(policies_test$policy_id[i] == vehicles_test$policy_id & vehicles_test$ownership_type == "owned"))
  policies_test$matching_leased_vehicles[i] = length(which(policies_test$policy_id[i] == vehicles_test$policy_id & vehicles_test$ownership_type == "leased"))
  if(i %% 1000 == 0) print(i)
}
policies_test$total_matching_vehicles = policies_test$matching_loaned_vehicles + policies_test$matching_owned_vehicles + policies_test$matching_leased_vehicles

for(i in 1:dim(policies_test)[1]){
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
  
  policies_test$min_vehicle_age[i] = min(matching_vehicles$age, na.rm = TRUE)
  policies_test$max_vehicle_age[i] = max(matching_vehicles$age, na.rm = TRUE)
  policies_test$avg_vehicle_age[i] = mean(matching_vehicles$age, na.rm = TRUE)
  
  if(i %% 1000 == 0) print(i)
}
policies_test$min_safety_rating[which(is.infinite(policies_test$min_safety_rating))] = NA
policies_test$max_safety_rating[which(is.infinite(policies_test$max_safety_rating))] = NA
policies_test$avg_safety_rating[which(is.nan(policies_test$avg_safety_rating))] = NA

policies_test$prop_high_education[which(is.nan(policies_test$prop_high_education))] = NA

policies_test$min_vehicle_age[which(is.infinite(policies_test$min_vehicle_age))] = NA
policies_test$max_vehicle_age[which(is.infinite(policies_test$max_vehicle_age))] = NA
policies_test$avg_vehicle_age[which(is.nan(policies_test$avg_vehicle_age))] = NA

policies_test$living_status = as.factor(policies_test$living_status)

test_predictions = data.frame(
  policy_id = policies_test$policy_id,
  conv_prob = predict(
    glm(
      convert_ind ~ discount + state_id + quoted_amt + 
        Prior_carrier_grp + credit_score + Cov_package_type + CAT_zone + 
        number_drivers + primary_parking + matching_owned_vehicles + 
        matching_leased_vehicles + living_status + min_age + 
        max_age + avg_age + prop_high_education + max_vehicle_age + 
        avg_vehicle_age,
      binomial, policies_train
    ), policies_test, type = "response"
  )
)
# If no predicted probablity, use mean conversion rate as default
test_predictions$conv_prob[is.na(test_predictions$conv_prob)] = mean(policies_train$convert_ind)
write.csv(test_predictions, "Troy's GLM exploration test predictions 2019-11-21.csv", row.names = FALSE)
# Test AUC = .63817, best so far