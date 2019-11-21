# Previous process gave observations with missing predictor values the average conversion rate, ignoring the predictors that have values.
# Can we do better?
# Here we assign missing predictors the mean (or mode if categorical) across the training set.
# Only test data get assigned values; training data is unchanged

# Load in "2019-11-10 edition datasets with cross-validation group assignments.RData" first

mode = function(x){
  tab = table(x)
  maxcount = max(tab)
  modes = names(tab)[tab == maxcount]
  return(modes[1]) #If multiple modes, default to first
}

policies_test_noNA = policies_test

policies_test_noNA$quoted_amt[is.na(policies_test_noNA$quoted_amt)] = mean(policies_train$quoted_amt, na.rm = TRUE)
policies_test_noNA$Prior_carrier_grp[is.na(policies_test_noNA$Prior_carrier_grp)] = mode(policies_train$Prior_carrier_grp)
policies_test_noNA$Cov_package_type[is.na(policies_test_noNA$Cov_package_type)] = mode(policies_train$Cov_package_type)
policies_test_noNA$credit_score[is.na(policies_test_noNA$credit_score)] = mean(policies_train$credit_score, na.rm = TRUE)
policies_test_noNA$CAT_zone[is.na(policies_test_noNA$CAT_zone)] = mode(policies_train$CAT_zone)

# Drop unused predictors
policies_test_noNA$zip = NULL
policies_test_noNA$county_name = NULL
policies_test_noNA$Agent_cd = NULL

# Verify that no NA values remain
sum(is.na(policies_test_noNA))

# Since training data is unchanged, model remains unchanged

test_predictions = data.frame(
  policy_id = policies_test_noNA$policy_id,
  conv_prob = predict(
    glm(
      convert_ind ~ discount + state_id + quoted_amt +
        Prior_carrier_grp + credit_score + Cov_package_type + CAT_zone +
        number_drivers + primary_parking,
      binomial, policies_train
    ), policies_test_noNA, type = "response"
  )
)

# Verify that no NA predictions are present
sum(is.na(test_predictions$conv_prob))

write.csv(test_predictions, "Troy's GLM exploration test predictions 2019-11-20.csv", row.names = FALSE)

# Test AUC = .61661

# OK, so that's a waste of time. What if we assign default values to the training set as well?

policies_train_noNA = policies_train

policies_train_noNA$quoted_amt[is.na(policies_train_noNA$quoted_amt)] = mean(policies_train$quoted_amt, na.rm = TRUE)
policies_train_noNA$Prior_carrier_grp[is.na(policies_train_noNA$Prior_carrier_grp)] = mode(policies_train$Prior_carrier_grp)
policies_train_noNA$Cov_package_type[is.na(policies_train_noNA$Cov_package_type)] = mode(policies_train$Cov_package_type)
policies_train_noNA$credit_score[is.na(policies_train_noNA$credit_score)] = mean(policies_train$credit_score, na.rm = TRUE)
policies_train_noNA$CAT_zone[is.na(policies_train_noNA$CAT_zone)] = mode(policies_train$CAT_zone)

# Drop unused predictors
policies_train_noNA$zip = NULL
policies_train_noNA$county_name = NULL
policies_train_noNA$Agent_cd = NULL

# Verify that no NA values remain
sum(is.na(policies_train_noNA))

# Now redo cross-validation

library(pROC)

# ---How to do cross-validation---
# The quotes have been randomly sorted into 10 groups of equal size. For each group i:
# - Fit a GLM with observations NOT in group i
# - Use that model to predict the probabilities of conversion for observations in group i
# Then, compute the AUC with the generated probabilities of conversion
glmcv = function(formula) {
  for(i in 1:10){
    fitting_data = policies_train_noNA[policies_train_noNA$group != i,]
    mod = glm(formula, binomial, fitting_data)
    policies_train_noNA$pred_prob[policies_train_noNA$group == i] = predict(mod, policies_train_noNA[policies_train_noNA$group == i,], type = "response")
  }
  
  auc(response = policies_train_noNA$convert_ind, predictor = policies_train_noNA$pred_prob)
}

glmcv(convert_ind ~ discount + Home_policy_ind + state_id + quoted_amt + Prior_carrier_grp +
        credit_score + Cov_package_type + CAT_zone + number_drivers + num_loaned_veh +
        num_owned_veh + num_leased_veh + total_number_veh + primary_parking)
# Cross-Validation AUC = .6356, lower than before

summary(step(glm(convert_ind ~ discount + Home_policy_ind + state_id + quoted_amt + Prior_carrier_grp +
                   credit_score + Cov_package_type + CAT_zone + number_drivers + num_loaned_veh +
                   num_owned_veh + num_leased_veh + total_number_veh + primary_parking,
                 binomial, policies_train_noNA), trace = 0))
# Dropped Home_policy_ind, num_loaned_veh, num_owned_veh, num_leased_veh, total_number_veh; no change

glmcv(convert_ind ~ discount + state_id + quoted_amt + 
        Prior_carrier_grp + credit_score + Cov_package_type + CAT_zone + 
        number_drivers + primary_parking)
# Cross-Validation AUC = .6364, lower than before

# A bigger waste of time; I won't even bother testing the test set