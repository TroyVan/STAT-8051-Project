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