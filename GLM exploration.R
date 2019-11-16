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

# First, start with some "reasonable" predictors

glmcv(convert_ind ~ quoted_amt + discount + Home_policy_ind + Prior_carrier_grp +
        credit_score + Cov_package_type + CAT_zone + number_drivers + total_number_veh +
        primary_parking)
# Cross-Validation AUC = .6337

summary(step(glm(convert_ind ~ quoted_amt + discount + Home_policy_ind + Prior_carrier_grp +
           credit_score + Cov_package_type + CAT_zone + number_drivers + total_number_veh +
           primary_parking, binomial, policies_train), trace = 0))
# Dropped Home_policy_ind, total_number_veh

glmcv(convert_ind ~ quoted_amt + discount + Prior_carrier_grp + 
        credit_score + Cov_package_type + CAT_zone + number_drivers + 
        primary_parking)
# Cross-Validation AUC = .6342

# Now let's add in almost all predictors

glmcv(convert_ind ~ discount + Home_policy_ind + state_id + quoted_amt + Prior_carrier_grp +
        credit_score + Cov_package_type + CAT_zone + number_drivers + num_loaned_veh +
        num_owned_veh + num_leased_veh + total_number_veh + primary_parking)
# Cross-Validation AUC = .6380

summary(step(glm(convert_ind ~ discount + Home_policy_ind + state_id + quoted_amt + Prior_carrier_grp +
                   credit_score + Cov_package_type + CAT_zone + number_drivers + num_loaned_veh +
                   num_owned_veh + num_leased_veh + total_number_veh + primary_parking,
                 binomial, policies_train), trace = 0))
# Dropped Home_policy_ind, num_loaned_veh, num_owned_veh, num_leased_veh, total_number_veh

glmcv(convert_ind ~ discount + state_id + quoted_amt + 
        Prior_carrier_grp + credit_score + Cov_package_type + CAT_zone + 
        number_drivers + primary_parking)
# Cross-Validation AUC = .6388

test_predictions = data.frame(
  policy_id = policies_test$policy_id,
  conv_prob = predict(
    glm(
      convert_ind ~ discount + state_id + quoted_amt +
        Prior_carrier_grp + credit_score + Cov_package_type + CAT_zone +
        number_drivers + primary_parking,
      binomial, policies_train
    ), policies_test, type = "response"
  )
)
# If no predicted probablity, use mean conversion rate as default
test_predictions$conv_prob[is.na(test_predictions$conv_prob)] = mean(policies_train$convert_ind)
write.csv(test_predictions, "Troy's GLM exploration test predictions 2019-11-16.csv")