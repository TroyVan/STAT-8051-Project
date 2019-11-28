# Goal: Produce actual vehicle counts
# Load in "2019-11-10 edition datasets with cross-validation group assignments.RData"

for(i in 1:dim(policies_train)[1]){
  policies_train$matching_loaned_vehicles[i] = length(which(policies_train$policy_id[i] == vehicles_train$policy_id & vehicles_train$ownership_type == "loaned"))
  policies_train$matching_owned_vehicles[i] = length(which(policies_train$policy_id[i] == vehicles_train$policy_id & vehicles_train$ownership_type == "owned"))
  policies_train$matching_leased_vehicles[i] = length(which(policies_train$policy_id[i] == vehicles_train$policy_id & vehicles_train$ownership_type == "leased"))
  if(i %% 1000 == 0) print(i)
}
policies_train$total_matching_vehicles = policies_train$matching_loaned_vehicles + policies_train$matching_owned_vehicles + policies_train$matching_leased_vehicles

policies_train$num_loaned_veh = NULL
policies_train$num_owned_veh = NULL
policies_train$num_leased_veh = NULL
policies_train$total_number_veh = NULL

for(i in 1:dim(policies_test)[1]){
  policies_test$matching_loaned_vehicles[i] = length(which(policies_test$policy_id[i] == vehicles_test$policy_id & vehicles_test$ownership_type == "loaned"))
  policies_test$matching_owned_vehicles[i] = length(which(policies_test$policy_id[i] == vehicles_test$policy_id & vehicles_test$ownership_type == "owned"))
  policies_test$matching_leased_vehicles[i] = length(which(policies_test$policy_id[i] == vehicles_test$policy_id & vehicles_test$ownership_type == "leased"))
  if(i %% 1000 == 0) print(i)
}
policies_test$total_matching_vehicles = policies_test$matching_loaned_vehicles + policies_test$matching_owned_vehicles + policies_test$matching_leased_vehicles

policies_test$num_loaned_veh = NULL
policies_test$num_owned_veh = NULL
policies_test$num_leased_veh = NULL
policies_test$total_number_veh = NULL

remove(i)
# Resulting dataset saved as "2019-11-28 edition datasets.RData"