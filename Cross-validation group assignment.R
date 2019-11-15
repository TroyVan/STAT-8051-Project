# Goal: Assign each quote into one of 10 groups of equal size at random
# Converted and non-converted quotes have separate drawings
# Start with "2019-11-10 edition datasets.RData" loaded in

policies_train$rownum = 1:nrow(policies_train)
converted_policies = data.frame(
  policy_id = policies_train$policy_id[policies_train$convert_ind == TRUE],
  rownum = policies_train$rownum[policies_train$convert_ind == TRUE]
)
non_converted_policies = data.frame(
  policy_id = policies_train$policy_id[policies_train$convert_ind == FALSE],
  rownum = policies_train$rownum[policies_train$convert_ind == FALSE]
)
policies_train$rownum = NULL

set.seed(45129) # Picked from [0,99999] using random.org
converted_policies = converted_policies[sample(nrow(converted_policies)),]
non_converted_policies = non_converted_policies[sample(nrow(non_converted_policies)),]

converted_policies$group = rep(1:10, length.out = nrow(converted_policies))
non_converted_policies$group = rep(1:10, length.out = nrow(non_converted_policies))

policy_group_assignments = rbind(converted_policies, non_converted_policies)
policy_group_assignments = policy_group_assignments[order(policy_group_assignments$rownum),]
policy_group_assignments$rownum = NULL
policies_train$group = policy_group_assignments$group

write.csv(policy_group_assignments, "Cross-Validation Group Assignments.csv", row.names = FALSE)

remove(converted_policies)
remove(non_converted_policies)