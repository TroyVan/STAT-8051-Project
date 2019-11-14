library(readr)
library(tidyverse)
library(pscl)
library(tweedie)
library(pROC)


# import data -------------------------------------------------------------


policies <- read_csv("policies.csv")
policies$AMT <- as.numeric(gsub('[$,]', '', policies$quoted_amt))
policies <- policies %>%
  mutate(DISC = if_else(discount == 'Yes', 1, 0))

train <- policies %>%
  filter(split == 'Train')

test <- policies  %>% 
  filter(split == 'Test')


# data visualization ------------------------------------------------------


ggplot(train, aes(convert_ind)) +
  geom_bar() +
  facet_grid(. ~ discount)

ggplot(train, aes(convert_ind)) +
  geom_bar() +
  facet_grid(. ~ number_drivers)


ggplot(train, aes(x = factor(convert_ind), y = credit_score)) +
  geom_boxplot()

ggplot(train, aes(x = factor(convert_ind), y = AMT)) +
  geom_boxplot()



# logit model construction ------------------------------------------------


logit_fit <- glm(convert_ind ~ AMT + factor(DISC) + credit_score + factor(num_loaned_veh) + factor(num_owned_veh) + factor(num_leased_veh) + factor(Cov_package_type) + factor(CAT_zone) + factor(primary_parking), data = train, family = 'binomial')
summary(logit_fit)

# predict for train
train_pred <- predict(logit_fit, train, type = 'response')
train$conv_prob <- train_pred
train <- train %>%
  select(policy_id, conv_prob, convert_ind) %>%
  mutate(conv_prob = replace_na(conv_prob, 0.5))

auc(train$convert_ind, train$conv_prob)

# predict for test
pred <- predict(logit_fit, test, type = 'response')

test$conv_prob <- pred
test <- test %>%
  select(policy_id, conv_prob) %>%
  mutate(conv_prob = replace_na(conv_prob, 0.5))

# write file
write.csv(test, 'test.csv', row.names = F)



# tweedie -----------------------------------------------------------------

form <- convert_ind ~ AMT + factor(DISC) + credit_score + factor(num_loaned_veh) + factor(num_owned_veh) + factor(num_leased_veh) + factor(Cov_package_type) + factor(CAT_zone) + factor(primary_parking) + factor(Home_policy_ind)

ini_coef <- glm(form, data = train, family = poisson)$coefficients

tw_fit <- glm(form , data = train, family=tweedie(var.power=1.5,link.power=0), start = ini_coef)
summary(tw_fit)


train_pred <- predict(tw_fit, train, type = 'response')
train$conv_prob <- train_pred
train <- train %>%
  mutate(conv_prob = replace_na(conv_prob, 0))

auc(train$convert_ind, train$conv_prob)

# predict for test
pred <- predict(tw_fit, test, type = 'response')

test$conv_prob <- pred
test <- test %>%
  select(policy_id, conv_prob) %>%
  mutate(conv_prob = replace_na(conv_prob, 0.5))

# write file
write.csv(test, 'test.csv', row.names = F)
