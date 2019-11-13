library(readr)
library(tidyverse)
library(pscl)
library(tweedie)
library(pROC)
policies <- read_csv("policies.csv")
policies$AMT <- as.numeric(gsub('[$,]', '', policies$quoted_amt))
policies <- policies %>%
  mutate(DISC = if_else(discount == 'Yes', 1, 0))

train <- policies %>%
  filter(split == 'Train')

test <- policies  %>% 
  filter(split == 'Test')

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

logit_fit <- glm(convert_ind ~ AMT + DISC + credit_score + factor(num_loaned_veh) + factor(num_owned_veh) + factor(num_leased_veh), data = train, family = 'binomial')
summary(logit_fit)
train_pred <- predict(logit_fit, train, type = 'response')
train$conv_prob <- train_pred
train <- train %>%
  select(policy_id, conv_prob, convert_ind) %>%
  mutate(conv_prob = replace_na(conv_prob, 0))

auc(train$convert_ind, train$conv_prob)

pred <- predict(logit_fit, test, type = 'response')

test$conv_prob <- pred
test <- test %>%
  select(policy_id, conv_prob) %>%
  mutate(conv_prob = replace_na(conv_prob, 0))

write.csv(test, 'test.csv', row.names = F)
