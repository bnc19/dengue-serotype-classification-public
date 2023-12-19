# Example script to fit machine learning classifiers and multinomial logistic 
# regression to PRNT data, in order to predict the infecting dengue serotype. 

# Data is provided to fit the models using all titre data (scenario A) and 
# using all titre data plus year of infection and age-group (modified scenario 
# B). School and age in years are not included in this data set to ensure 
# anonymity. 

# Code is provided to run the GBM model on both data sets, using the 
# run_model function. However, any of the machine learning classifiers can be 
# implemented by changing the classifier_name argument: 

# - rf: random forrest 
# - gbm: gradient boosted machine
# - nnet: neutral network
# - svm_poly: support vector machine
# - mlr: multinomial logistic regression 

# The other function arguments:

# - model_predictors: data frame of model predictors 
# - model_outcome: vector of outcomes 
# - split_prop: proportion of data used to train the model 
# - Seed: set seed 

# In the paper, each model was fit 100 times, using 100 bootstrap samples of the 
# test (10%) and train (90%) sets, in order to calculate mean and 95% confidence 
# intervals. Here, the code is set up to run each model just once using a 50/50  
# test and train split, which takes ~5 minutes per model, on an 8 core laptop. 

# The function run_model returns a list containing a data frame of the 
# performance metrics and the model fits. 

# preamble ---------------------------------------------------------------------
# install.packages("tidyverse")
# install.packages("caret")
# install.packages("nnet")
# install.packages("ranger")
# install.packages("kernlab")
# install.packages("gbm")


# setwd("to_share")

library("tidyverse")
library("caret")
library("nnet")
library("ranger")
library("kernlab")
library("gbm")

source("R/model_fitting_func.R")

# import all data --------------------------------------------------------------

# outcome data 
all_outcome = as.factor(read.csv("data/all_outcome.csv")$x)

# predictors data 
titre_predictors = read.csv("data/titre_predictors.csv")[,-1]
all_predictors = read.csv("data/all_predictors.csv")[,-1]
all_predictors$age_class = as.factor(all_predictors$age_class) 

# run gbm with titre predictors ------------------------------------------------

M1 = run_model(
  classifier_name = "gbm",
  model_predictors = titre_predictors,
  model_outcome = all_outcome,
  split_prop = 0.9,
  Seed = 1
)

# run gbm with all predictors --------------------------------------------------

M2 = run_model(
  classifier_name = "gbm",
  model_predictors = all_predictors,
  model_outcome = all_outcome,
  split_prop = 0.5,
  Seed = 1
)

# plot performance -------------------------------------------------------------

plot_performance = M1[[1]] %>%  
  mutate(scenario = "Scenario A") %>% 
  bind_rows(mutate(M2[[1]], scenario = "Scenario B")) %>% 
  filter(metric %in% c("Accuracy", "Kappa", "Sensitivity", "Specificity")) %>% 
  ggplot(aes(
    x = metric,
    y = value)) +
  geom_bar(
    stat = "identity",
    aes(fill = split),
    position = position_dodge(width = 1)) +
  facet_wrap(~scenario) + 
  scale_y_continuous(limits = c(-0.1, 1), breaks = seq(0, 1, 0.2)) +
  theme_bw(base_size = 25) 

plot_performance

