# create partitions of test and train data 

format_train_test_data = function(model_predictors,
                                  model_outcome,
                                  split_prop) {

  # split data
  partition = createDataPartition(model_outcome, p = split_prop)
  
  # extract training and test predictors 
  train_predictors =  model_predictors[partition$Resample1, ]
  test_predictors =   model_predictors[-partition$Resample1,]
  
  # extract training and test outcome  
  train_outcome = model_outcome[partition$Resample1]
  test_outcome =  model_outcome[-partition$Resample1]
  
  out = list(train_predictors, test_predictors, train_outcome, test_outcome)
  names(out) = c("train_predictors", "test_predictors", "train_outcome", "test_outcome")
  return(out)
}

# run classifier models using train and test data 

fit_model = function(classifier_name,
                     train_predictors,
                     train_outcome,
                     test_predictors,
                     test_outcome,
                     Seed
)
{

# This function fits one classifier on the current training data
# hyperparameters are tuned through LOOCV
  
  
  source('R/RF_func.R')
  source('R/GBM_func.R')
  source('R/NN_func.R')
  source('R/SVM_func.R')
  source('R/MLR_func.R')

  
  # given the classifier name, fit the correct model
  model_fit = switch(
    classifier_name,
    rf = fitRf(train_predictors, train_outcome, Seed),
    gbm = fitgbm (train_predictors, train_outcome, Seed),
    nnet = fitNnet(train_predictors, train_outcome, Seed),
    svm_poly = fit_svm_poly(train_predictors, train_outcome, Seed),
    mlr = fitlmr(train_predictors, train_outcome, Seed)
  )
  
  # retrieve the highest performance values achieved by a hyper-parameter search
  trainCM = get_CM(model_fit, classifier_name, train_outcome)
  
  # predict on test data
  test_out = predict(model_fit, test_predictors)
  testCM = confusionMatrix(test_out, test_outcome)
  
  testing_df = 
    data.frame(
      test_predictors,
      test_outcome = test_outcome,
      test_predicted = test_out
    )
  
  
  out = list(model_fit, trainCM, testCM, testing_df)
  names(out) = c("model_fit", "trainCM", "testCM", "testing_df")
  return(out)
}



# calculate weighted average from class specific metrics 

weighted_av = function(class_data, train_CM_tab, test_CM_tab = NULL){
# number in each class- weights 

train_N = train_CM_tab %>%
  as.data.frame() %>%
  group_by(Reference) %>%
  summarise(N = sum(Freq)) %>%
  mutate(serotype = ifelse(Reference == "four" | Reference == 4 , 4,
                           ifelse(
                             Reference == "three"| Reference == 3 , 3,
                             ifelse(
                               Reference == "two"| Reference == 2, 2,
                                    ifelse(
                                      Reference == "one"| Reference == 1, 1,NA)
                           )))) %>%
  arrange(serotype)

if(is.null(test_CM_tab)){
  out  = class_data %>%  
    mutate(serotype = ifelse(class == "four", 4,
                             ifelse(
                               class == "three", 3,
                               ifelse(class == "two", 2, 1)
                             ))) %>% 
    arrange(split,serotype) %>%  
    mutate(N = train_N$N) %>% 
    dplyr::summarise(across(Sensitivity:'Balanced Accuracy',~ 
                              weighted.mean(., w = N , na.rm = T))) %>% 
    mutate(split = "train")
} else{
  
  
  test_N = test_CM_tab %>%
    as.data.frame() %>%
    group_by(Reference) %>%
    summarise(N = sum(Freq)) %>%
    mutate(serotype = ifelse(Reference == "four", 4,
                             ifelse(
                               Reference == "three", 3,
                               ifelse(Reference == "two", 2, 1)
                             ))) %>%
    arrange(serotype)
  
  
  out  = class_data %>%  
    mutate(serotype = ifelse(class == "four", 4,
                             ifelse(
                               class == "three", 3,
                               ifelse(class == "two", 2, 1)
                             ))) %>% 
    arrange(split,serotype) %>%  
    mutate(N = c(test_N$N, train_N$N)) %>%  
    group_by(split) %>%  
    dplyr::summarise(across(Sensitivity:'Balanced Accuracy',~ 
                              weighted.mean(., w = N , na.rm = T))) 
}

 
return(out)
}


# function to calculate confusion matrix 
get_CM = function (model_fit,
                   classifier_name,
                   train_outcome) {
  library(dplyr)
  library(caret)
  
  # calculate PPV and NPV from model predictions and observations
  getPred = function(model_fit,
                     classifier_name,
                     train_outcome) {
    model = model_fit
    
    # RF pred and obs
    
    rfResults = function(model) {
      Pred = model$finalModel$predictions
      Obs = model$trainingData$.outcome
      
      BT = model$bestTune
      row = rownames(BT)
      best = as.data.frame(model$results[row, ])
      
      out = data.frame(Pred, Obs)
      return(list(out, best))
    }
    
    
    # gbm pred and obs
    
    gbmResults =  function(model) {
      BT = model$bestTune
      allPred = model$pred
      bestmodel = filter(
        allPred,
        shrinkage == BT$shrinkage &
          interaction.depth == BT$interaction.depth &
          n.minobsinnode == BT$n.minobsinnode &
          n.trees == BT$n.trees
      )
      
      Pred = (bestmodel$pred)
      Obs = (bestmodel$obs)
      out = data.frame(Pred, Obs)
      
      row = rownames(BT)
      best = as.data.frame(model$results[row, ])
      return(list(out, best))
    }
    
    
    # nn pred and obs
    
    nnResults =  function(model) {
      BT = model$bestTune
      allPred = model$pred
      bestmodel = filter(allPred,
                         size == BT$size &
                           decay == BT$decay &
                           bag == BT$bag)
      
      Pred = (bestmodel$pred)
      Obs = (bestmodel$obs)
      out = data.frame(Pred, Obs)
      
      row = rownames(BT)
      best = as.data.frame(model$results[row, ])
      return(list(out, best))
    }
    
    
    # SVM poly pred and obs
    
    PsvmResults =  function(model) {
      BT = model$bestTune
      allPred = model$pred
      bestmodel = filter(allPred,
                         degree == BT$degree &
                           scale == BT$scale &
                           C == BT$C)
      
      Pred = (bestmodel$pred)
      Obs = (bestmodel$obs)
      out = data.frame(Pred, Obs)
      
      row = rownames(BT)
      best = as.data.frame(model$results[row, ])
      return(list(out, best))
    }
    
    # mlr pred and obs
    
    mlrResults = function(model) {
      BT = model$bestTune
      allPred = model$pred
      bestmodel = filter(allPred,
                         decay == BT$decay)
      
      
      Pred = (bestmodel$pred)
      Obs = (bestmodel$obs)
      out = data.frame(Pred, Obs)
      
      BT = model$bestTune
      row = rownames(BT)
      best = as.data.frame(model$results[row, ])
      
      return(list(out, best))
    }
    
    # use classifier_name to obtain predicted and observed
    
    PredOb = switch(
      classifier_name,
      rf = rfResults(model_fit),
      gbm = gbmResults (model_fit),
      nnet = nnResults(model_fit),
      svm_poly = PsvmResults(model_fit),
      svm_rad = RsvmResults(model_fit),
      xgboost = xgbResults(model_fit),
      mlr = mlrResults(model_fit)
    )
    
    
    CM =  confusionMatrix(PredOb[[1]]$Pred, PredOb[[1]]$Obs)
    return(CM)
    
  }
  
  
  # create data frame of all results
  CM = getPred(model_fit, classifier_name, train_outcome)
  
  return(CM)
}


# overall function to run model fitting (on the cluster)
 # splits data 
 # runs the classifier
 # extracts training and test performance 
 # returns a data frame of performance metrics 

run_model = function(classifier_name,
                     model_predictors,
                     model_outcome,
                     split_prop,
                     Seed) {

  
  data = format_train_test_data(model_predictors, model_outcome, split_prop) 
  
  
  train_predictors = data$train_predictors
  train_outcome = data$train_outcome
  test_predictors = data$test_predictors
  test_outcome = data$test_outcome
  
  
  results = fit_model(
    classifier_name = classifier_name,
    train_predictors = train_predictors,
    train_outcome = train_outcome,
    test_predictors = test_predictors,
    test_outcome = test_outcome,
    Seed = Seed
  )
  
  
# extract class specific metrics (e.g. sens, spec)
  train_CM = results$trainCM
  test_CM = results$testCM
  
  train_class_data = train_CM$byClass %>% 
    as.data.frame() %>%  
    mutate(split = "train")
  
  class_data = test_CM$byClass %>%  
    as.data.frame() %>% 
  mutate(split = "test") %>%  
    bind_rows(train_class_data) %>% 
  rownames_to_column(var = "class") %>%  
    separate(class, into = c(NA, "class"), sep = " ") %>% 
    separate(class, into = c("class", NA)) %>% 
    remove_rownames() 


# calculate weighted average from class specific data 
  train_CM_tab = train_CM$table
  test_CM_tab = test_CM$table
  
   summary_class_data =  weighted_av(class_data,train_CM_tab,test_CM_tab )
  
 # extract overall accuracy and kappa 
  
  train_performance = train_CM$overall %>% 
  as.data.frame() %>%  
  mutate(split = "train") %>%  
  rownames_to_column(var = "metric") %>%  
  filter(metric == "Accuracy" | metric == "Kappa")
  
  performance_data = test_CM$overall %>% 
    as.data.frame() %>%  
    mutate(split = "test") %>%  
    rownames_to_column(var = "metric") %>%  
    filter(metric == "Accuracy" | metric == "Kappa") %>%  
    bind_rows(train_performance) %>% 
    pivot_wider(id_cols = split, names_from = metric, values_from = '.') %>% 
    left_join(summary_class_data) %>% 
    pivot_longer(cols = -split, names_to = "metric")
 
  # return model fit 
  model_fit =  results$model_fit
  
  out = list(performance_data, model_fit)
  
  return(out)
}




