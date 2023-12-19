

fitRf = function(train_predictors, train_outcome, Seed){

  # tuning grid 
  rfGrid =  expand.grid(
    mtry = c(1:dim(train_predictors)[2]),
    splitrule = c("gini", "extratrees"),
    min.node.size = 1
  )
  
  fitcontrol = trainControl(method = "LOOCV",
                            summaryFunction = multiClassSummary)
  set.seed(Seed)
  
  model_fit = train(
    x = train_predictors,
    y = train_outcome,
    method = 'ranger' ,
    trControl = fitcontrol,
    metric = 'Kappa',
    tuneGrid = rfGrid,
    importance = "impurity"
  )
  
  
  
  return(model_fit)
}
