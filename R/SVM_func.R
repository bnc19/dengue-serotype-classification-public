

fit_svm_poly = function(train_predictors,
                        train_outcome,
                        Seed){

  PsvmGrid = expand.grid(
    degree = c(2:4),
    scale = c(0.001, 0.005, 0.01),
    C = c(2 ^ (-2:9))
  )
  
  fitcontrol = trainControl(method = "LOOCV",
                            summaryFunction = multiClassSummary)
  
  set.seed(Seed)
  model_fit = train(
    x = train_predictors,
    y = train_outcome,
    method = 'svmPoly',
    metric = "Kappa",
    preProc = c("center", "scale"),
    tuneGrid = PsvmGrid,
    trControl = fitcontrol
  )
  

  
  return(model_fit)
}

