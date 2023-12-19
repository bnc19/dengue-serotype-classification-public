

fitNnet = function(train_predictors,train_outcome, Seed)
{

  nnetGrid = expand.grid(
    decay = c(0, 0.1, 0.25, 0.5),
    size = c(3:15),
    # hidden units to the output
    # The next option is to use bagging instead of different random seeds
    # same as bagging trees
    bag = FALSE
  )
  
  
  fitcontrol = trainControl(method = "LOOCV",
                            summaryFunction = multiClassSummary)
  
  set.seed(Seed) 
  
  model_fit = train(
    x = train_predictors,
    y = train_outcome,
    method = "avNNet",
    repeats = 5,
    # average over 5 models
    metric = "Kappa",
    preProc = c("center", "scale"),
    # "spatialSign" transformation to consider for outliers
    tuneGrid = nnetGrid,
    trControl = fitcontrol
  )
  
  
  return(model_fit)
}

