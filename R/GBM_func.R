
fitgbm = function(train_predictors,
                  train_outcome,
                  Seed){

  
  # packages
  if (length(train_outcome) < 100) {
    gbmGrid = expand.grid(
      interaction.depth = seq(1, 7, by = 2),
      n.trees = seq(200, 1000, by = 200),
      shrinkage = c(0.001, 0.01, 0.1),
      n.minobsinnode = c(1)
    )
    
  } else  {
    gbmGrid = expand.grid(
      interaction.depth = seq(1, 7, by = 2),
      n.trees = seq(200, 1000, by = 200),
      shrinkage = c(0.001, 0.01, 0.1),
      n.minobsinnode = c(5)
    )
  }
  
  fitcontrol = trainControl(method = "LOOCV",
                            summaryFunction = multiClassSummary)
  
  set.seed(Seed)
  model_fit = train(
    x = train_predictors,
    y = train_outcome,
    method = 'gbm',
    metric = "Kappa",
    tuneGrid = gbmGrid,
    trControl = fitcontrol,
    verbose = F
  )
  
  
  return(model_fit)
}
