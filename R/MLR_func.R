
fitlmr = function(train_predictors, train_outcome, Seed)
{
  fitcontrol = trainControl(method = "LOOCV",
                            summaryFunction = multiClassSummary)
  set.seed(Seed)
  
  model_fit = train(
    x = train_predictors,
    y = train_outcome,
    method = 'multinom' ,
    trControl = fitcontrol,
    tuneLength = 1 ,
    metric = 'Kappa'
  )

  return(model_fit)
}
