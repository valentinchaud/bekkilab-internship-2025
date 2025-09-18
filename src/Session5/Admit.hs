{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE OverloadedStrings #-}

module Session5.Admit where

import Prelude
import Control.Monad (forM, forM_, foldM,when)
import qualified Data.Vector as V

import Torch.Tensor (asValue, Tensor, asTensor, toCPU)
import Torch.Functional (mseLoss, toDType,gt,nllLoss')
import Torch.Device (Device(..), DeviceType(..))
import Torch.NN (sample, flattenParameters)
import Torch.Optim (mkAdam)
import Torch.Train (update, showLoss)
import Torch.Layer.MLP (MLPHypParams(..), ActName(..), mlpLayer)
import Torch.Tensor.TensorFactories (asTensor'')
import Torch.DType (DType(Bool, Float))

import Session5.Data
  ( Applicant(..)
  , chanceOfAdmit
  , loadData
  , dataToTensor
  , targetToTensor
  , createBatches
  )

import ML.Exp.Chart (drawLearningCurve)

import Session5.Evaluation
  ( confusionMatrix
  , accuracy
  , precision
  , recall
  , f1Score
  , f1ScorePerClass
  , microF1Score
  , macroF1Score
  , weightedF1Score)
main :: IO ()
main = do
  let numEpochs = 30
      batchSize = 16
      device    = Device CPU 0
      hypParams = MLPHypParams device 7 [(16, Relu), (16, Relu), (1, Sigmoid)]
      lrVal     = 2e-3 :: Float
      lrTensor  = asTensor'' device [lrVal]
      threshold = 0.5 :: Float

  Right trainData <- loadData "data/train.csv"
  Right evalData  <- loadData "data/eval.csv"
  Right validData <- loadData "data/valid.csv"

  let trainBatches = createBatches batchSize trainData

  let xEval = dataToTensor evalData
      yEvalContinuous = targetToTensor evalData

  let yEvalBool = gt yEvalContinuous (asTensor threshold)


  initModel <- sample hypParams
  let initOptim = mkAdam 0 0.9 0.999 (flattenParameters initModel)

  let loop model optim losses epoch
        | epoch > numEpochs = return (model, reverse losses)
        | otherwise = do
            let trainStep (mdl, opt, lossesAcc) (bx, by) = do
                  let yPred = mlpLayer mdl bx
                      loss  = mseLoss by yPred
                      lv    = asValue loss :: Float
                  (mdl', opt') <- update mdl opt loss lrTensor
                  return (mdl', opt', lv : lossesAcc)

            (modelAfterEpoch, optimAfterEpoch, lvs) <- foldM
                trainStep
                (model, optim, [])
                trainBatches

            let avgLoss = sum lvs / fromIntegral (length lvs)
            showLoss 1 epoch avgLoss
            loop modelAfterEpoch optimAfterEpoch (avgLoss:losses) (epoch + 1)

  (trainedModel, losses) <- loop initModel initOptim [] 1

  drawLearningCurve "graph-loss.png" "Training Loss Curve" [("loss", losses)]

  let yEvalPredContinuous = mlpLayer trainedModel xEval
      evalLoss = asValue (mseLoss yEvalContinuous yEvalPredContinuous) :: Float
  putStrLn $ "\n📊 Loss on eval.csv (MSE) : " ++ show evalLoss

  let yEvalPredBool = gt yEvalPredContinuous (asTensor threshold)

  let yEvalPredFloat = Torch.Functional.toDType Torch.DType.Float yEvalPredBool
      yEvalFloat     = Torch.Functional.toDType Torch.DType.Float yEvalBool


  putStrLn "\n📈 Eval metrics on eval.csv :"

  let confMatrix = confusionMatrix yEvalPredFloat yEvalFloat
  putStrLn $ "Confusion matrix :\n" ++ show confMatrix

  let acc = accuracy yEvalPredFloat yEvalFloat
  putStrLn $ "Accuracy : " ++ show acc

  let prec = precision yEvalPredFloat yEvalFloat
  putStrLn $ "Precision : " ++ show prec

  let rec = recall yEvalPredFloat yEvalFloat
  putStrLn $ "Recall : " ++ show rec

  let (f1_pos_class, f1_neg_class) = f1ScorePerClass yEvalPredFloat yEvalFloat
  putStrLn $ "F1 Score (Per Class) - Positive: " ++ show f1_pos_class ++ ", Negative: " ++ show f1_neg_class

  let microF1 = microF1Score yEvalPredFloat yEvalFloat
  putStrLn $ "Micro-F1 Score : " ++ show microF1

  let macroF1 = macroF1Score yEvalPredFloat yEvalFloat
  putStrLn $ "Macro-F1 Score : " ++ show macroF1

  let weightedF1 = weightedF1Score yEvalPredFloat yEvalFloat
  putStrLn $ "Weighted-F1 Score : " ++ show weightedF1


  putStrLn "\n🔎 Predictions on valid.csv :"
  forM_ (V.toList $ V.take 10 validData) $ \applicant -> do
    let input :: [Float]
        input = [ fromIntegral (greScore applicant)
                , fromIntegral (toeflScore applicant)
                , fromIntegral (universityRating applicant)
                , realToFrac (sop applicant) :: Float
                , realToFrac (lor applicant) :: Float
                , realToFrac (cgpa applicant) :: Float
                , fromIntegral (research applicant)
                ]
        tensorInput = asTensor'' device input
        predContinuous = mlpLayer trainedModel tensorInput
        predictedContinuousValue = asValue predContinuous :: Float
        predictedBinaryValue = if predictedContinuousValue > threshold then 1 else 0
        actual      = chanceOfAdmit applicant
        actualBinary = if actual > realToFrac threshold then 1 else 0
    putStrLn $ "Predicted (continu): " ++ show predictedContinuousValue ++ " | Predicted (binaire): " ++ show predictedBinaryValue ++ " | Real (continu): " ++ show actual ++ " | Real (binaire): " ++ show actualBinary

  putStrLn "\n✅ Training model finished !"
