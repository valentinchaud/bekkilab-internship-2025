{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE OverloadedStrings #-}

module Session5.AdmitNLL where

import Prelude hiding (exp)
import Control.Monad (forM, forM_, foldM,when)
import qualified Data.Vector as V

-- hasktorch
import Torch.Tensor (asValue, Tensor, asTensor, toCPU,select) -- Added gt, toCPU, asTensor
import Torch.Functional (toDType,gt,nllLoss',transpose,logSoftmax,KeepDim(RemoveDim), Dim(..),argmax,squeezeAll,exp) -- MODIFIED: Added toDType, removed round
import Torch.Device (Device(..), DeviceType(..))
import Torch.NN (sample, flattenParameters)
import Torch.Optim (mkAdam)
import Torch.Train (update, showLoss)
import Torch.Layer.MLP (MLPHypParams(..), ActName(..), mlpLayer)
import Torch.Tensor.TensorFactories (asTensor'')
import Torch.DType (DType(Bool, Float,Int64))

import Session5.Data
  ( Applicant(..)
  , chanceOfAdmit
  , loadData
  , dataToTensor
  , targetToTensor
  , createBatches
  )

-- courbe d'apprentissage
import ML.Exp.Chart (drawLearningCurve)

-- evaluation metrics
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
      hypParams = MLPHypParams device 7 [(16, Relu), (16, Relu), (2, Id)]
      lrVal     = 2e-3 :: Float
      lrTensor  = asTensor'' device [lrVal]
      threshold = 0.5 :: Float

  Right trainData <- loadData "data/train.csv"
  Right evalData  <- loadData "data/eval.csv"
  Right validData <- loadData "data/valid.csv"

  let trainBatches = createBatches batchSize trainData

  let xEval = dataToTensor evalData
      yEvalContinuous = targetToTensor evalData -- Continuous target values (0.0 to 1.0)
      yEvalTargetIndices = squeezeAll $ Torch.Functional.toDType Int64 (gt yEvalContinuous (asTensor threshold))


  initModel <- sample hypParams
  let initOptim = mkAdam 0 0.9 0.999 (flattenParameters initModel)

  let loop model optim losses epoch
        | epoch > numEpochs = return (model, reverse losses)
        | otherwise = do
            let trainStep (mdl, opt, lossesAcc) (bx, by_continuous) = do
                  let by_indices = squeezeAll $ toDType Int64 (gt by_continuous (asTensor threshold))
                  let outputLogits = mlpLayer mdl bx
                      logProbs = logSoftmax (Dim 1) outputLogits
                      loss  = nllLoss' by_indices logProbs
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

  drawLearningCurve "graph-loss-nll.png" "Training Loss Curve" [("loss", losses)]

  let yEvalPredLogits = mlpLayer trainedModel xEval -- Shape: [numEvalSamples, 2]
      yEvalPredLogProbs = logSoftmax (Dim 1) yEvalPredLogits
      evalLoss = asValue (nllLoss' yEvalTargetIndices yEvalPredLogProbs) :: Float -- Ensure yEvalTargetIndices is LongTensor and squeezed
  putStrLn $ "\n📊 Loss on eval.csv (NLLoss) : " ++ show evalLoss

  let yEvalPredIndices = argmax (Dim 1) RemoveDim yEvalPredLogProbs -- Shape: [numEvalSamples]

  let yEvalPredFloat = Torch.Functional.toDType Torch.DType.Float yEvalPredIndices
      yEvalFloat     = Torch.Functional.toDType Torch.DType.Float yEvalTargetIndices -- yEvalTargetIndices was already 0 or 1


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
        predLogits = mlpLayer trainedModel tensorInput
        predLogProbs = logSoftmax (Dim 0) predLogits
        predProbs = exp (select 0 1 predLogProbs)
        predictedContinuousValue = asValue predProbs :: Float
        predictedBinaryValue = if predictedContinuousValue > threshold then 1 else 0
        actual      = chanceOfAdmit applicant -- This is continuous
        actualBinary = if actual > realToFrac threshold then 1 else 0 -- Binary actual for comparison
    putStrLn $ "Predicted (continu): " ++ show predictedContinuousValue ++ " | Predicted (binaire): " ++ show predictedBinaryValue ++ " | Real (continu): " ++ show actual ++ " | Real (binaire): " ++ show actualBinary

  putStrLn "\n✅ Finished to train model !"
