module Session5.Evaluation where

import Torch.Tensor (Tensor,size,asValue,asTensor)
import Torch.Functional.Internal (sumAll,eq)
import Torch.DType (DType(Int64))
import Torch.Functional (toDType,logicalAnd)
import Torch.TensorFactories(onesLike,zerosLike)
truePositive ::
    Tensor ->
    Tensor ->
    Int
truePositive predicted expected =
    let ones = onesLike expected
        expectedPositive = eq expected ones
        predictedPositive = eq predicted ones
        comparedElem = logicalAnd predictedPositive expectedPositive
        matchesInt = toDType Int64 comparedElem
        truepos = sumAll matchesInt Int64
    in asValue truepos :: Int

trueNegative ::
    Tensor ->
    Tensor ->
    Int
trueNegative predicted expected =
    let zeros = zerosLike expected
        expectedNegative = eq expected zeros
        predictedNegative = eq predicted zeros
        comparedElem = logicalAnd predictedNegative expectedNegative
        matchesInt = toDType Int64 comparedElem
        trueneg = sumAll matchesInt Int64
    in asValue trueneg :: Int

falsePositive :: Tensor -> Tensor -> Int
falsePositive predicted expected =
    let
        ones = onesLike expected
        zeros = zerosLike expected

        -- Vérifier où 'predicted' est égal à 1 (positif)
        predictedIsPositive = eq predicted ones
        -- Vérifier où 'expected' est égal à 0 (négatif)
        expectedIsNegative = eq expected zeros

        -- 'comparedElem' est True où predicted est positif ET expected est négatif
        comparedElem = logicalAnd predictedIsPositive expectedIsNegative

        matchesInt = toDType Int64 comparedElem
        fpCount = sumAll matchesInt Int64
    in
        asValue fpCount :: Int


falseNegative ::
    Tensor ->
    Tensor ->
    Int
falseNegative predicted expected =
    let
        ones = onesLike expected
        zeros = zerosLike expected

        predictedIsNegative = eq predicted zeros

        expectedIsPositive = eq expected ones

        comparedElem = logicalAnd predictedIsNegative expectedIsPositive

        matchesInt = toDType Int64 comparedElem
        fnCount = sumAll matchesInt Int64
    in
        asValue fnCount :: Int

accuracy ::
    Tensor -> -- predicted result
    Tensor -> -- expected result
    Double -- accuracy
accuracy predicted expected = fromIntegral (truePositive predicted expected + trueNegative predicted expected) / fromIntegral  (size 0 predicted)

precision ::
    Tensor ->
    Tensor ->
    Double
precision predicted expected = fromIntegral(truePositive predicted expected) / (fromIntegral (truePositive predicted expected + falsePositive predicted expected))


recall ::
    Tensor ->
    Tensor ->
    Double
recall predicted expected = fromIntegral(truePositive predicted expected) / (fromIntegral (truePositive predicted expected + falseNegative predicted expected))

confusionMatrix ::
   Tensor ->
   Tensor ->
   Tensor
confusionMatrix predicted expected =
   let tp_value = truePositive predicted expected
       tn_value = trueNegative predicted expected
       fp_value = falsePositive predicted expected
       fn_value = falseNegative predicted expected
       confusion_matrix = [[tp_value,fn_value], [fp_value, tn_value]]
   in
       (asTensor confusion_matrix)
f1Score ::
   Tensor ->
   Tensor ->
   Double
f1Score predicted expected = 2 * ((precision predicted expected) * (recall predicted expected)) / ((precision predicted expected) + (recall predicted expected))

f1ScoreNegativeClass :: Tensor -> Tensor -> Double
f1ScoreNegativeClass predicted expected =
    let tn_val = fromIntegral $ trueNegative predicted expected
        fp_val = fromIntegral $ falsePositive predicted expected
        fn_val = fromIntegral $ falseNegative predicted expected
        -- For negative class:
        -- Precision_neg = TN / (TN + FN)
        -- Recall_neg = TN / (TN + FP)
        precision_neg = if (tn_val + fn_val) == 0 then 0.0 else tn_val / (tn_val + fn_val)
        recall_neg = if (tn_val + fp_val) == 0 then 0.0 else tn_val / (tn_val + fp_val)
    in if (precision_neg + recall_neg) == 0 then 0.0 else 2 * (precision_neg * recall_neg) / (precision_neg + recall_neg)

f1ScorePerClass :: Tensor -> Tensor -> (Double, Double)
f1ScorePerClass predicted expected =
    let f1_pos = f1Score predicted expected
        f1_neg = f1ScoreNegativeClass predicted expected
    in (f1_pos, f1_neg)

microF1Score :: Tensor -> Tensor -> Double
microF1Score predicted expected =
    let tp = fromIntegral $ truePositive predicted expected
        fp = fromIntegral $ falsePositive predicted expected
        fn = fromIntegral $ falseNegative predicted expected

        precision_micro = if (tp + fp) == 0 then 0.0 else tp / (tp + fp)
        recall_micro = if (tp + fn) == 0 then 0.0 else tp / (tp + fn)
    in if (precision_micro + recall_micro) == 0
        then 0.0
        else 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro)

macroF1Score :: Tensor -> Tensor -> Double
macroF1Score predicted expected =
    let (f1_pos, f1_neg) = f1ScorePerClass predicted expected
    in (f1_pos + f1_neg) / 2.0

weightedF1Score :: Tensor -> Tensor -> Double
weightedF1Score predicted expected =
    let (f1_pos, f1_neg) = f1ScorePerClass predicted expected

        tp = fromIntegral $ truePositive predicted expected
        fn = fromIntegral $ falseNegative predicted expected
        tn = fromIntegral $ trueNegative predicted expected
        fp = fromIntegral $ falsePositive predicted expected

        support_pos = tp + fn -- True instances of positive class
        support_neg = tn + fp -- True instances of negative class
        total_support = support_pos + support_neg

    in if total_support == 0
        then 0.0
        else (f1_pos * support_pos + f1_neg * support_neg) / total_support
