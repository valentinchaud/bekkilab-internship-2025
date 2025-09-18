module Session5.Main where

import Torch.Tensor (asTensor)
import Session5.Evaluation
import Session5.Data
main :: IO ()
main = do
    let predicted = asTensor ([1, 0, 1, 0, 0, 1, 1, 0] :: [Int])
        expected  = asTensor ([1, 0, 1, 1, 0, 0, 1, 0] :: [Int])

    putStrLn "Testing classification metrics with:"
    putStrLn $ "Predicted: " ++ show ([1, 0, 1, 0, 0, 1, 1, 0] :: [Int])
    putStrLn $ "Expected:  " ++ show ([1, 0, 1, 1, 0, 0, 1, 0] :: [Int])
    putStrLn ""

    let tp = truePositive predicted expected
        tn = trueNegative predicted expected
        fp = falsePositive predicted expected
        fn = falseNegative predicted expected
        acc = accuracy predicted expected
        prec = precision predicted expected
        rec = recall predicted expected
        f1 = f1Score predicted expected
        cm = confusionMatrix predicted expected
    trainData <- loadData "data/train.csv"
    evalData <- loadData "data/eval.csv"
    validData <- loadData "data/valid.csv"

    putStrLn $ "True Positives: " ++ show tp ++ " (Expected: 3)"
    putStrLn $ "True Negatives: " ++ show tn ++ " (Expected: 3)"
    putStrLn $ "False Positives: " ++ show fp ++ " (Expected: 1)"
    putStrLn $ "False Negatives: " ++ show fn ++ " (Expected: 1)"
    putStrLn $ "Accuracy: " ++ show acc ++ " (Expected: 0.75)"
    putStrLn $ "Precision: " ++ show prec ++ " (Expected: 0.75)"
    putStrLn $ "Recall: " ++ show rec ++ " (Expected: 0.75)"
    putStrLn $ "F1 Score: " ++ show f1 ++ " (Expected: 0.75)"
    putStrLn $ "Confusion Matrix (expected [[3,1],[1,3]]):" ++ show cm
