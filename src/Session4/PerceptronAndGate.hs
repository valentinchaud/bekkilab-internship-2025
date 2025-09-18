module Session4.PerceptronAndGate where

import Torch hiding (step)
import Torch.Functional.Internal (where')

trainingData :: [([Int],Int)]
trainingData = [([1,1],1),([1,0],0),([0,1],0),([0,0],0)]

step :: Tensor -> Tensor
step x = where' (x >=. 0) (ones' (shape x)) (zeros' (shape x))

perceptron ::
	Tensor -> -- x
	Tensor -> -- weights
	Tensor -> -- bias
	Tensor    -- output
perceptron x weights b = step ( add b (matmul x (transpose2D weights)))


calculateError ::
  Tensor -> -- expectedOutput
  Tensor -> -- predictedOutput
  Tensor    -- total error
calculateError expectedOutput predictedOutput = sub expectedOutput predictedOutput

train::
   Tensor -> -- weights
   Tensor -> -- bias
   Float ->  -- learningRate
   Int ->    -- max epochs
   Int ->    -- current epoch
   [(Tensor, Tensor)] -> -- history of weights
   [(Tensor, Tensor)]    -- new weights history
train weights b learningRate maxEpochs currentEpoch history =
  if currentEpoch >= maxEpochs
  then history
  else
    let inputs = stack (Dim 0) (map (asTensor . (map (fromIntegral :: Int -> Float)) . fst) trainingData)
        expectedOutputs = reshape [4,1] $ asTensor (map snd trainingData)
        predictedOutputs = perceptron inputs weights b
        errors = calculateError expectedOutputs predictedOutputs

        weightUpdates = mulScalar learningRate (matmul (transpose2D errors) inputs)

        newWeights = add weights weightUpdates

        biasUpdate = mulScalar learningRate (sumAll errors)
        newBias = add b biasUpdate

        newHistory = history ++ [(newWeights, newBias)]
    in train newWeights newBias learningRate maxEpochs (currentEpoch + 1) newHistory

main :: IO ()
main = do
  weights <- randintIO' 0 10 [1, 2]
  b <- randintIO' 0 10 [1, 1]

  let inputs = stack (Dim 0) (map (asTensor . (map (fromIntegral :: Int -> Float)) . fst) trainingData)
      expectedOutputs = reshape [4,1] $ asTensor (map snd trainingData)

  let initialPredictedOutputs = perceptron inputs weights b
  let initialElementErrors = calculateError expectedOutputs initialPredictedOutputs
  putStrLn $ "Sum of initial errors: " ++ show (Torch.abs (sumAll initialElementErrors))

  let learningRate = 0.1 :: Float
  let maxEpochs = 60

  let history = train weights b learningRate maxEpochs 0 [(weights, b)]
  let (finalWeights, finalBias) = last history

  putStrLn $ "Final weights: " ++ show (finalWeights)
  putStrLn $ "Final bias: " ++ show (finalBias)

  let finalPerceptron = perceptron inputs finalWeights finalBias
  putStrLn $ "Final perceptron" ++ show(finalPerceptron)
  putStrLn $ "Final error" ++ show (sumAll (calculateError finalPerceptron expectedOutputs))

  return ()
