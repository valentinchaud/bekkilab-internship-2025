module Session3.Main where

import Torch.Tensor (asTensor, asValue, Tensor)
import Torch.Functional (mul,sub)
import Control.Monad(foldM_)
import Text.Printf (printf)
-- Make sure foldLoop is in scope if defined in LinearRegression,
-- or import it if it's defined elsewhere.
import Session3.LinearRegression

main :: IO ()
main = do
  putStrLn "--- Initial Values ---"
  let initialA = asTensor(0.00:: Float)
  let initialB = asTensor(0.00 :: Float)
  let initialCost = cost ys (linear (initialA, initialB) xs)
  let xsList = asValue xs :: [Float]
  let ysList = asValue ys :: [Float]
  putStrLn $ "Initial Cost: " ++ show (asValue initialCost :: Float)
  putStrLn "--- Starting Training ---"

  let learningRate = asTensor(28e-6 :: Float) -- ** Adjust this value **
  let numEpochs = 1000000 :: Int;
  mapM_
    ( \(x,y) -> do
        let estimatedY = linear (initialA, initialB) (asTensor(x :: Float))
        putStrLn ("correct = " ++ show y)
        putStrLn ("estimated = " ++ show estimatedY)
        putStrLn "********"
    )
    (zip xsList ysList)

  foldM_ (\(currentA, currentB) epoch -> do
      let gradientA = calculateNewA xs ys currentA currentB
      let gradientB = calculateNewB xs ys currentA currentB

      let newA = sub currentA (mul learningRate gradientA)
      let newB = sub currentB (mul learningRate gradientB)

      let currentCost = cost ys (linear (newA, newB) xs)

      -- Print epoch info
      printf "Epoch %d: Slope (a)=%.4f, Intercept (b)=%.4f, Cost=%.4f\n"
             epoch
             (asValue newA :: Float)
             (asValue newB :: Float)
             (asValue currentCost :: Float)

      return (newA, newB)

    ) (initialA, initialB) [1..numEpochs]

