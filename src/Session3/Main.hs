module Session3.Main where
import Torch.Tensor (asTensor, asValue)
import Session3.LinearRegression (linear, cost, xs, ys)
main :: IO ()
main = do
  let sampleA = asTensor([0.555] :: [Float])
  let sampleB = asTensor([94.585026] :: [Float])
  let ysList = asValue ys :: [Float]
  let estimatedYs = linear (sampleA,sampleB) xs

  mapM_
    ( \(expected,correct) -> do
          putStrLn("correct = " ++ show correct)
          putStrLn("estimated = " ++ show expected)
          putStrLn("********")
    )
    (zip ((asValue estimatedYs :: [Float])) ysList)

  putStrLn("cost = " ++ show (cost ys estimatedYs))



  return ()

