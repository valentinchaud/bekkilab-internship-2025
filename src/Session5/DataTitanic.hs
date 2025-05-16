-- titanic_preprocessing.hs
-- Prétraitement et mise en batch du dataset Titanic en Haskell avec Hasktorch

{-# LANGUAGE OverloadedStrings, RecordWildCards #-}
module Session5.DataTitanic where

import qualified Data.ByteString.Lazy as BL
import Data.Csv (FromNamedRecord(..), decodeByName, (.:))
import qualified Data.Vector as V
import Torch.Tensor (Tensor, asTensor, shape, asValue)
import Control.Monad (when)
import Data.List (transpose)

-- 1. Définir le type Passenger et instancier FromNamedRecord
data Passenger = Passenger
  { pPassengerId :: !Int
  , pSurvived    :: !Int
  , pPclass      :: !Int
  , pName        :: !String
  , pSex         :: !String
  , pAge         :: !(Maybe Double)
  , pSibSp       :: !Int
  , pParch       :: !Int
  , pTicket      :: !String
  , pFare        :: !Double
  , pCabin       :: !(Maybe String)
  , pEmbarked    :: !(Maybe String)
  } deriving (Show)

instance FromNamedRecord Passenger where
  parseNamedRecord r = Passenger
    <$> r .:  "PassengerId"
    <*> r .:  "Survived"
    <*> r .:  "Pclass"
    <*> r .:  "Name"
    <*> r .:  "Sex"
    <*> r .: "Age"
    <*> r .:  "SibSp"
    <*> r .:  "Parch"
    <*> r .:  "Ticket"
    <*> r .:  "Fare"
    <*> r .: "Cabin"
    <*> r .: "Embarked"

-- 2. Chargement du fichier
loadPassengers :: FilePath -> IO (V.Vector Passenger)
loadPassengers path = do
  csvData <- BL.readFile path
  case decodeByName csvData of
    Left err      -> error ("Parsing CSV: " ++ err)
    Right (_, vec)-> return vec

data PreprocessingParams = PreprocessingParams
  { ppMeanAge :: Double
  , ppMinVals :: [Float] -- One min for each feature column
  , ppMaxVals :: [Float] -- One max for each feature column
  } deriving (Show)

calculatePreprocessingParams :: V.Vector Passenger -> PreprocessingParams
calculatePreprocessingParams trainPassengers =
  let
    agesInitial = V.mapMaybe pAge trainPassengers
    meanAgeInitial = if V.null agesInitial then 0 else sum agesInitial / fromIntegral (V.length agesInitial)
    passengersWithImputedAge = V.map (\p@Passenger{..} -> p{ pAge = Just (maybe meanAgeInitial id pAge) }) trainPassengers

    rawTrainingFeatures = V.toList $ V.map toNumericInternal passengersWithImputedAge
        where
            toNumericInternal Passenger{..} =
                [ realToFrac (maybe 0 id pAge)
                , realToFrac pFare
                , realToFrac pSibSp
                , realToFrac pParch
                , encodeSex pSex
                , encodePclass pPclass
                , encodeCabin pCabin
                , encodeEmbarked pEmbarked
                ]

    cols = transpose rawTrainingFeatures
    minVals = map minimum cols
    maxVals = map maximum cols
  in PreprocessingParams meanAgeInitial minVals maxVals

applyPreprocessParams :: PreprocessingParams -> V.Vector Passenger -> ([[Float]], [[Float]]) -- (raw, normalized)
applyPreprocessParams PreprocessingParams{..} passengers =
  let
    passengersWithImputedAge = V.map (\p@Passenger{..} -> p{ pAge = Just (maybe ppMeanAge id pAge) }) passengers

    rawFeatures = V.toList $ V.map toNumericInternal passengersWithImputedAge
         where
            toNumericInternal Passenger{..} =
                [ realToFrac (maybe 0 id pAge)
                , realToFrac pFare
                , realToFrac pSibSp
                , realToFrac pParch
                , encodeSex pSex
                , encodePclass pPclass
                , encodeCabin pCabin
                , encodeEmbarked pEmbarked
                ]

    normalizeVal minV maxV val = if (maxV - minV) == 0 then 0 else (val - minV) / (maxV - minV)

    normalizedFeatures = map (\featureRow ->
                                zipWith3 (\minV maxV val -> normalizeVal minV maxV val) ppMinVals ppMaxVals featureRow
                             ) rawFeatures
  in (rawFeatures, normalizedFeatures)


imputeAge :: V.Vector Passenger -> V.Vector Passenger
imputeAge v =
  let ages  = V.mapMaybe pAge v
      meanA = sum ages / fromIntegral (V.length ages)
  in V.map (\p@Passenger{..} -> p{ pAge = Just (maybe meanA id pAge) }) v

encodeSex :: String -> Float
encodeSex "male"   = 0; encodeSex "female" = 1; encodeSex _ = 0

encodePclass :: Int -> Float
encodePclass 1 = 0; encodePclass 2 = 1; encodePclass 3 = 2; encodePclass _ = 0

encodeEmbarked :: Maybe String -> Float
encodeEmbarked (Just "C") = 0; encodeEmbarked (Just "Q") = 1; encodeEmbarked (Just "S") = 2; encodeEmbarked _ = 0

encodeCabin :: Maybe String -> Float
encodeCabin Nothing = 0; encodeCabin (Just _) = 1

normalizeColumns :: [[Float]] -> [[Float]]
normalizeColumns rows =
  let cols    = transpose rows
      normCol xs = let mn = minimum xs; mx = maximum xs; r = mx - mn
                    in if r==0 then replicate (length xs) 0 else map (\x -> (x-mn)/r) xs
  in transpose (map normCol cols)

-- Retourne (rawFeatures, normalizedFeatures)
preprocess :: V.Vector Passenger -> ([[Float]], [[Float]])
preprocess v =
  let v'        = imputeAge v
      rawFeats  = V.toList $ V.map toNumeric v'
      normFeats = normalizeColumns rawFeats
  in (rawFeats, normFeats)
  where
    toNumeric Passenger{..} =
      [ realToFrac (maybe 0 id pAge)
      , realToFrac pFare
      , realToFrac pSibSp
      , realToFrac pParch
      , encodeSex pSex
      , encodePclass pPclass
      , encodeCabin pCabin
      , encodeEmbarked pEmbarked
      ]

labels :: V.Vector Passenger -> [Float]
labels v = V.toList $ V.map (realToFrac . pSurvived) v


-- 5. Création de mini-batches
createBatches :: Int -> V.Vector Passenger -> [V.Vector Passenger]
createBatches batchSize vec = go 0
  where
    n = V.length vec
    go i
      | i >= n    = []
      | otherwise = let len   = min batchSize (n - i)
                        chunk = V.slice i len vec
                    in chunk : go (i + batchSize)

-- 6. Test rapide dans main
main :: IO ()
main = do
  raw <- loadPassengers "data/titanic.csv"
  let batches = createBatches 32 raw
  when (null batches) $ error "No data batches created"
  let chunk0 = head batches
  -- Extraire informations string
  let names0   = V.toList $ V.map pName chunk0
      tickets0 = V.toList $ V.map pTicket chunk0
      cabins0  = V.toList $ V.map (maybe "" id . pCabin) chunk0
      emb0     = V.toList $ V.map (maybe "" id . pEmbarked) chunk0
  -- Preprocessing numérique
  let (raw0, norm0) = preprocess chunk0
      labels0       = labels chunk0

  putStrLn "--- Raw string infos first batch ---"
  putStrLn "Names:" >> mapM_ print (take 3 names0)
  putStrLn "Tickets:" >> mapM_ print (take 3 tickets0)
  putStrLn "Cabins:" >> mapM_ print (take 3 cabins0)
  putStrLn "Embarked:" >> mapM_ print (take 3 emb0)

  putStrLn "--- Raw features first batch (non-normalized) ---"
  mapM_ print (take 3 raw0)
  putStrLn "--- Normalized features first batch ---"
  mapM_ print (take 3 norm0)
  putStrLn "--- Labels first batch ---"
  print (take 3 labels0)
  return ()
