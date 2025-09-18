{-# LANGUAGE OverloadedStrings #-}

module Session5.Data where

import Torch.Tensor (Tensor, asTensor)
import qualified Data.ByteString.Lazy as BL
import Data.Csv
import qualified Data.Vector as V
import Control.Applicative
import Control.Monad (mzero)



data Applicant = Applicant
  { serialNo      :: !Int
  , greScore       :: !Int
  , toeflScore     :: !Int
  , universityRating :: !Int
  , sop            :: !Double
  , lor            :: !Double
  , cgpa           :: !Double
  , research       :: !Int
  , chanceOfAdmit  :: !Double
  } deriving (Show)

instance FromRecord Applicant where
   parseRecord v
     | V.length v == 9 = Applicant
        <$> v.! 0
        <*> v.! 1
        <*> v .! 2
        <*> v .! 3
        <*> v .! 4
        <*> v .! 5
        <*> v .! 6
        <*> v .! 7
        <*> v .! 8
     | otherwise = mzero

instance FromNamedRecord Applicant where
  parseNamedRecord r = Applicant
    <$> r .: "Serial No."
    <*> r .: "GRE Score"
    <*> r .: "TOEFL Score"
    <*> r .: "University Rating"
    <*> r .: "SOP"
    <*> r .: "LOR "
    <*> r .: "CGPA"
    <*> r .: "Research"
    <*> r .: "Chance of Admit "

loadData :: FilePath -> IO (Either String (V.Vector Applicant))
loadData path = do
     csvData <- BL.readFile path
     case decode HasHeader csvData of
       Left err -> do
          return $ Left ("Erreur de parsing CSV: " ++ err)
       Right v  -> do
          return $ Right v

dataToTensor :: V.Vector Applicant -> Tensor
dataToTensor va = asTensor (elements :: [[Float]])
  where
    elements = V.toList $ V.map (\r -> [
      fromIntegral (greScore r) :: Float,
      fromIntegral (toeflScore r) :: Float,
      fromIntegral (universityRating r) :: Float,
      realToFrac (sop r) :: Float,
      realToFrac (lor r) :: Float,
      realToFrac (cgpa r) :: Float,
      fromIntegral (research r) :: Float
      ]) va

targetToTensor :: V.Vector Applicant -> Tensor
targetToTensor va = asTensor (elements :: [Float])
  where
    elements = V.toList $ V.map (\r -> realToFrac (chanceOfAdmit r) :: Float) va

cgpaToTensor :: V.Vector Applicant -> Tensor
cgpaToTensor va = asTensor (elements :: [Float])
  where
    elements = V.toList $ V.map (\r -> realToFrac (cgpa r) :: Float) va
-- | Génère une liste de mini‐batches de taille batchSize
createBatches
  :: Int                    -- ^ taille du batch
  -> V.Vector Applicant     -- ^ dataset complet
  -> [(Tensor, Tensor)]     -- ^ liste de (features, targets)
createBatches batchSize vec = go 0
  where
    n = V.length vec
    go i
      | i >= n    = []
      | otherwise =
          let len   = min batchSize (n - i)
              chunk = V.slice i len vec
          in  ( dataToTensor   chunk
              , targetToTensor chunk
              )
             : go (i + batchSize)
