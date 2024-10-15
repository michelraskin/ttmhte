# ttmhte
 

 ## Steps to run 

 1. Copy files from `S:\LCICM\Databases\cardiac_arrest_nantes\Donnees a transferer` into a folder outside of the repo, call it `replicatedfiles`, also create a new directory called `formatteddata`
 2. Run `hyperionInitialPreprocessing.ipynb`
 3. Run `hyperionGeneratePredictorsDf.ipynb`

 ## File descritions:

 ### hyperionInitialPreprocessing.ipynb

 Converts from the initial data format to csvs and translates description file

 ### hyperionGeneratePredictorsDf.ipynb

 Creates the dataframe that will be shared across all analysis

 ### hyperionDataVisualization.ipynb

 Data visualization of the different components of hyperion

 ### hyperionBART.ipynb

 BART analysis on hyperion

 ### hyperionUnsupervised.ipynb

 Unsupervised machine learning analysis on hyperion

 ### scratchpad

 Files used to iterate on progress, not meant to be the main source of analysis
