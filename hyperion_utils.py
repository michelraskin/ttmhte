import sys
import os
dir1 = os.path.abspath(os.path.join(os.getcwd(), '../analysisFunctions'))  
from analysis_functions import *

def getDf(myDfs, merge):
  myNewDf = myDfs[merge]
  myNewDf = myNewDf.merge(myDfs['ds'], on='SUBJID')
  myNewDf = myNewDf[(myNewDf['DS_DATA_REFUS'] != 1)]
  myNewDf = myNewDf.merge(myDfs['cpc'], on='SUBJID')
  myNewDf = myNewDf.merge(myDfs['bras_rando'], on='SUBJID')
  myNewDf = getType(myNewDf)
  return myNewDf

def getType(myNewDf):
  myCpc1Filter = {
    'Name': 'CPC_SC3',
    'Op': 'EQ', 
    'Value': 1
  }
  myCpc2Filter = {
    'Name': 'CPC_SC3',
    'Op': 'EQ', 
    'Value': 2
  }
  myHypothermiaFilter = {
    'Name': 'groupe',
    'Op': 'EQ',
    'Value': 1
  }
  myNormothermiaFilter = {
    'Name': 'groupe',
    'Op': 'EQ',
    'Value': 0
  }
  myCpc12Filter = ['OR', [[myCpc1Filter], [myCpc2Filter]]]
  myCpc12HypothermiaFilter = ['AND', [myCpc12Filter, [myHypothermiaFilter]]]
  myCpc12NormothermiaFilter = ['AND', [myCpc12Filter, [myNormothermiaFilter]]]
  myNotCpc12NormothermiaFilter = ['AND', [['NOT', [myCpc12Filter]], [myNormothermiaFilter]]]
  myNotCpc12HypothermiaFilter = ['AND', [['NOT', [myCpc12Filter]], [myHypothermiaFilter]]]
  myNewDf['Type'] = 'NotCondideredInStudy'
  myCpc12HypMap = decodeFilterInfo(myNewDf, myCpc12HypothermiaFilter)
  myNewDf.loc[myCpc12HypMap, 'Type'] = 'CPC12HYpothermia'
  myCpc12NotHypMap = decodeFilterInfo(myNewDf, myCpc12NormothermiaFilter)
  myNewDf.loc[myCpc12NotHypMap, 'Type'] = 'CPC12Normothermia'
  myNotCpc12HypMap = decodeFilterInfo(myNewDf, myNotCpc12HypothermiaFilter)
  myNewDf.loc[myNotCpc12HypMap, 'Type'] = 'NotCPC12Hypothermia'
  myNotCpc12NotHypMap = decodeFilterInfo(myNewDf, myNotCpc12NormothermiaFilter)
  myNewDf.loc[myNotCpc12NotHypMap, 'Type'] = 'NotCPC12Normothermia'
  return myNewDf

def readDfs():
  read_path = '../formatteddata\\'
  myDfs = {}
  for file in os.listdir(read_path):
    myDfs[file.split('.')[0]] = pd.read_csv(read_path + file)
  return myDfs

def getGroupByPatients(myDf, aColumn, aRename):
  return myDf.groupby(aColumn).agg({'SUBJID': 'count'}).reset_index().rename(columns={aColumn: aRename, 'SUBJID': 'Patients'}).set_index(aRename)