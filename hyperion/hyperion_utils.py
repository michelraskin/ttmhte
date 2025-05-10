import sys
import os
dir1 = os.path.abspath(os.path.join(os.getcwd(), '../../analysisFunctions'))    
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
    read_path = './formatteddata/'
    myDfs = {}
    for file in os.listdir(read_path):
        myDfs[file.split('.')[0]] = pd.read_csv(read_path + file)
    return myDfs

def getGroupByPatients(myDf, aColumn, aRename):
    return myDf.groupby(aColumn).agg({'SUBJID': 'count'}).reset_index().rename(columns={aColumn: aRename, 'SUBJID': 'Patients'}).set_index(aRename)


import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def getTrainTestFunctions(aPredictedColumn = 'LastMGCSPositive', aTreatmentColumn = 'Hypothermia', aTestSize = 0.3, aTreatmentSplit = False, aDropColumns = [], aSkipTemp = True, aCv = False, aNumSplits = 3):
    myPredictorsDf = pd.read_csv('predictorsDf.csv')
    myColumns = []
    if (aSkipTemp):
        myColumns = [x for x in myPredictorsDf.columns if 'EMP' in x]
    myFilter = myPredictorsDf['groupe'] != 2

    # Get output data
    myXValue = myPredictorsDf.drop(columns= myColumns + ['CPC_SC3', 'CPC12', 'J0_SEX', 'SUBJID', 'BARTHEL_SC', 'SOFA_SC7', 'DS_DC', 'DAYS_ALIVE_30', 'J0_GLASGOW_CONTROLE', 'J0_CORDA_DOS'])
    myXValue = myXValue
    myXValue = myXValue.select_dtypes(exclude=['object'])
    myYValue = myPredictorsDf[aPredictedColumn]
    myYValue = myYValue.astype(int)
    if (aCv):
        skf = StratifiedKFold(n_splits=aNumSplits, shuffle=True)
        return myXValue, myYValue, skf
    if (aTreatmentSplit):
        myXValue = myXValue.drop(columns=[aTreatmentColumn])
        X_train, X_test, T_train, T_test, y_train, y_test = train_test_split(myXValue, myPredictorsDf[aTreatmentColumn], myYValue, stratify=myPredictorsDf[[aPredictedColumn, aTreatmentColumn]], test_size=aTestSize)
        return myPredictorsDf, X_train, X_test, T_train, T_test, y_train, y_test
    else:
        X_train, X_test, y_train, y_test = train_test_split(myXValue, myYValue, stratify=myPredictorsDf[[aPredictedColumn, aTreatmentColumn]], test_size=aTestSize)
        return myPredictorsDf, X_train, X_test, y_train, y_test