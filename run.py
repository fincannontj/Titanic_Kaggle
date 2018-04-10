# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 20:41:56 2018

@author: fincannontj
"""
from titanic_model_functions import *
import pandas as pd
import numpy as np
import os.path
import matplotlib.pyplot as plt
import seaborn as sns


def importer(filename):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path=dir_path+'/Data/'+ filename 
    df = pd.read_csv(file_path)
    return df

def main():  
    train = importer('train.csv')  

    holdout = importer('test.csv')   



    train = pre_process(train)

    holdout = pre_process(holdout)

    cols = select_features(train)

    result = select_model(train,cols)


    #ensembles= ensemble_modeling(result, cols, holdout)
    #plot heatmap

    best_rf_model = result[2]["best_model"]
    best_vc_model = result[3]["best_model"]

    save_submission_file(best_rf_model,holdout,cols, filename='Submission1.csv')
    save_submission_file(best_rf_model,holdout, cols, filename='Submission2.csv')
    


if __name__ == '__main__':
    main()