#! /usr/bin/python

import pandas as pd
import numpy as np
import gzip
import dill as pickle
import math
from scipy.optimize import curve_fit

import argparse
from tqdm import tqdm


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Set up argument parser
parser = argparse.ArgumentParser(description='compute relaxation.')
parser.add_argument('--year', type=int, required=True, help='Year for the dataset')
parser.add_argument('--outdir', type=str, required=True, help='Output directory for the  model')
parser.add_argument('--verbose', type=str2bool,  help='set true for verbose')
args = parser.parse_args()
OUTDIR=args.outdir
year=args.year
verbose=args.verbose

datapath=f'{OUTDIR}/relaxation_{year}.csv'

df=pd.read_csv(datapath,index_col=0).T.rename(columns={0:'R',1:'L',2:'RL'})




def triangle_area(a, b, c):
    """Calculate the area of a triangle given its side lengths using Heron's formula."""
    s = (a + b + c) / 2
    a=(s * (s - a) * (s - b) * (s - c))
    if a > 0:
         return math.sqrt(s * (s - a) * (s - b) * (s - c))
    else:
         return 0.

def calculate_changes(triangle1, triangle2):
    """Calculate changes in area and side lengths between two triangles."""
    area1 = triangle_area(*triangle1)
    area2 = triangle_area(*triangle2)

    area_change = area2 - area1
    #side_length_changes = [triangle2[i] - triangle1[i] for i in range(3)]

    return area_change


def getTau(df):

    Z=df.head(1).values[0]
    def getChange(row,R0=Z[0],L0=Z[1],RL0=Z[2]):
        return calculate_changes((R0,L0,RL0),(row.R,row.L,row.RL))
    
    df['dA']=df.apply(getChange,axis=1)
    N=4
    df_= df[N:]

    response_data = df_['dA'].values
    time = np.arange(len(response_data))

    def decay_function(t, A, tau, C):
        return A * np.exp(-t / tau) + C

    params, covariance = curve_fit(decay_function, time, response_data)
    return params[1],np.sqrt(covariance[1][1])


tau,cov=getTau(df)

print(year,tau,cov)
pd.DataFrame([[year, tau, cov]], columns=['year',  'tau', 'cov']).to_csv(f'{OUTDIR}/tau_{year}.csv')

