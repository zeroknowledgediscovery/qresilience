#! /usr/bin/python

import os
import pandas as pd
import numpy as np
from quasinet.qnet import Qnet, save_qnet, load_qnet, qdistance
from quasinet.qsampling import qsample
import argparse
from tqdm import tqdm
import math
from scipy.optimize import curve_fit


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
parser = argparse.ArgumentParser(description='Train Qnet models for social data.')
parser.add_argument('--year', type=int, required=True, help='Year for the dataset')
parser.add_argument('--samplesize', type=int, default=1000, help='Sample size for the training data')
parser.add_argument('--outdir', type=str, required=True, help='Output directory for the Qnet model')
parser.add_argument('--verbose', type=str2bool,  help='set true for verbose')
parser.add_argument('--stratify', type=str, default=None, help='stratification variable')
args = parser.parse_args()

# Use arguments
year = args.year
samplesize = args.samplesize
OUTDIR = args.outdir
VAR=args.stratify

POLEFILE = './data/polar_vectors.csv'
MUTFILE = './data/immutable.csv'
FEATURESBYYEAR = './data/features_by_year_GSS.csv'
features_by_year = pd.read_csv(FEATURESBYYEAR,
                               keep_default_na=True,
                               index_col=0).set_index('year').loc[year].values[0]
cols=eval(features_by_year) 
data = pd.read_csv(f'./data/gss_{year}.csv', keep_default_na=False, dtype=str)[cols]

training_data = data.sample(samplesize)

if VAR:
    if args.verbose:
        print('stratification requested ',VAR)
    
    vdict=data[VAR].value_counts().to_dict()
    data_s={k:training_data[training_data[VAR]==k] for k in vdict.keys()}
    training_data={k:d.loc[:, d.ne('').any()] for k,d in data_s.items()}
    training_index = {k:training_data[k].index.values for k in vdict.keys()}
    qmodel_path = {k:f'{OUTDIR}/gss_{year}{k}.pkl.gz' for k in vdict.keys()}

    for k in vdict.keys():
        if not os.path.exists(qmodel_path[k]):
            if args.verbose:
                print('training qnet ...',k)
            X_training = training_data[k].values.astype(str)
            Q = Qnet(feature_names=training_data[k].columns, alpha=.1)
            Q.fit(X_training)
            Q.training_index = training_index[k]
            save_qnet(Q, qmodel_path[k].replace('.gz',''), gz=True)
            if args.verbose:
                print('saved qnet',k)
            
        else:
            Q={k:load_qnet(qmodel_path[k]) for k in vdict.keys()}
    
else:
    qmodel_path = f'{OUTDIR}/gss_{year}.pkl.gz'
    if not os.path.exists(qmodel_path):
        X_training = training_data.values.astype(str)
        Q = Qnet(feature_names=training_data.columns, alpha=.1)
        Q.fit(X_training)
        Q.training_index = training_index
        save_qnet(Q, qmodel_path.replace('.gz',''), gz=True)
    else:
        Q=load_qnet(qmodel_path)


sp=pd.read_csv(POLEFILE, index_col=0).T
T=10000
if VAR:
    
    for k in vdict.keys():
        NULL={k:np.array(['']*len(Q[k].feature_names)).astype('U100') for k in vdict.keys()} 
        sp_={k:pd.concat([pd.DataFrame(columns=Q[k].feature_names),
                       sp])[Q[k].feature_names].fillna('').values.astype(str) 
             for k in vdict.keys()} 
        
        D=pd.DataFrame({m:
                        (qdistance(qsample(sp_[k][0],Q[k],steps=m),NULL[k],Q[k],Q[k]),
                         qdistance(qsample(sp_[k][1],Q[k],steps=m),NULL[k],Q[k],Q[k]),
                         qdistance(qsample(sp_[k][0],Q[k],steps=m),
                                   qsample(sp_[k][1],Q[k],steps=m),Q[k],Q[k]))
                        for m in tqdm(np.arange(1,T,100))})
        D.to_csv(f'{OUTDIR}/relaxation_{year}{k}.csv')
        tau,cov=getTau(D.T.rename(columns={0:'R',1:'L',2:'RL'}))
        print(year,k,tau,cov)
        pd.DataFrame([[year, k, tau, cov]], columns=['year', 'k', 'tau', 'cov']).to_csv(f'{OUTDIR}/tau_{year}{k}.csv')
    
else:
    sp_=pd.concat([pd.DataFrame(columns=feature_names),
               sp])[feature_names].fillna('').values.astype(str)

    NULL=np.array(['']*len(Q.feature_names)).astype('U100')
    D=pd.DataFrame({m:
                    (qdistance(qsample(sp_[0],Q,steps=m),NULL,Q,Q),
                     qdistance(qsample(sp_[1],Q,steps=m),NULL,Q,Q),
                     qdistance(qsample(sp_[0],Q,steps=m),
                               qsample(sp_[1],Q,steps=m),Q,Q))
                    for m in tqdm(np.arange(1,T,100))})
    D.to_csv(f'{OUTDIR}/relaxation_{year}.csv')
    tau,cov=getTau(D.T.rename(columns={0:'R',1:'L',2:'RL'}))

    print(year,tau,cov)
    pd.DataFrame([[year, tau, cov]], columns=['year',  'tau', 'cov']).to_csv(f'{OUTDIR}/tau_{year}.csv')






