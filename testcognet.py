from quasinet.qnet import load_qnet
from quasinet.qnet import qdistance
from quasinet.qsampling import qsample
import pandas as pd
import numpy as np

Q=load_qnet('gss_2016.pkl.gz',gz=True)
POLEFILE='polar_vectors.csv'
sp=pd.read_csv(POLEFILE, index_col=0).T
feature_names=Q.feature_names
sp_=pd.concat([pd.DataFrame(columns=feature_names),
               sp])[feature_names].fillna('').values.astype(str)

s=qsample(sp_[0],Q,steps=2)

print(s)

d=qdistance(sp_[0],sp_[1],Q,Q)

print(d)
