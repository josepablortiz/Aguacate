import pandas as pd
import numpy as np
import scipy.stats as st
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.special import inv_boxcox
from scikit_posthocs import posthoc_conover, posthoc_dunn



df_avoP = pd.read_excel('AvocadoP.xlsx')
df_avoM = pd.read_excel('AvocadoM.xlsx')

df_avoP['Date'] = pd.to_datetime(df_avoP['Date'])
df_avoM['Date'] = pd.to_datetime(df_avoM['Date'])

df_avoM = df_avoM.set_index('Date')
df_avoM = df_avoM[df_avoM.index>'2016-01-03']
df_avoP = df_avoP.set_index('Date')
df_avoP['Mean Price'] = (df_avoP['Low Price']+df_avoP['High Price'])/2
df_avoP = df_avoP[df_avoP['Item Size'].isin(['8s','9s','10s','16s','18s','20s','24s',
                                             '32s','36s','40s','48s','60s','70s','84s'] )]
df_avoP['Type'] = df_avoP['Type'].fillna('Conventional')
df_avoP['Mes Dummy'] = df_avoP.index.month
df_avoP = df_avoP.dropna(subset = ['Mean Price'])
df_avoP_M = df_avoP[df_avoP['City Name']=='MEXICO CROSSINGS THROUGH TEXAS']
df_avoP_M_conv = df_avoP_M[df_avoP_M['Type']=='Conventional']
df_avoP_M_conv = df_avoP_M_conv.dropna(subset=['Mean Price'])


#### ARIMA


tamaños = [['32s','36s','40s','48s'],['60s'],['70s'],['84s']]
modelo = [[(3,0,3),(3,0,2),(3,0,3)],[(2,0,3),(2,0,3),(2,0,0)],[(2,0,1),(2,0,1),(2,0,1)],[(2,0,0),(2,0,1),(2,0,0)]]



for i in range(4):
    df_avoP_1= df_avoP_M_conv[df_avoP_M_conv['Item Size'].isin(tamaños[i])]
    for j in range(3):    
        para = modelo[i][j]
        if j == 0:
            precios_semanal = df_avoP_1.resample('W').min()['Low Price'].backfill()
        elif j ==1:
            precios_semanal = df_avoP_1.resample('W').mean()['Mean Price'].backfill()
        elif j ==2:
            precios_semanal = df_avoP_1.resample('W').max()['High Price'].backfill()

        exog = pd.get_dummies(precios_semanal.index.week)
        exog.index = precios_semanal.index
        
        n_train = int(1*precios_semanal.size)
        n_test = precios_semanal.size-n_train
    
        df = pd.DataFrame()
        df.index = precios_semanal.index[:n_train]
    
        precios_semanal_train = precios_semanal[:n_train]
        exog_train = exog[:n_train]
        aux_semana = precios_semanal_train.index[-1].week
        exog_pred = np.eye(53)[aux_semana:aux_semana+4]

        precios_semanal_test = precios_semanal[n_train:]
        exog_test = exog[n_train:]
    
        precios_semanal_train, lam_p = st.boxcox(precios_semanal_train)
        mean_precio_train = precios_semanal_train.mean()
    
        df['Precio'] = precios_semanal_train
        
        mod = SARIMAX(df, order=para,exog = exog,trend='c')
        res = mod.fit(maxiter=500, disp=False)
            
        print(str(tamaños[i]),inv_boxcox(res.forecast(4,exog = exog_pred),lam_p))
        

