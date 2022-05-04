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


##Mann*Whitney Tipo
st.mannwhitneyu(df_avoP_M[df_avoP_M['Type']=='Organic']['Mean Price'],df_avoP_M[df_avoP_M['Type']=='Conventional']['Mean Price'],alternative = 'greater')

##Variedad
st.mannwhitneyu(df_avoP[df_avoP['Variety']=='HASS']['Mean Price'],df_avoP[df_avoP['Variety']!='HASS']['Mean Price'],alternative='greater')

##Tamaño
kruskal = st.kruskal(df_avoP_M_conv[df_avoP_M_conv['Item Size']=='32s']['Mean Price'],df_avoP_M_conv[df_avoP_M_conv['Item Size']=='36s']['Mean Price'],df_avoP_M_conv[df_avoP_M_conv['Item Size']=='40s']['Mean Price'],
                     df_avoP_M_conv[df_avoP_M_conv['Item Size']=='48s']['Mean Price'],df_avoP_M_conv[df_avoP_M_conv['Item Size']=='60s']['Mean Price'],df_avoP_M_conv[df_avoP_M_conv['Item Size']=='70s']['Mean Price'],
                     df_avoP_M_conv[df_avoP_M_conv['Item Size']=='84s']['Mean Price'])
dunn = posthoc_dunn(df_avoP_M_conv,'Mean Price','Item Size',p_adjust='bonferroni')

#lugar de origen
kruskal = st.kruskal(df_avoP[df_avoP['City Name']=='MEXICO CROSSINGS THROUGH TEXAS']['Mean Price'],df_avoP[df_avoP['City Name']=='CHILE IMPORTS - PORT OF ENTRY LOS ANGELES AREA']['Mean Price'],df_avoP[df_avoP['City Name']=='SOUTH DISTRICT CALIFORNIA']['Mean Price'],
                     df_avoP[df_avoP['City Name']=='PERU IMPORTS - PORTS OF ENTRY PHILADELPHIA AREA AND NEW YORK CITY AREA']['Mean Price'],df_avoP[df_avoP['City Name']=='CARIBBEAN IMPORTS - PORTS OF ENTRY SOUTH FLORIDA']['Mean Price'])
dunn = posthoc_dunn(df_avoP,'Mean Price','City Name',p_adjust='bonferroni')

#Mes del Año
dunn = posthoc_dunn(df_avoP,'Mean Price','Mes Dummy',p_adjust='bonferroni')
