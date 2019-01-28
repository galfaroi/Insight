from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import *
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
%matplotlib inline
from sklearn.neural_network import MLPRegressor
import plotly.graph_objs as go
import numpy as np
from torchdiffeq import odeint
from keras import optimizers

PATH = "/home/german/Desktop/tricar/code/stack_13_17.csv"
data_sin2da= pd.read_csv(PATH)
del data_sin2da['id']
data_sin2da = data_sin2da.fillna(0)
sin2da = data_sin2da

del sin2da['malla']
del sin2da['temporada']

sin2da.head()

sin2da.max()
#sin2da = pd.get_dummies(sin2da, prefix=['malla'])
df = sin2da
df.describe()
df.iloc[413]
#normalization
sin2da_norm = (df - df.mean())/df.std()
sin2da_norm.head()
mean_semana  = df['semana'].mean()
std_semana = df['semana'].std()
mean_dia  = df['dia_transplante'].mean()
std_dia = df['dia_transplante'].std()
mean_sem_tra  = df['semana_trans'].mean()
std_sem_tra = df['semana_trans'].std()

mean = df['rendimiento'].mean()
std = df['rendimiento'].std()
mean
std
normalization_data = {'std': mean, 'std': std, 'mean_semana': mean_semana, 'std_semana': std_semana, 'mean_dia': mean_dia, \
                      'std_dia': std_dia, 'mean_sem_tra': mean_sem_tra, 'std_sem_tra': std_sem_tra}
%pwd
import json
with open('/home/german/Desktop/insight_project/norm.json', 'w') as outfile:
    json.dump(normalization_data, outfile)

sin2da_norm.head()

y = sin2da_norm['rendimiento']
del sin2da_norm['rendimiento']

x.columns

x = sin2da_norm
y.shape
x.shape
X_train, X_test, y_train, y_test = train_test_split(x.values, y, test_size=0.2, random_state=42)


#simple linear regression
regr = LinearRegression(normalize=True)
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
X_test[3]

# Fit the Bayesian Ridge Regression and an OLS for comparison
clf = BayesianRidge(compute_score=True, normalize=True)
clf.fit(X_train, y_train)
y_pred_clf = clf.predict(X_test)

#random forest
rforest = RandomForestRegressor(max_depth=7, random_state=0, n_estimators=100)
rforest.fit(X_train, y_train)
y_pred_rforest = rforest.predict(X_test)

#gaussian process using scikit-sklearn
kernel = C(2.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
k1 = 50.0**2 * RBF(length_scale=10.0)
gp_m_9 = GaussianProcessRegressor(kernel=k1, n_restarts_optimizer=12, normalize_y=True)
gp_m_9.fit(X_train, y_train)
y_pred_gp, sigma = gp_m_9.predict(X_test, return_std=True)
#MLPClassifier
nn = MLPRegressor(
    hidden_layer_sizes=(10,),  activation='tanh', solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=2000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, beta_1=0.9,  beta_2=0.999, epsilon=1e-08)
nn.fit(X_train, y_train)
y_predict_nn = nn.predict(X_test)


X_train.shape
import GPy
kernel = GPy.kern.RBF(3)
y_=   y
y_ = np.reshape(y_, (len(y_),1))
X_train_, X_test_, y_train_, y_test_ = train_test_split(x.values, y_, test_size=0.2, random_state=42)
X_train.shape
kg = GPy.kern.RBF(input_dim=3, ARD = True,variance =1, lengthscale=2.)
kb = GPy.kern.Bias(input_dim=3)
kp = GPy.kern.sde_StdPeriodic(input_dim=3)
kk = kg + kb + kp
ker = GPy.kern.Matern52(3, variance=1, ARD=True) + GPy.kern.Bias(input_dim=3) + GPy.kern.sde_StdPeriodic(input_dim=3)
#k = GPy.kern.RBF(input_dim=3, variance=1., lengthscale=1.)
kmlp = GPy.kern.MLP(3) + GPy.kern.Bias(1) + GPy.kern.sde_StdPeriodic(input_dim=3) +GPy.kern.Matern52(3, variance=2, ARD=True)
m = GPy.models.GPRegression(X_train_, y_train_, kmlp)
#m.optimize()
for i in range(10):
    m.optimize('bfgs', max_iters=4000) #first runs EP and then optimizes the kernel parameters
    print('iteration:', i,)
    print(m)
    print("")
display(m)
# 1: Saving a model:
import pickle
pickle.dump(m, open( "modelGPy.p", "wb" ) )
# Model creation, without initialization:
m_= pickle.load( open( "modelGPy.p", "rb" ) )

X_test[0]
y_predict_gpy_ = m_.predict(X_test)
quantiles_ = m_.predict_quantiles(X_test)




X_test.shape
X_test[1]
y_predict_gpy = m.predict(X_test)
quantiles = m.predict_quantiles(X_test)
y_predict_gpy[:5]
conf_down =quantiles[0]
conf_up = quantiles[1]
y_predict_gpy = y_predict_gpy[0]
y_predict_gpy.shape

y_gpy = y_predict_gpy.ravel()
conf_down = conf_down.ravel()
conf_up = conf_up.ravel()

y_gpy.shape
conf_down.shape
conf_up.shape
y_test.shape
#create DataFrame
d = {'y_test': y_test, 'y_pred': y_gpy, 'con_up':conf_up, 'conf_down_n': conf_down }
ys = df = pd.DataFrame(data=d)
ys.head()
resultados = ys*std+mean
X_test.shape
resultados.shape
result = X_test.join(resultados, how='inner')
type(resultados)



plt.figure()
df = resultados
df.to_csv('resultados_gp_pepinos_dec19.csv')
df = df.sort_index()
df_= df[:50]
df_.plot.area(stacked=False, figsize=(18, 8));
df.plot(subplots=True, figsize=(18, 12));

plt.figure(figsize=(20,10))
plt.plot(df_.index,  df_.y_test, df_, 'r')
plt.plot(df_.index, df_.y_pred, df_, 'b')
plt.fill_between(df_.index, df_.conf_down_n, df_.con_up, color='b', alpha=0.2)

#Plotly
trace_high = Scatter(
    x=df_.index,
    y=df_.con_up,
    name = "Estimado_Max",
    line = dict(color = '#17BECF'),
    opacity = 0.8)

trace_low = Scatter(
    x=df_.index,
    y=df_.conf_down_n,
    name = "Estimado_Min",
    line = dict(color = '#17BECF'),
    opacity = 0.8)

trace_yhat = Scatter(
    x = df_.index,
    y=df_.y_pred,
    name = "Promedio",
    line = dict(color = '#7F7F7F'),
    opacity = 0.8)

trace_data = Scatter(
    x= df_.index,
    y = df_.y_test,
    name = "Produccion",
    #line = dict(color = '#7F7F7F'),
    mode = 'markers')
    #opacity = 0.8)

df_
data = [trace_high,trace_low,trace_yhat, trace_data]
iplot(data, filename = 'filling-interior-area')

df_.columns


#iplot([table], filename='table-of-mining-data')
'''
def metrica_pancho(y_prueba, y_predecida):
    resta = y_prueba - y_predecida
    resta = abs(resta)
    return sum(resta)/len(y_prueba)
'''
mean_squared_error(y_test, y_pred)
mean_squared_error(y_test, y_pred_clf)
mean_squared_error(y_test, y_pred_gp)
mean_squared_error(y_test, y_pred_rforest)
mean_squared_error(y_test, y_predict_nn)
#32 sin kernel periodico
mean_squared_error(y_test, y_predict_gpy)
#semanas desde transplante  + mlp 0.28778474564180645
#SIN semanas desde trasplante + mlp 0.2949808818147355
#SIN semanas desde trasnplante + kk 0.2689871738193314
#con semanas desde trasplante + KK 0.2864149916162263
#con semanas desde trasplante + ker 0.28778765989850036
#sin semanas desde trasplante + ker 0.2798044839988994

np.reshape(y_predict_gpy,0)
y_gpy.shape

metrica_pancho(y_test, y_gpy)


y_unnorm = y_test[:20]*std+mean
Y_pred_rf_un = y_predict_gpy[:20]*std+mean
y_unnorm.shape
Y_pred_rf_un.shape

conf_down_n = conf_down*std+mean
conf_down_n[:10]
conf_up_n = conf_up*std + mean
conf_up_n[:10]
Y_pred_rf_un[:10]
y_test[:5]*std+mean

yy=Y_pred_rf_un[0]
yy.shape
con_up_n = conf_up_n.ravel()
conf_down_n = conf_down_n.ravel()
con_up_n.shape




ys.head(10)
ys  = ys.T

sin2da.ix[231]
ys.T*std+mean


len(y_unnorm)
y_unnorm
Y_pred_rf_un
plt.plot(y_unnorm)
len(Y_pred_rf_un)
plt.plot(Y_pred_rf_un)



type(y_test)
metrica_pancho(y_test,Y_pred_rf_un)
