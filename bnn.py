import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from pyro.infer import EmpiricalMarginal, SVI, Trace_ELBO, TracePredictive
from functools import partial
from RegressionModel import RegressionModel

%pwd

import torch
import torch.nn as nn
from torch.nn.functional import normalize  # noqa: F401

import pyro
from pyro.distributions import Bernoulli, Normal  # noqa: F401
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.optim import Adam
from pyro.distributions import Normal, Uniform, Delta
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')


df = pd.read_csv('lstm_ts.csv')

df.head()
del df['Unnamed: 0']
del df['malla']
#normalization
sin2da_norm = (df - df.mean())/df.std()
#no normalization
#sin2da_norm = df
df_mean = df.mean
df_std = df.std
sin2da_norm = sin2da_norm.fillna(0)
x = sin2da_norm[['transplant_day', 'transplant_week']]
del sin2da_norm['transplant_day']
del sin2da_norm['transplant_week']
#y=np.log(y)
y = sin2da_norm
y.shape
x.shape


# Create random Tensors to hold inputs and outputs
X_train, X_test, y_train, y_test = train_test_split(x.values, y, test_size=0.2, random_state=42)
X_train.shape
X_train[:,1]
y_train.shape
X_train[:5]
y_test.shape

#convert to tensor
x_data = torch.tensor(X_train, dtype=torch.float)
y_data = torch.tensor(y_train.values, dtype=torch.float)
x_data.shape
y_data.shape

x_data_ = torch.tensor(x.values, dtype=torch.float)
y_data_ = torch.tensor(y.values, dtype=torch.float)
x_test_t = torch.tensor(X_test, dtype=torch.float)
y_test_t = torch.tensor(y_test.values, dtype=torch.float)
# NN with one linear layer
class RegressionModel(nn.Module):
    def __init__(self, p):
        super(RegressionModel, self).__init__()
        self.linear = nn.Linear(p, 36)

    def forward(self, x):
        # x * w + b
        x.view(x.size(0), -1)
        return self.linear(x)


model = torch.nn.Sequential(
    torch.nn.Linear(p, 6),
    torch.nn.ReLU(),
    torch.nn.Linear(6, 1))

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, 1)   # output layer
    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

X_train.shape
N = 1236
p = 3

softplus = nn.Softplus()
regression_model = model #Net(p,p)
regression_model = RegressionModel(2)


loss_fn = torch.nn.MSELoss()#MSELoss(reduction='sum')
optim = torch.optim.SGD(regression_model.parameters(), lr=0.01)#torch.optim.Adam(regression_model.parameters(), lr=0.1)

num_iterations = 200000
for j in range(num_iterations):
        # run the model forward on the data
        y_pred = regression_model(x_data).squeeze(-1)
        # calculate the mse loss
        loss = loss_fn(y_pred, y_data)
        # initialize gradients to zero
        optim.zero_grad()
        # backpropagate
        loss.backward()
        # take a gradient step
        optim.step()
        if (j + 1) % 50 == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss.item()))
    # Inspect learned parameters
print("Learned parameters:")
for name, param in regression_model.named_parameters():
    print(name, param.data.numpy())
x_data.size()
x_data.squeeze(-1).size(-1)
#me quede ajustando los vectores de los priors 
def model(x_data, y_data):
    # weight and bias priors
    w_prior = Normal(torch.zeros(1, 3), torch.ones(1, 3)).to_event(1)
    b_prior = Normal(torch.tensor([[8.]]), torch.tensor([[1000.]])).to_event(1)
    f_prior = Normal(0., 1.)
    priors = {'linear.weight': w_prior, 'linear.bias': b_prior, 'factor': f_prior}
    scale = pyro.sample("sigma", Uniform(0., 10.))
    # lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module("module", regression_model, priors)
    # sample a nn (which also samples w and b)
    lifted_reg_model = lifted_module()
    with pyro.plate("map", len(x_data)):
        # run the nn forward on data
        prediction_mean = lifted_reg_model(x_data).squeeze(-1)
        # condition on the observed data
        pyro.sample("obs",
                    Normal(prediction_mean, scale),
                    obs=y_data)
        return prediction_mean


from pyro.contrib.autoguide import AutoDiagonalNormal
guide = AutoDiagonalNormal(model)
optim = Adam({"lr": 0.03})
svi = SVI(model, guide, optim, loss=Trace_ELBO(), num_samples=2000)
type(svi)

type(guide)
def train():
    pyro.clear_param_store()
    for j in range(num_iterations):
        # calculate the loss and take a gradient step
        loss = svi.step(x_data, y_data)
        if j % 100 == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(x_data)))

train()

#evaluation
for name, value in pyro.get_param_store().items():
    print(name, pyro.param(name))
get_marginal = lambda traces, sites:EmpiricalMarginal(traces, sites)._get_samples_and_weights()[0].detach().cpu().numpy()

def summary(traces, sites):
    marginal = get_marginal(traces, sites)
    site_stats = {}
    for i in range(marginal.shape[1]):
        site_name = sites[i]
        marginal_site = pd.DataFrame(marginal[:, i]).transpose()
        describe = partial(pd.Series.describe, percentiles=[.05, 0.25, 0.5, 0.75, 0.95])
        site_stats[site_name] = marginal_site.apply(describe, axis=1) \
            [["mean", "std", "5%", "25%", "50%", "75%", "95%"]]
    return site_stats

def wrapped_model(x_data, y_data):
    pyro.sample("prediction", Delta(model(x_data, y_data)))

posterior = svi.run(x_test, y_test)
print(posterior)
trace_pred = TracePredictive(wrapped_model,
                             posterior,
                             num_samples=1000)
post_pred = trace_pred.run(x_test, y_test)
post_summary = summary(post_pred, sites= ['prediction', 'obs'])
mu = post_summary["prediction"]
y = post_summary["obs"]
len(y)
mu[:5]
y_test
mu.head()
y.head()
preds = []
for i in range(100):
    sampled_reg_model = guide(x_test)
    pred = sampled_reg_model(x_test).data.numpy().flatten()
    preds.append(pred)




x_data[:, 0]
predictions = pd.DataFrame({
    "week": x_data[:, 1],
    "dia_trans": x_data[:, 2],
    "mu_mean": mu["mean"],
    "mu_perc_5": mu["5%"],
    "mu_perc_95": mu["95%"],
    "y_mean": y["mean"],
    "y_perc_5": y["5%"],
    "y_perc_95": y["95%"],
    "true_rendimiento": y_data,})

predictions.head()

pred = predictions*std+mean
pred.head()
sem_mean = df['semana'].mean()
sem_std =  df['semana'].std()

dia_mean = df['dia_transplante'].mean()
dia_std = df['dia_transplante'].std()

predictions['dia_trans'] = predictions['dia_trans']*dia_std + dia_mean
predictions['week']=predictions['week']*sem_std+sem_mean
predictions[['mu_mean', 'mu_perc_5','mu_perc_95', 'true_rendimiento', 'y_mean', 'y_perc_5', 'y_perc_95']] = \
   predictions[['mu_mean', 'mu_perc_5','mu_perc_95', 'true_rendimiento', 'y_mean', 'y_perc_5', 'y_perc_95']] *mean + mean
predictions.head(20)
predictions[['dia_trans', 'week']].head()
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
#pyro.get_param_store
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
pred_ = predictions
pred_ = pred_.sort_values(by=["week"])
fig.suptitle("Regression line 90% CI", fontsize=16)
ax[0].plot(pred_["week"],
           pred_["mu_mean"])
ax[0].fill_between(pred_["week"],
                   pred_["mu_perc_5"],
                   pred_["mu_perc_95"],
                   alpha=0.5)
ax[0].plot(pred_["week"],
           pred_["true_rendimiento"],
           "o")
ax[0].set(xlabel="weeks",
          ylabel="Yield",
          title="Yields Greenhouses")
ax[1].plot(pred_["week"],
           pred_["mu_mean"])
ax[1].fill_between(pred_["week"],
                   pred_["y_perc_5"],
                   pred_["y_perc_95"],
                   alpha=0.5)
ax[1].plot(pred_["week"],
           pred_["true_rendimiento"],
           "o")
ax[1].set(xlabel="weeks",
          ylabel="Yield",
          title="Yields Greenhouses")
