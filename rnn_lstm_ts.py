import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pyro
from pyro.distributions import Bernoulli, Normal  # noqa: F401
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.optim import Adam
from pyro.distributions import Normal, Uniform, Delta
import torch
import torch.nn as nn
from torch.autograd import Variable

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

# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),)
loss_fn = torch.nn.MSELoss()

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(10000):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x_data)

    # Compute and print loss.
    loss = loss_fn(y_pred, y_data)
    print(t, loss.item())

    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
y_predict_ = model(x_test_t)
mean_squared_error(y_predict_.detach().cpu().numpy(), y_test_t.detach().cpu().numpy())

y_predict_.detach().cpu().numpy()
n_feature
#Bayesian neural network feb7
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, 36)   # output layer
        #self.relu(n_hidden)
    def forward(self, x):
        x = torch.tanh(self.hidden(x))
        x = self.predict(x)
        return x


  # number of features


X_train.shape
x_data = X_train
y_pred.shape
y_data.shape
x_data = torch.tensor(X_train, dtype=torch.float)

#N, D_in, H, D_out = 1, 2, 72, 36
first_layer= x_data.shape[1]
first_layer

second_layer = 36 # first_layer * 6
regression_model = Net(first_layer, second_layer)

loss_fn = torch.nn.MSELoss()#MSELoss(reduction='sum')
#optim = torch.optim.SGD(regression_model.parameters(), lr=0.01)#torch.optim.Adam(regression_model.parameters(), lr=0.1)
learning_rate = 1e-4
optim = torch.optim.Adam(regression_model.parameters(), lr=learning_rate)
num_iterations = 2000
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
x_data_.size()
x_data.squeeze(-1).size(-1)
y_predict_reg = regression_model(x_test_t)
mean_squared_error(y_predict_reg.detach().cpu().numpy(), y_test_t.detach().cpu().numpy())
regression_model
#Bayesian Model
def model(x_data_, y_data_):

    # weight and bias priors
    mu = Variable(torch.zeros(second_layer, first_layer)).type_as(x_data_)
    sigma = Variable(torch.ones(second_layer, first_layer)).type_as(x_data_)
    bias_mu = Variable(torch.zeros(second_layer)).type_as(x_data_)
    bias_sigma = Variable(torch.ones(second_layer)).type_as(x_data_)
    w_prior, b_prior = Normal(mu, sigma), Normal(bias_mu, bias_sigma)

    mu2 = Variable(torch.zeros(36, second_layer)).type_as(x_data_)
    sigma2 = Variable(torch.ones(36, second_layer)).type_as(x_data_)
    bias_mu2 = Variable(torch.zeros(36)).type_as(x_data_)
    bias_sigma2 = Variable(torch.ones(36)).type_as(x_data_)
    w_prior2, b_prior2 = Normal(mu2, sigma2), Normal(bias_mu2, bias_sigma2)

    priors = {'hidden.weight': w_prior,
              'hidden.bias': b_prior,
              'predict.weight': w_prior2,
              'predict.bias': b_prior2}
    scale = Variable(torch.ones(x_data_.size(0))).type_as(x_data_)
    # lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module("module", regression_model, priors)
    # sample a nn (which also samples w and b)
    lifted_reg_model = lifted_module()
    with pyro.plate("map",len(x_data_)):
        print('tensor_1')
        print(x_data_.shape)
        # run the nn forward on data
        prediction_mean = lifted_reg_model(x_data_).squeeze(-1)
        print('tensor_2')
        print(x_data_.shape)
        # condition on the observed data
        pyro.sample("obs",
                    Normal(prediction_mean, scale),
                    obs=y_data_)
        return prediction_mean


from pyro.contrib.autoguide import AutoDiagonalNormal
guide = AutoDiagonalNormal(model)
optim = Adam({"lr": 0.01})
svi = SVI(model, guide, optim, loss=Trace_ELBO(), num_samples=50000)
type(svi)

type(guide)
def train():
    pyro.clear_param_store()
    for j in range(num_iterations):
        # calculate the loss and take a gradient step
        loss = svi.step(x_data_, y_data_)
        if j % 100 == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(x_data_)))
x_data.shape
y_data.shape
train()

y_preds = Variable(torch.zeros(30, 1))
sampled_reg_model = guide(x_test_t)
type(sampled_reg_model)
sampled_reg_model.keys()

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

posterior = svi.run(x_test_t, y_test_t)

# posterior predictive distribution we can get samples from
trace_pred = TracePredictive(wrapped_model,
                             posterior,
                             num_samples=10000)
post_pred = trace_pred.run(x_test_t, None)

post_summary = summary(post_pred, sites= ['prediction', 'obs'])
mu = post_summary["prediction"]

y = post_summary["obs"]
y.head(10)
mu_ = mu*std+mean
mu_['mean'][:10]
y_test[:10]*std+mean
X_train[:12]

y_train_u = y_train*std+mean
y_train_u.head(5)
mu_ = mu['mean']*std+mean
mu_[:5]
_y_*std+mean

y_test_ = y_test*std+mean
y_test_[:10]

mean_squared_error(y_test, mu['mean'])

#697735.1645495887
predictions = pd.DataFrame({
    "cont_": x_data[:, 0],
    "rugged": x_data[:, 1],
    "mu_mean": mu["mean"],
    "mu_perc_5": mu["5%"],
    "mu_perc_95": mu["95%"],
    "y_mean": y["mean"],
    "y_perc_5": y["5%"],
    "y_perc_95": y["95%"],
    "true_gdp": y_data,
})

for i in range(100):
    sampled_reg_model = guide(x_test)
    # run the regression model and add prediction to total
    y_preds = y_preds + sampled_reg_model(x_test)
# take the average of the predictions
y_preds = y_preds / 100
