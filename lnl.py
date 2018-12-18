
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from numpy.linalg import inv
from sklearn.decomposition import PCA

class LinearRegression:
    def fit(self, X, y):
        self.coef_ = inv(X.T.dot(X)).dot(X.T).dot(y)
class PcaRegression:
    def fit(self, X, y):
        pca = PCA(n_components=2)
        pca_df = pd.DataFrame(pca.fit(df[g_expls]).transform(df[g_expls]), columns=g_expls)
        model = LinearRegression()
        model.fit(pca_df[g_expls], df['y'])
        self.coef_ = np.dot(model.coef_, pca.components_)


# In[3]:


def rmse(predicted, target):
    return np.sqrt(np.mean((predicted-target)**2))


# In[81]:


def generate_correlated_data(range1, range2, range3, corr, nb_points):
    xx = np.array(range1)
    yy = np.array(range2)
    zz = np.array(range3)
    means = [xx.mean(), yy.mean(), zz.mean()]  
    stds = [xx.std() / 4, yy.std() / 4, zz.std() / 4]
    covs = [[stds[0]**2, stds[0]*stds[1]*corr, 0], 
            [stds[0]*stds[1]*corr, stds[1]**2, 0], 
            [0, 0, stds[2]**2]] 
    m = np.random.multivariate_normal(means, covs, nb_points)
    df = pd.DataFrame(m, columns=['x1', 'x2', 'x3'])
    df['y'] = df['x1'] + 5 * df['x3']
    df['x2'] = corr * 1000 * df['x2'] + (1 - corr) * df['x3']
    df = df.drop('x3', axis=1)
    df=(df-df.mean())/df.std()
    return df


# In[117]:


def slide_test(model, df, size, features):
    serie = []
    for i in range(df.shape[0] - size):
        sub_df = df[i: i + size]
        model.fit(sub_df[features], sub_df['y'])
        coefs = {features[i]: model.coef_[i] for i in range(len(features))}
        predictions = np.dot(sub_df[features], model.coef_)
        coefs['rmse'] = rmse(predictions, sub_df['y'])
        serie.append(coefs)
    return pd.DataFrame(data=serie)


# # Trying to predict

# In[87]:


df = generate_correlated_data([-500, 500], [-200, 200], [-200, 200], 0.9, 300)
df.head(10)


# In[90]:


model = LinearRegression()
model.fit(df[g_expls], df['y'])
print(f'y = f(x1, x2) = {model.coef_[0]} * x1 + {model.coef_[1]} * x2')


# In[93]:


predicted_y = np.dot(df[g_expls], model.coef_)
print('RMSE:', rmse(predicted_y, df['y']))


# In[94]:


coef_df = slide_test(LinearRegression(), df, 100, ['x1', 'x2'])
coef_df[g_expls].plot()


# In[85]:


g_expls = ['x1', 'x2']
df.plot(kind='scatter', x='x1', y='x2')


# In[86]:


df.corr()


# # Model presentation

# ### Running linear regression

# In[95]:


model = LinearRegression()
model.fit(df[g_expls], df['y'])
model.coef_


# In[97]:


df['predicted_y'] = np.dot(df[g_expls], model.coef_)
print('RMSE:', rmse(df['predicted_y'], df['y']))


# ### Running PCA

# In[65]:


import numpy as np
import matplotlib.pyplot as plt

eigenvectors, eigenvalues, V = np.linalg.svd(df[['x1', 'x2']].T, full_matrices=False)
projected_data = np.dot(df[['x1', 'x2']], eigenvectors)
sigma = projected_data.std(axis=0).mean()

fig, ax = plt.subplots(figsize=(8, 20))
ax.scatter(x=df['x1'], y=df['x2'])
for axis in eigenvectors:
    start, end = [0, 0], sigma * axis
    ax.annotate(
        '', xy=end, xycoords='data',
        xytext=start, textcoords='data',
        arrowprops=dict(facecolor='red', width=2.0))
ax.set_aspect('equal')
plt.show()


# In[66]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_df = pd.DataFrame(pca.fit(df[g_expls]).transform(df[g_expls]), columns=g_expls)
pca_df.corr()


# In[67]:


pca_df.plot(kind='scatter', x=g_expls[0], y=g_expls[1])


# In[68]:


pca_df.corr()


# In[69]:


model = LinearRegression()
model.fit(pca_df[g_expls], df['y'])
model.coef_ = np.dot(model.coef_, pca.components_)


# In[118]:


df['predicted_y'] = np.dot(df[g_expls], model.coef_)
print('RMSE:', rmse(df['predicted_y'], df['y']))


# # Building robust models

# In[104]:


df = generate_correlated_data([-500, 500], [-200, 200], [-200, 200], 0.9, 300)


# ### Overtime OLS

# In[105]:


coef_df = slide_test(LinearRegression(), df, 100, ['x1', 'x2'])
coef_df[g_expls].plot()


# In[106]:


coef_df['rmse'].plot()


# ### Overtime PCA

# In[120]:


coef_df = slide_test(PcaRegression(), df, 100, ['x1', 'x2'])
coef_df[['x1', 'x2']].plot()


# In[121]:


coef_df['rmse'].plot()


# # Reducing noise and dimensionality

# In[23]:


def generate_noisy_data():
    serie = dict()
    for i in range(700):
        serie[f'x{i +1}'] = np.random.uniform(-500, 500, 1000)
    df = pd.DataFrame(serie)
    features = ['x1', 'x2', 'x3', 'x7']
    df['y'] = 5 * df['x1'] + 15 * df['x2'] + 30 * df['x3'] - 40 * df['x7']
    for feature in features:
        df[feature] = df[feature] - np.random.uniform(-10, 10, df.shape[0])
    for feature in serie.keys():
        if feature in features:
            continue
        df[feature] = np.random.uniform(-10, 10, df.shape[0])
    df = df - df.mean()
    return df, list(serie.keys())


# In[75]:


df, g_features = generate_noisy_data()
df.shape


# In[25]:


df.plot(kind='scatter', x='x1', y='x5')


# In[26]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(df.loc[:, df.columns != 'y'], df['y'])
predicted_y = model.predict(df.loc[:, df.columns != 'y'])
rmse(predicted_y, df['y'])


# In[27]:


from sklearn.decomposition import PCA

pca = PCA(n_components=df.shape[1] -1)
pca_df = pd.DataFrame(pca.fit(df[g_features]).transform(df[g_features]), columns=g_features)
explained_variance = pd.DataFrame(pca.explained_variance_ratio_.cumsum())
explained_variance.plot()


# In[28]:


explained_variance[:10].plot()


# In[29]:


class PcaRegression:
    def __init__(self, n_components):
        self._n_components = n_components
        
    def fit(self, X, y):
        pca = PCA(n_components=self._n_components)
        pca_df = pd.DataFrame(pca.fit(X).transform(X), columns=[f'pca_{i +1}' for i in range(self._n_components)])
        model = LinearRegression()
        model.fit(pca_df, y)
        self.coef_ = np.dot(model.coef_, pca.components_)
        
    def predict(self, X):
        return np.dot(X, self.coef_)


# In[30]:


model = PcaRegression(3)
model.fit(df[g_features], df['y'])
predicted = model.predict(df[g_features])
rmse(predicted_y, df['y'])


# In[55]:


pca = PCA(n_components=3)
pca_df = pd.DataFrame(pca.fit(df[g_features]).transform(df[g_features]), columns=[f'pca_{i +1}' for i in range(3)])
pca_df['y'] = df['y']
pca_df.head(10)


# In[69]:


pca_df.plot(kind='scatter', x='pca_1', y='y')


# In[73]:


pca_df.plot(kind='scatter', x='pca_2', y='y')


# In[81]:


pca_df.plot(kind='scatter', x='pca_3', y='y')


# In[90]:


from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
pca_df = pd.DataFrame(pca.fit(df[g_features]).transform(df[g_features]), columns=[f'pca_{i +1}' for i in range(2)])
model = LinearRegression()
model.fit(pca_df, df['y'])
coefs = np.dot(model.coef_, pca.components_)
pca_df['y'] = df['y']

X, Y = np.meshgrid(pca_df['pca_1'], pca_df['pca_2'])
zs = np.array([coefs[0] * x + coefs[1] * y for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

fig = plt.figure(figsize=(15, 30))
ax = fig.add_subplot(211, projection='3d')
ax.scatter3D(pca_df['pca_1'], pca_df['pca_2'], pca_df['y'], cmap='Blues')
ax.plot_surface(X, Y, Z)


# # Ridge regression

# In[132]:


df = generate_correlated_data([-500, 500], [-200, 200], [-200, 200], 0.99, 300)
df.head(10)


# In[133]:


df.corr()


# In[134]:


g_expls = ['x1', 'x2']
df.plot(kind='scatter', x=g_expls[0], y=g_expls[1])


# In[154]:


class LinearRegression:
    def fit(self, X, y):
        self.coef_ = inv(X.T.dot(X)).dot(X.T).dot(y)
        
from sklearn.linear_model import Ridge
coef_linear = slide_test(LinearRegression(), df, 100)
coef_ridge = slide_test(Ridge(2), df, 100)
coef_df = pd.concat([coef_linear, coef_ridge], axis=1)
coef_df.columns = ['linear_rmse', 'linear_x1', 'linear_x2', 'ridge_rmse', 'ridge_x1', 'ridge_x2']
coef_df[['linear_x1', 'linear_x2', 'ridge_x1', 'ridge_x2']].plot(figsize=(15, 10))


# In[151]:


coef_df[['linear_rmse', 'ridge_rmse']].plot(figsize=(15, 10))


# In[155]:


coef_df[['linear_rmse', 'ridge_rmse']].describe()


# In[161]:


multiple_ridge = pd.DataFrame()
for alpha in range(10):
    coef_ridge = slide_test(Ridge(alpha), df, 100)
    multiple_ridge[f'ridge_x1_alpha_{alpha}'] = coef_ridge['x1']
    multiple_ridge[f'ridge_x2_alpha_{alpha}'] = coef_ridge['x2']
    
multiple_ridge.plot(figsize=(15, 10))


# # References

# http://statisticsbyjim.com/regression/multicollinearity-in-regression-analysis/

# https://onlinecourses.science.psu.edu/stat857/node/155/
