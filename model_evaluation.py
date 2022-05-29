import math
from statistics import LinearRegression
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

data_folder = 'C:/Users/Nazanin.Dameshghi/PycharmProjects/FBRA/data'
X = pd.read_csv(f'{data_folder}/features.csv')
y = pd.read_csv(f'{data_folder}/labels.csv')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

clf = GradientBoostingRegressor(n_estimators=20, learning_rate=.1, max_depth=7, random_state=42)
clf.fit(X_train, y_train.values.ravel())
predicts = clf.predict(X_test)
rmse = mean_squared_error(y_test, predicts, squared=False)
print(f"GBR: rmse: {rmse:,.4f}")

lm = LinearRegression()
lm.fit(X_train, y_train.values.ravel())
lm_prd = lm.predict(X_test)
lm_rmse = mean_squared_error(y_test, lm_prd, squared=False)
print(f"LR rmse: {lm_rmse:,.4f}")

new_y_train = y_train.copy()
new_y_train = np.log10(new_y_train[["PMPM_Cost"]])

new_y_test = y_test.copy()
new_y_test = np.log(new_y_test[["PMPM_Cost"]])

new_y_test["log_PMPM_Cost"] = np.log(new_y_test["PMPM_Cost"])
new_y_test.hist(column='log_PMPM_Cost')
sns.distplot(df_set['price'], fit=norm)
fig = plt.figure()

def get_sorted_features(feature_significance, feature_names):
    feats = []
    n = len(feature_significance)
    for idx in range(n - 1):
        feats.append((feature_names[idx], feature_significance[idx]))
    feats.sort(key=lambda x: -x[1])
    return feats


def my_test(features_sorted_list, n_features):
    selected_features = gnf(features_sorted_list, n_features)
    f_df = X_train[selected_features]
    clf = GradientBoostingRegressor(n_estimators=20, learning_rate=.1, max_depth=7, random_state=42)
    clf.fit(f_df, y_train.values.ravel())
    predicts = clf.predict(X_test[selected_features])
    rmse = mean_squared_error(y_test, predicts, squared=False)
    print(f"GBR: rmse: {rmse:,.4f}")

    rf = RandomForestRegressor(n_estimators=10, max_depth=6, random_state=42)
    rf.fit(f_df, y_train.values.ravel())
    rf_prd = rf.predict(X_test[selected_features])
    rf_rmse = mean_squared_error(y_test, rf_prd, squared=False)
    print(f"RF rmse: {rf_rmse:,.4f}")

    lm = LinearRegression()
    lm.fit(f_df, y_train.values.ravel())
    lm_prd = lm.predict(X_test[selected_features])
    lm_rmse = mean_squared_error(y_test, lm_prd, squared=False)
    print(f"LR rmse: {lm_rmse:,.4f}")

#check with the log10 of response

    lm = LinearRegression()
    lm.fit(f_df, new_y_train.values.ravel())
    lm_prd = lm.predict(X_test[selected_features])
    lm_rmse = mean_squared_error(y_test, lm_prd, squared=False)
    print(f"LR_2 rmse: {lm_rmse:,.4f}")

def gnf(tuple_list, n):
    result = []
    for idx in range(n):
        result.append(tuple_list[idx][0])
    return result


features_sorted_list = get_sorted_features(clf.feature_importances_, clf.feature_names_in_)
n_features = len(features_sorted_list)

for n in range(1, 12):
    print(f"Running the experiment with {n} best features")
    my_test(features_sorted_list, n)
    print("\n")




models = [GradientBoostingRegressor(n_estimators=100, learning_rate=.08, max_depth=8, random_state=42),
          LinearRegression(),

          ]


def try_model(model):
    model.fit(X_train, y_train)
    predicts = model.predict(X_test)
    mse = mean_squared_error(y_test, predicts)
    return mse


for model in models:
    mse = try_model(model)
    print(f"Model: {model.__class__}\t MSE: {mse:,.4f}")

lr = LinearRegression()
lr.fit(X_train, y_train)
predicts = lr.predict(X_test)
lr_mse = mean_squared_error(y_test, predicts)
print(f"{lr_mse:,.2f}")
clf.score(X_test, y_test)

params = {
    "n_estimators": 500,
    "max_depth": 4,
    "min_samples_split": 5,
    "learning_rate": 0.01,
    "loss": "squared_error",
}
reg = GradientBoostingRegressor(**params)
reg.fit(X_train, y_train.values.ravel())

mse = mean_squared_error(y_test, reg.predict(X_test))
print("The root mean squared error (MSE) on test set: {:,.4f}".format(mse))

sum = 0
for idx in range(10):  # predicts.shape[0]):
    this_sum = abs(predicts[idx][0] - y_test.iloc[idx]['PMPM_Cost'])
    sum += this_sum
    print(f"{predicts[idx][0]}\t{y_test.iloc[idx]['PMPM_Cost']} = {this_sum} \t\t ---> {sum}")

import matplotlib.pyplot as plt
#
# xgb.plot_tree(xg_reg,num_trees=0)
# plt.rcParams['figure.figsize'] = [50, 10]
# plt.show()
#
# xgb.plot_importance(xg_reg)
# plt.rcParams['figure.figsize'] = [5, 5]
# plt.show()
#
# parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
#               'objective':['reg:linear'],
#               'learning_rate': [.03, 0.05, .07], #so called `eta` value
#               'max_depth': [5, 6, 7],
#               'min_child_weight': [4],
#               'silent': [1],
#               'subsample': [0.7],
#               'colsample_bytree': [0.7],
#               'n_estimators': [500]}
# xgb1 = XGBRegressor()
# xgb_grid = GridSearchCV(xgb1,
#                         parameters,
#                         cv = 2,
#                         n_jobs = 5,
#                         verbose=True)
#
# xgb_grid.fit(X_train,
#          y_train)

# print(xgb_grid.best_score_)
# print(xgb_grid.best_params_)
