# Importación de paquetes
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
from matplotlib.pyplot import figure

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

sns.set(style="darkgrid")

# Leemos el fichero csv con los datos
df = pd.read_csv('c:\\tmp\\TFM\\gmd_02.csv', sep=';')


# Revisar la raza si se agrupan las razas con menos ocurrencias
agrupar_razas = {93 : 93, 85 : 93, 90 : 93, 95 : 93, 94 : 93, 82 : 93, 80 : 80, 96 : 80, 88 : 88, 0 : 0, 23 : 0, 84 : 0, 66 : 0, 18 : 0, 68 : 88, 7 : 7, 89 : 7, 65 : 7, 15 : 15, 97 : 7, 69 : 69, 81 : 81}
df.replace({'ct_raza' : agrupar_razas}, inplace=True)

# Convertimos los tipos
df["ct_integra"] = df["ct_integra"].astype("category")
#df["ct_tipo"] = df["ct_tipo"].astype("category")
df["ct_raza"] = df["ct_raza"].astype("category")
df["ct_fase"] = df["ct_fase"].astype("category")
df['EntradaInicial']= pd.to_datetime(df['EntradaInicial'])
df['EntradaFinal']= pd.to_datetime(df['EntradaFinal'])
df["na_rega"] = df["na_rega"].astype("category")
df["NumBajas"] = df["NumBajas"].astype("int64")
df["gr_codpos"] = df["gr_codpos"].astype("category")
df["gr_poblacion"] = df["gr_poblacion"].astype("category")
df["na_nombre2"] = df["na_nombre2"].astype("category")

# Funcion para convertir en One Hot Encoding
def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return(res)

# Cargamos las variables objetivo y las usadas
y = df['GMD']
x0 = df[['ct_integra','ct_tipo', 'ct_raza', 'IncPeso', 'DiasMedios', 'NumAnimales', 
         'na_rega', 'PesoEntMedio', 'PesoRecMedio', 'NumBajas', 'GPS_Longitud', 'GPS_Latitud', 
         'semanaEntrada', 'añoEntrada', 'PorcHembras', 'PiensoCerdaDia']]
features_to_encode = ['ct_raza']   # , 'na_rega']
x1 = x0.copy()
x1.drop(['ct_integra','na_rega'], inplace=True, axis=1)
for feature in features_to_encode:
    x1 = encode_and_bind(x1, feature)

# attributes = ['IncPeso', 'DiasMedios', 'GMD', 'NumAnimales', 'PesoEntMedio', 'PesoRecMedio', 'NumBajas', 'semanaEntrada', 'añoEntrada', 'PorcHembras', 'PiensoCerdaDia']
# scatter_matrix(x1[attributes])
# plt.show()

# División de los datos en train y test
# ==============================================================================
X_train, X_test, y_train, y_test = train_test_split(x1, y, test_size = 0.2, random_state = 123)

## Vemos de escalar las variables para que no se vean influenciadas por la escala.
scaler = RobustScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# Creación del modelo
# ==============================================================================
rf = RandomForestRegressor(
            n_estimators = 100,
            criterion    = 'squared_error',
            max_depth    = None,
            max_features = 'sqrt',
            oob_score    = False,
            n_jobs       = -1,
            random_state = 123
         )
rf.fit(X_train_s, y_train)

# Graficar diferencias entre valor predicho y real en datos de test
def graficoDiferencias(modelo, X_test_s, y_test):
    y_pred = modelo.predict(X_test_s)
    diferencia = abs(y_pred - y_test)
    g = sns.jointplot(x=y_test, y=y_pred)
    # Draw a line of x=y
    x0, x1 = g.ax_joint.get_xlim()
    y0, y1 = g.ax_joint.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    g.ax_joint.plot(lims, lims, '-r')
    g.ax_joint.scatter(x=y_test, y=y_pred, c=diferencia.values, cmap=sns.dark_palette("#69d", reverse=True, as_cmap=True))
    plt.show()

# Graficar las diferencias
print('Score R2:',rf.score(X_test_s, y_test))
# graficoDiferencias(rf, X_test_s, y_test)

# Mostrar las variables más importantes
important_features_dict = {}
for idx, val in enumerate(rf.feature_importances_):
    important_features_dict[idx] = val

important_features_list = sorted(important_features_dict, key=important_features_dict.get, reverse=True)
print(f'Las 10 características más relevantes pera la regresión son:') # {x1.columns[important_features_list[:10]]}')
print('\tOrden\tCaracterística\tImportancia')
for i in range(10):
    print('\t', i+1, '\t', x1.columns[important_features_list[i]], '\t', important_features_dict.get(i))


# Hago una optimización de los hiperparámetros para RandomForest
from sklearn.model_selection import RandomizedSearchCV
random_grid = {'bootstrap': [True, False],
               'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],
               'max_features': [1.0, 'sqrt', 2, 5, 10, 20],
               'min_samples_leaf': [1, 2, 4, 10],
               'min_samples_split': [2, 5, 10, 20],
               'n_estimators': [20, 50, 75, 100, 150, 250, 500]}
rf_random = RandomizedSearchCV(scoring="neg_mean_squared_error", estimator = rf, param_distributions = random_grid, n_iter = 300, cv = 3, verbose=1, random_state=123, n_jobs = -1)
rf_random.fit(X_train_s, y_train)
# rf_random.best_params_
# rf_random.best_score_
# rf_random.best_estimator_
print('Score R2:',rf_random.best_estimator_.score(X_test_s, y_test))
graficoDiferencias(rf_random.best_estimator_, X_test_s, y_test)

# Medimos las diferencias de la predicción según RMSE
from sklearn.metrics import mean_squared_error
y_pred_rf_01 = rf_random.best_estimator_.predict(X_test_s)
mean_squared_error(y_test, y_pred_rf_01)


##import lazypredict
##from lazypredict.Supervised import LazyRegressor
### Borramos el modelo que tarda mucho
##del lazypredict.Supervised.REGRESSORS[29:32]    # PassiveAggressiveRegressor, PoissonRegressor, QuantileRegressor
##reg = LazyRegressor(verbose=1, ignore_warnings=False, custom_metric=None)
##models, predictions = reg.fit(X_train_s, X_test_s, y_train, y_test)
##
##print(models)
