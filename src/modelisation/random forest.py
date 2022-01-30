#  on affiche tous les paramètres qui peuvent être optimisée

rf = RandomForestClassifier(random_state = 0)
from pprint import pprint
print('Parameters currently in use:\n')
pprint(rf.get_params())


# Un random search permet de se faire une première idée des valeurs optimales des différents hyper-paramètres, en balayant de façon très large 
# les différentes possibilités et en selectionnant les meilleures combinaison par validation croisée.

# On affiche la grille à tester.

# nombre d'arbres
n_estimators = [500, 1000, 2000, 3000, 4000, 5000]
# profondeur max de l'arbre
max_depth = [20]
max_depth.append(None)
# nombre d'échantillon min nécessaire par noeuds
min_samples_split = [2, 4]#[2]
# nombre d'échantillon min nécessaire par feuilles
min_samples_leaf = [1, 2]#[1]

# création de la grille
random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
              }
pprint(random_grid)

# création du modèle
rf = RandomForestClassifier(random_state = 0, max_features = 'sqrt', bootstrap = True)

# random search
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=2, random_state=0, n_jobs = -1)

# fit le modèle
rf_random.fit(train_features, train_labels)

pd_res = pd.concat([pd.DataFrame(rf_random.cv_results_["params"]),pd.DataFrame(rf_random.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
pd_res = pd_res.sort_values('Accuracy', ascending=False)
print(rf_random.best_params_)
pd_res.head(5)


# On retient la combinaison gagnante

# Le nombre d’échantillon minimal requis par nœuds et par feuilles ayant été déterminé par random search, on approfondit nos recherches sur le nombre
# et la profondeur des arbres par l’intermédiaire d’un grid search (qui est une autre méthode d’optimisation).

# création du modèle
rf = RandomForestClassifier(random_state = 0, bootstrap=True)

# grid search
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(X_train, Y_train)

pd_res = pd.concat([pd.DataFrame(grid_search.cv_results_["params"]),pd.DataFrame(grid_search.cv_results_["mean_test_score"], columns=["Accuracy"])],
                   axis=1)
pd_res = pd_res.sort_values('Accuracy', ascending=False)
pd_res.head(5)

# La meilleure option ici semble être de choisir ... arbres d’une profondeur maximale de ... niveaux.


# On entraîne maintenant notre modèle avec les valeurs d’hyperparamètres que l’on a trouvé.

# création du modèle
rf = RandomForestClassifier(n_estimators=4000, max_features='sqrt', max_depth=20, min_samples_split=2, min_samples_leaf=1, bootstrap=True, 
                            criterion='gini' ,random_state=0)

# fit le modèle
rf.fit(X_train, Y_train)

# prédictions
predictions = rf.predict(X_test)

# Zero_one_loss error
errors = zero_one_loss(Y_test, predictions, normalize=False)
print('zero_one_loss error :', errors)

# Accuracy Score
accuracy_test = accuracy_score(Y_test, predictions)
print('accuracy_score on test dataset :', accuracy_test)

print(classification_report(predictions, Y_test))

