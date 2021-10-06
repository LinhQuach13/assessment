import numpy as np
import pandas as pd
from scipy import stats

#sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

#visualization
import matplotlib.pyplot as plt
import seaborn as sns


#cms_library python library
from cms_procedures import get_procedure_attributes, get_procedure_success





###### ACQUIRE DATA ########

def get_attribute(a_dict):
    '''
    This function takes in the attribute dictionary iterates through keys that are 0-500. Each new key/value pair is appended to empty dictionary and returns a new dictionary with all the key/values pairs within the range given.
    '''
    #dictionary of attributes from cms_procedures library
    a_dict= get_procedure_attributes(procedure_id= None)
    # initializing range 
    i, j = 0, 501
    # using loop to iterate through all keys
    res = dict()
    for key, val in a_dict.items():
        if int(key) >= i and int(key) <= j:
            res[key] = val
            return res
        
        

def get_procedure(procedure_dict):
    '''
    This function takes in the procedure dictionary iterates through keys that are 0-500. Each new key/value pair is appended to empty dictionary
    and returns a new dictionary with all the key/values pairs within the range given.
    '''
    #dictionary of procedures from cms_procedures library       
    procedure_dict= get_procedure_success(procedure_id)
    # initializing range 
    i, j = 0, 501
    # using loop to iterate through all keys
    proc = dict()
    for key, val in procedure_dict.items():
        if int(key) >= i and int(key) <= j:
            proc[key] = val
            return proc
        


def merge_two_dicts(res, proc):
    '''
    Given two dictionaries, merge them into a new dict as a copy.
    '''
    z = res.copy()
    z.update(proc)
    return z



def convert_to_df(z):
    '''
    This function takes in a dictionary and converts it to a dataframe.
    '''
    #Specify orient='index' to create the DataFrame using dictionary keys as rows:
    df= pd.DataFrame.from_dict(z, orient='index')
    return df





####### PREPARE ########

def drop_duplicates(df):
    '''
    This function drops duplicates by unique identifier procedure_id
    '''
    df= df.drop_duplicates(subset = ['procedure_id'], inplace= True)
    return df


def drop_nulls(df):
    """
    This function drop all rows with NaNs in procedure_id 
    """
    df= df.dropna(subset=['procedure_id'])
    return df
    



    
######## Modeling ##########

def train_validate_test_split(df, target, seed=123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, 
                                            stratify=df[target])
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       stratify=train_validate[target])
    return train, validate, test



def model_split(train, validate, test, target):
    '''
    this function takes in the train, validate and test subsets
    then splits for X (features) and y (target).
    '''
    X_train, y_train = train.drop(columns= [target]), train[target]
    X_validate, y_validate = validate.drop(columns= [target]), validate[target]
    X_test, y_test = test.drop(columns= [target]), test[target]
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test



def rf_model(X_train, y_train, n_est= 100, c= 'gini', max_dp= None, mss= 2, msl= 1, min_weight_fraction_lf= 0.0, max_ft= 'auto', max_leaf_nd= None, min_impurity_dc= 0.0, min_impurity_splt= None, bootstrp= True, oob_score= False, n_jb= None, random_st= None, verbo= 0, warm_st= False, cw= None, ccpAlpha= 0.0, maxSamples= None):
    '''
    This function takes in the X_train, y_train subsets and the parameters as listed below for the Random Forest Model
    then runs it through the Random Forest model object. Accuracy score is outputted here.
    Parameters:
- n_est: int, default=100
    The number of trees in the forest.

       The default value of ``n_estimators`` changed from 10 to 100
       in 0.22.

- c : {"gini", "entropy"}, default="gini"
    The function to measure the quality of a split. Supported criteria are
    "gini" for the Gini impurity and "entropy" for the information gain.
    Note: this parameter is tree-specific.

- max_dp : int, default=None
    The maximum depth of the tree. If None, then nodes are expanded until
    all leaves are pure or until all leaves contain less than
    min_samples_split samples.

- mss : int or float, default=2
    The minimum number of samples required to split an internal node:

    - If int, then consider `min_samples_split` as the minimum number.
    - If float, then `min_samples_split` is a fraction and
      `ceil(min_samples_split * n_samples)` are the minimum
      number of samples for each split.

       Added float values for fractions.

- msl : int or float, default=1
    The minimum number of samples required to be at a leaf node.
    A split point at any depth will only be considered if it leaves at
    least ``min_samples_leaf`` training samples in each of the left and
    right branches.  This may have the effect of smoothing the model,
    especially in regression.

    - If int, then consider `min_samples_leaf` as the minimum number.
    - If float, then `min_samples_leaf` is a fraction and
      `ceil(min_samples_leaf * n_samples)` are the minimum
      number of samples for each node.

    .. versionchanged:: 0.18
       Added float values for fractions.

- min_weight_fraction_lf : float, default=0.0
    The minimum weighted fraction of the sum total of weights (of all
    the input samples) required to be at a leaf node. Samples have
    equal weight when sample_weight is not provided.

- max_ft : {"auto", "sqrt", "log2"}, int or float, default="auto"
    The number of features to consider when looking for the best split:

    - If int, then consider `max_features` features at each split.
    - If float, then `max_features` is a fraction and
      `round(max_features * n_features)` features are considered at each
      split.
    - If "auto", then `max_features=sqrt(n_features)`.
    - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
    - If "log2", then `max_features=log2(n_features)`.
    - If None, then `max_features=n_features`.

    Note: the search for a split does not stop until at least one
    valid partition of the node samples is found, even if it requires to
    effectively inspect more than ``max_features`` features.

- max_leaf_nd : int, default=None
    Grow trees with ``max_leaf_nodes`` in best-first fashion.
    Best nodes are defined as relative reduction in impurity.
    If None then unlimited number of leaf nodes.

- min_impurity_dc : float, default=0.0
    A node will be split if this split induces a decrease of the impurity
    greater than or equal to this value.

    The weighted impurity decrease equation is the following::

        N_t / N * (impurity - N_t_R / N_t * right_impurity
                            - N_t_L / N_t * left_impurity)

    where ``N`` is the total number of samples, ``N_t`` is the number of
    samples at the current node, ``N_t_L`` is the number of samples in the
    left child, and ``N_t_R`` is the number of samples in the right child.

    ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
    if ``sample_weight`` is passed.

    .. versionadded:: 0.19

- min_impurity_splt : float, default=None
    Threshold for early stopping in tree growth. A node will split
    if its impurity is above the threshold, otherwise it is a leaf.

    .. deprecated:: 0.19
       ``min_impurity_split`` has been deprecated in favor of
       ``min_impurity_decrease`` in 0.19. The default value of
       ``min_impurity_split`` has changed from 1e-7 to 0 in 0.23 and it
       will be removed in 1.0 (renaming of 0.25).
       Use ``min_impurity_decrease`` instead.

- bootstrp : bool, default=True
    Whether bootstrap samples are used when building trees. If False, the
    whole dataset is used to build each tree.

- oob_score : bool, default=False
    Whether to use out-of-bag samples to estimate the generalization score.
    Only available if bootstrap=True.

- n_jb : int, default=None
    The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
    :meth:`decision_path` and :meth:`apply` are all parallelized over the
    trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
    context. ``-1`` means using all processors. See :term:`Glossary
    <n_jobs>` for more details.

- random_st: int, RandomState instance or None, default=None
    Controls both the randomness of the bootstrapping of the samples used
    when building trees (if ``bootstrap=True``) and the sampling of the
    features to consider when looking for the best split at each node
    (if ``max_features < n_features``).
    See :term:`Glossary <random_state>` for details.

- verbo : int, default=0
    Controls the verbosity when fitting and predicting.

- warm_st : bool, default=False
    When set to ``True``, reuse the solution of the previous call to fit
    and add more estimators to the ensemble, otherwise, just fit a whole
    new forest. See :term:`the Glossary <warm_start>`.

- cw : {"balanced", "balanced_subsample"}, dict or list of dicts, default=None
    Weights associated with classes in the form ``{class_label: weight}``.
    If not given, all classes are supposed to have weight one. For
    multi-output problems, a list of dicts can be provided in the same
    order as the columns of y.

    Note that for multioutput (including multilabel) weights should be
    defined for each class of every column in its own dict. For example,
    for four-class multilabel classification weights should be
    [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
    [{1:1}, {2:5}, {3:1}, {4:1}].

    The "balanced" mode uses the values of y to automatically adjust
    weights inversely proportional to class frequencies in the input data
    as ``n_samples / (n_classes * np.bincount(y))``

    The "balanced_subsample" mode is the same as "balanced" except that
    weights are computed based on the bootstrap sample for every tree
    grown.

    For multi-output, the weights of each column of y will be multiplied.

    Note that these weights will be multiplied with sample_weight (passed
    through the fit method) if sample_weight is specified.

- ccpAlpha : non-negative float, default=0.0
    Complexity parameter used for Minimal Cost-Complexity Pruning. The
    subtree with the largest cost complexity that is smaller than
    ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
    :ref:`minimal_cost_complexity_pruning` for details.


- maxSamples : int or float, default=None
    If bootstrap is True, the number of samples to draw from X
    to train each base estimator.

    - If None (default), then draw `X.shape[0]` samples.
    - If int, then draw `max_samples` samples.
    - If float, then draw `max_samples * X.shape[0]` samples. Thus,
      `max_samples` should be in the interval `(0, 1)`


    '''
    #Create the random forest object
    rf = RandomForestClassifier(bootstrap= bootstrp, 
                            class_weight= cw, 
                            criterion= c,
                            min_samples_leaf= msl,
                            n_estimators= n_est,
                            max_depth= max_dp, 
                            random_state= randomst)
    # Fit a model
    rf.fit(X_train, y_train)
    # make predictions
    pred = rf.predict(X_train)
    return (f'Accuracy training score: {rf.score(X_train, y_train):.2%}')
    