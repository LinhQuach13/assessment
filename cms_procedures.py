mport numpy as np
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

import random

def get_procedure_attributes(procedure_id = None):
    '''
    This will take in a procedure_id if one is given and it will be assigned to attributes such as type of procedure, how long it lasted, the severity of the condition being addressed. If a procedure_id is not given it will be randomly generated for the attributes.
    '''
    if procedure_id = None:
        procedure_id= np.random.randit(50000, size= 1)
    else: 
        procedure_id = procedure_id
        
        #create dictionary
        d = {'procedure_id': procedure_id, 'procedure_type': ['debridement', 'hysterectomy'], 'duration': [90.5, 97], 'severity': [1,10]}
        #convert dictionary to dataframe
        df = pd.DataFrame(data=d)
        return df


def get_procedure_success(procedure_id):
    '''
    This will take in the procedure_id and this assumes target success is converted to 1 for Yes and 0 for No.
    '''
    if 1:
        x= True
    else: 
        x= False
    return x
        

def get_procedure_outcomes(procedure_id):
    '''
    This creates a dictionary with procedure_id and attributes such as severity of post procedure complications, pain, recurrence of original condition.
    '''
    #create dictionary
    d = {'procedure_id': procedure_id, 'post_op': [1, 2.5], 'pain': [3, 4.5], 'recurrence': [5, 10]}
    #convert dictionary to dataframe
    df_outcomes = pd.DataFrame(data=d)
    
    return df_outcomes

    
    
    
    

    