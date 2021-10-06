# Technical Assessment


# Assumptions for acquire:
- Assumed procedure_id is auto incremented: created two functions one to obtain dictionary of attributes with keys who ranged from 0 to 1000, a second function to obtain dictionary of procedures with keys who ranged from 0 to 1000.
- After combining the dictionary of attributes and dictionary of procedures together they were converted to a dataframe. This will make exploration and modeling the data easier.

# Assumptions for preparation:
- Due to time constraints will assume there are not null values in the fixed set of attributes.
- Will assume the only possible nulls are in procedure_id column.
- Created a function to drop nulls in procedure_id column. It will be necessary to drop any null values that are in the procedure_id column because this step is necessary to be able to feed the data into the model and this column is our target thus there must be values in it.


# Choosing the model:
This is a binary classification because it is a classification with two possible outcomes (success or failure of hospital procedures).
- The classification algorithm used was a Random Forest Model for several reasons listed below: 
     - It is a robust model that is able to handle outliers and data that is not normally distributed.
     - This model is works well with a wide variety of data.
     - Other models such as the logistic regression model are not as versatile to a variety of datasets because it assumes linear relationships which may not be the case for this dataset.
     - This Random Forest model has a reduction in over-fitting compare to the Decision Tree model.


# Files:
- train_model.py: This .py file contains functions for acquire, prepare, and training the model.
- success_prediction.py: This .py file contains a function that returns a prediction of success (True or False).


    