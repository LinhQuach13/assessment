# Technical Assessment


# Assumptions for acquire:
- Assumed procedure_id is auto-incremented (numeric): created two functions one to obtain dictionary of attributes with keys that ranged from 0 to 500, a second function to obtain dictionary of procedures with keys that ranged from 0 to 500.
- After combining the dictionary of attributes and dictionary of procedures together they were converted to a dataframe. This will make prepping and modeling the data easier.

# Assumptions for preparation:
- Due to time constraints will assume there are no null values in the fixed set of attributes.
- Will assume the only possible nulls are in procedure_id column.
- It is assumed there are likely to be duplicates after acquiring all of the necessary data.
- Created a function named drop_duplicates to remove any duplicate data by its unique identifier procedure_id.
- Created a function to drop nulls in procedure_id. It will be necessary to drop any null values that are in procedure_id because this step is needed to be able to feed the data into the model and this column is our target thus there must be values in it.


# Choosing the model:
This is a binary classification because it is a classification with two possible outcomes (success or failure of hospital procedures).
- The classification algorithm used was a Random Forest Model for several reasons listed below: 
     - It is a robust model that is able to handle outliers and data that is not normally distributed.
     - This model is works well with a wide variety of data.
     - Other models such as the logistic regression model are not as versatile to a variety of datasets because it assumes linear relationships which may not be the case for this dataset.
     - This Random Forest model has a reduction in over-fitting compare to the Decision Tree model.
     - The cons of this model are that it has less interpretability than other models such as the Logistic Regression model.


# Files:
- train_model.py: This .py file contains functions for acquire, prepare, and training the model.
- success_prediction.py: This .py file contains a function that returns a prediction of success (True or False).

## How would the code run:
- Ideally with more time I would have made the .py files as scripts that could be run by calling only the name of the script. As it is right now you can run the functions from each .py file by importing the file (i.e. import train_model) and then calling the the specific function you want to use (i.e. train_model.get_attribute(a_dict)).
- Each .py file would have to be used separately as it is now but with additional time the files would have been created to work together such as making the success_prediction.py file obtain model_split function from the train_model.py. This would made for a more automated process.
- With additional time a function would have been provided that polls the data to confirm whether or not the assumptions made in choosing the model type were True. 

    