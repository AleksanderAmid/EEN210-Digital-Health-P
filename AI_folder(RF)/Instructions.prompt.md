Recorded Data is in the format : 
[text](../Model_Test_DATA.csv)

When Training the model there will be labeled. And the label connections is 5 = Fall.

We want to use Random Forest (#Ai_RandomForst.py) that takes in a CSV file via the fileexplorer (tkinder), and trains a model that can predict a fall. It should always save the model .joblib in the same map as the model.

We want a #Evaluation.py that takes in and loads the model.joblib (Via the fileexplorer), and then shows some evaulations (Accuracy, Confusion Matrix, Precision, Recall, F1-Score)
