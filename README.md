# personal-loan-project
This project seeks to automate the loan eligibility assessment for a finance company dealing in home loans across diverse geographic regions. To automate this process, we have designed a model to identify which customers are eligible for loan approval. Furthermore, based on the information provided by customers, we also analyze the relationships between independent variables and the target variable.

# Dataset
**loan-train.csv** is used for training the model and testing its accuracy, including all independent variables and the target variable.  
**loan-test.csv** contains all the independent variables, but not the target variable. We will use our model on this dataset to predict whether a customer's loan has been approved.

# Libraries
- Numpy
- Pandas
- Matplotlib.pyplot
- Seaborn
- scipy.stats
- sklearn

# Command
After cloning the repository, you will find four folders:
In the **data_preprocessing** folder, there is only one .ipynb file. Navigate to this folder and enter jupyter notebook in your terminal to open it.
```bash
jupyter notebook
```
Both the **Statistical_Analysis** and **MachineLearning** folders contain a main.py file. Enter python3 main.py in your terminal to run them.
```bash
python3 main.py
```
The **loan_data_set** folder contains all the datasets required for the project, and any images and CSV files generated will also be saved here.

# Output
After you have executed all the commands, you should expect to see three PNG image files and one CSV file named loan-test-predictions in the loan_data_set folder.

# Contributing
If you want to contribute to this project, please fork this repository and create a pull request, or drop me an email at kha112@sfu.ca