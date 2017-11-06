# CreditCardFraud
Using Basic Neural Net Framework to Predict the Nature of Credit Card Transactions.

The imported datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced. Due to confidentiality issues, the original features of the data are not provided.

If the values of features differ in magnitude by a large amount, it may take a large amount of time for a learning algorithm to converge, if it even converges at all. Thus, the first step was to scale the features, namely the Amount feature.

Next, because the data was so imbalanced (fraud accounts for 0.172% of transactions), I resampled the data. I used undersampling. In the future, I may use SMOTE (Synthetic Minority OverSampling Technique), as that is the convention. 

Once the data is resampled, I split the data into training and test data. I used a 70%-30% train test split. In the future, I may do a 60%-40% train test split, where the 40% is also split 20-20 in cross validation and test. I did not use cross validation in this example.

Now I set up the Neural Net. After splitting the data, I created an instance of the model. I used MLP classifier (Multi Layered Perceptron) to fit the data, and then run predictions. I used three hidden layers with the # of neurons equivalent to the number of features of the data. It is worth altering hidden layer sizes, the solver, and other parameters of MLPClassifier to see which combination returns the best results for your dataset. I returned a confusion matrix of the results as well as a classification report of three instances (1: Fit and Run on Undersampled Data 2: Fit and Run on Skewed Data 3: Fit to UnderSample and Run on Whole Data)




Imported Data from Kaggle Credit Card Fraud Detection Kernel. 
Credit for the data Goes to: Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. Calibrating Probability with Undersampling for Unbalanced Classification. In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015 (https://www3.nd.edu/~rjohns15/content/papers/ssci2015_calibrating.pdf)

The dataset has been collected and analysed during a research collaboration of Worldline and the Machine Learning Group (http://mlg.ulb.ac.be/)
