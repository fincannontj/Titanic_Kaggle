This code was generated for the getting started Titanic kraggle competition as part of the Dataquest.io kraggle mission course.

The code cleans the feature data filling in Nan results,categorical binning for fare and age data, and the parsing of name data to generate gender columns. The best features are then selected by recursive feature elimination with cross validation.

Using the selected features 3 machine learning algorithms are trained (Logistic Regression, K-nearest neighbors, and Random Forest) selecting the best model variables for training and then fitting a Voting Classifier ensemble model to the output of the three models. 

The result from the best performing model is then saved as Submission 1 and the ensemble results as Submission 2.


