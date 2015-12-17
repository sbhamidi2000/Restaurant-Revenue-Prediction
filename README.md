# Restaurant-Revenue-Prediction
Predict annual restaurant sales based on objective measurements
Site: https://www.kaggle.com/c/restaurant-revenue-prediction

#Dataset for building the predictors
TFI has provided a dataset with 137 restaurants in the training set, and a test set of 100000 restaurants. The data columns include the open date, location, city type, and three categories of obfuscated data: Demographic data, Real estate data, and Commercial data. The revenue column indicates a (transformed) revenue of the restaurant in a given year and is the target of predictive analysis. 

File descriptions

train.csv - the training set. Use this dataset for training your model. 
test.csv - the test set. To deter manual "guess" predictions, Kaggle has supplemented the test set with additional "ignored" data. These are not counted in the scoring.
sampleSubmission.csv - a sample submission file in the correct format

More Info: https://www.kaggle.com/c/restaurant-revenue-prediction/data

Evaluation

Root Mean Squared Error (RMSE)

Submissions are scored on the root mean squared error. RMSE is very common and is a suitable general-purpose error metric. Compared to the Mean Absolute Error, RMSE punishes large errors:

RMSE=1n∑i=1n(yi−y^i)2−−−−−−−−−−−−√,
where y hat is the predicted value and y is the original value.

Submission File
For every restaurant in the dataset, submission files should contain two columns: Id and Prediction. 
The file should contain a header and have the following format:

Id,Prediction
0,1.0
1,1.0
2,1.0
etc.

# Approach to prediction

Arrive at an ensemble predictor that gives the least RMSE. This is an iterative approach where each predictor in the ensemble is optimized for the least RMSE for combining into an ensemble. Several alternative approaches including NN and SVD were discarded for suboptimal performance. Final choice was a combination of Linear predictor with first order polynomials for rigidity, bias over variance. Random forest performed well over NN and SVD on this data in combination with Boruta feature selector.

# Result Rank: 59/2527 (Top10%)

