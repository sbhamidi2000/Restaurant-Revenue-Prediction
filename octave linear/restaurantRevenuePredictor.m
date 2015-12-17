% Restaurant Revenue Prediction with Polynomial features

%Regularized Linear Regression and Bias-Variance

%% Initialization
clear ; close all; clc


%% Load Data
data = csvread('trainrun.csv');
eval = csvread('testrun.csv');
eval = eval(2:end,1:38);

data = data(2:end,:);
indices = randperm(length(data));

X = data(indices(1:83), 1:38);
y = data(indices(1:83), 39);



Xval = data(indices(84:109), 1:38);
yval = data(indices(84:109), 39);
Xtest = data(indices(110:135), 1:38);
ytest = data(indices(110:135), 39);

% m = Number of examples
m = size(X, 1);


%% =========== Feature Mapping for Polynomial Regression =============


% Map X onto Polynomial Features and Normalize
X_poly = polyMolyFeatures(X, 2);
[X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
X_poly = [ones(m, 1), X_poly];                   % Add Ones

% Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyMolyFeatures(Xtest, 2);
X_poly_test = bsxfun(@minus, X_poly_test, mu);
X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);
X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];         % Add Ones

% Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyMolyFeatures(Xval, 2);
X_poly_val = bsxfun(@minus, X_poly_val, mu);
X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];           % Add Ones


%Eval
eval_poly_val = polyMolyFeatures(eval, 2);
eval_poly_val = bsxfun(@minus, eval_poly_val, mu);
eval_poly_val = bsxfun(@rdivide, eval_poly_val, sigma);
eval_poly_val = [ones(size(eval_poly_val, 1), 1), eval_poly_val];           % Add Ones

%% =========== Learning Curve for Polynomial Regression =============
lambda_vec = [0,0.1,0.2,0.4,0.8,1.0,10,100];
%lambda_vec = [1];
submitx = [];
for i = 1:length(lambda_vec)
lambda = lambda_vec(i);
[theta] = trainLinearReg(X_poly, y, lambda);


[error_train, error_val] = ...
    learningCurve(X_poly, y, X_poly_val, yval, lambda);


%% ===========Validation for Selecting Lambda =============



% Finally 
prediction(:,i) = X_poly_test*theta;
submitx = [submitx prediction(:,i) ytest];

end