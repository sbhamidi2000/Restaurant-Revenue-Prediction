%% Restaurant Revenue prediction Neural Network Learning
%

clear ; close all; clc

%% Setup the parameters you will use 
input_layer_size  = 38;  % 38
hidden_layer_size = 25;   % 25 hidden units
num_labels = 1; 

%% =========== Part 1: Load and Visualize Data =============
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

data = dlmread('trainrun.csv');
X = data(2:138, 1:38);
y = data(2:138, 39);

m = size(X, 1);


%% ================ Part 2: Loading Parameters ================
% Load some pre-initialized neural network parameters.
% Unroll parameters 
Theta1 = ones(25,39) ; 
Theta2= ones(1,26);
nn_params = [Theta1(:) ; Theta2(:)];

%% ================ Part 3: Compute Cost (Feedforward) ================

fprintf('\nFeedforward Using Neural Network ...\n')

% Weight regularization parameter (we set this to 0 here).
lambda = 0;

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

fprintf(['Cost at parameters  %f '...
         '\n'], J);


%% =============== Part 4: Implement Regularization ===============

fprintf('\nChecking Cost Function (w/ Regularization) ... \n')

% Weight regularization parameter (we set this to 1 here).
lambda = 1;

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

fprintf(['Cost at parameters : %f '...
         '\n'], J);

%% ================ Part 5: Initializing Pameters ================
% Implement a two layer neural network

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


%% =============== Part 6: Implement Backpropagation ===============

fprintf('\nChecking Backpropagation... \n');

%  Check gradients by running checkNNGradients
checkNNGradients;

%% =============== Part 7: Implement Regularization ===============


fprintf('\nChecking Backpropagation (w/ Regularization) ... \n')

%  Check gradients by running checkNNGradients
lambda = 3;
checkNNGradients(lambda);

% Also output the costFunction debugging values
debug_J  = nnCostFunction(nn_params, input_layer_size, ...
                          hidden_layer_size, num_labels, X, y, lambda);

fprintf(['\n\nCost at (fixed) debugging parameters (w/ lambda = 10): %f ' ...
         '\n(this value should be about 0.576051)\n\n'], debug_J);


%% =================== Part 8: Training NN ===================
%. To train your neural network, we will now use "fmincg"
%
fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', 50000);

%  You should also try different values of lambda using cross validation set 
lambda = 100; 

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

                 %% ================= Part 9: Implement Predict =================

pred = predict(Theta1, Theta2, X);

%fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


