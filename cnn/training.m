% Clear workspace and command window
clear; clc;

% Load the training, validation, and test datasets
trainDataPath = fullfile(pwd, 'data', 'trainData.mat');
valDataPath = fullfile(pwd, 'data', 'valData.mat');
testDataPath = fullfile(pwd, 'data', 'testData.mat');

load(trainDataPath);
load(valDataPath);
load(testDataPath);

% Load the untrained ResNet-18 network
netPath =  fullfile(pwd, 'cnn-model', 'raw_resnet18_net_nn128-64-32_nh3.mat');  
load(netPath);
lgraph_1 = lgraph_3;  % Assuming 'lgraph_2' is the loaded graph

% Normalize training target data
%[trainData.Targets, C, S] = normalize(trainData.Targets);
%valData.Targets = (valData.Targets - C) / S;

% Define training hyperparameters
numTrain = height(trainData);
miniBatchSize = 64;
validationFrequency = floor(numTrain / miniBatchSize);

% Set up training options
options = trainingOptions('adam', ...
    'MiniBatchSize', miniBatchSize, ...
    'MaxEpochs', 15, ...
    'InitialLearnRate', 1e-4, ...
    'LearnRateSchedule', 'none', ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', valData, ...
    'ValidationFrequency', validationFrequency, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% Train the network
net = trainNetwork(trainData, lgraph_1, options);

% Load test data and normalize its targets
load(testDataPath);
testData = sortrows(testData, 'Images', 'ascend');
%testData.Targets = (testData.Targets - C) / S;

% Predict with the trained model
YPredicted = predict(net, testData);

% Denormalize the predicted outputs and ground truth
%YPredicted = YPredicted * S + C;
%testData.Targets = testData.Targets * S + C;

% Plot predictions vs ground truth
figure(59);
plot(YPredicted, 'LineWidth', 1.5);
hold on;
plot(testData.Targets, 'LineWidth', 1.5);
legend('Predictions', 'Ground Truth');
title('Predictions vs Ground Truth');
xlabel('Sample');
ylabel('Target Value');
grid on;

% Calculate and display MSE and RMSE
mse = mean((testData.Targets - YPredicted).^2);
disp(['MSE: ', num2str(mse)]);
rmse = sqrt(mse);
disp(['RMSE: ', num2str(rmse)]);

% Save the model
save(fullfile(pwd, 'cnn-model', 'trained_model.mat'), 'net');
