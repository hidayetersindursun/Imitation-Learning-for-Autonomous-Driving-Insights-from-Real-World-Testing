% Load testing data from created tables (resized to 224x224 and augmented with darker and brighter)

clear; clc;

% Load testing data
testDataPath = fullfile(pwd, "data/testData.mat");
load(testDataPath);

% Load trained model for inference
load cnn-model\trained_model.mat;

% Sort test data by image paths
testData = sortrows(testData, 'Images', 'ascend');

% Predict using the trained model
YPredicted = predict(net, testData);

% Close existing figures
close all;

% Plot predictions vs ground truth
figure;
plot(YPredicted, "LineWidth", 1.5);
hold on;
plot(testData.Targets, "LineWidth", 1.5);
legend("Predictions", "Ground Truth");
xlabel("Test Data Index");
ylabel("Output Value");
title("Model Predictions vs Ground Truth");

% Calculate and display MSE and RMSE
mse = mean((testData.Targets - YPredicted).^2);
disp(['MSE: ', num2str(mse)]);

rmse = sqrt(mean((testData.Targets - YPredicted).^2));
disp(['RMSE: ', num2str(rmse)]);
