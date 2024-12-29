clear; clc;
folder = 'data';
sequenceLength = 3;

% Define the directory where your folders are located
mainDirectory = [folder, '\train'];

% Get a list of all folders in the main directory
folders = dir(fullfile(mainDirectory, '*'));

dataTable = [];
% Loop through each folder
for i = 1:numel(folders)
    if folders(i).isdir && ~strcmp(folders(i).name, '.') && ~strcmp(folders(i).name, '..')
        currentFolder = fullfile(mainDirectory, folders(i).name);
        disp(['Processing folder: ' currentFolder]);
        data_path = currentFolder;
        data = readtable(fullfile(currentFolder, [folders(i).name '.csv']), 'Delimiter', ',');
        if ismember('Var2', data.Properties.VariableNames)
            targetValue = data.Var2;
        else
            targetValue = data.Targets;
        end
        
        imds = imageDatastore(fullfile(data_path), ...
            "IncludeSubfolders", true, "FileExtensions", ".jpg", "LabelSource", "foldernames");
        trainData = table(imds.Files, targetValue, 'VariableNames', {'Images', 'Targets'});
        trainData = sortrows(trainData, "Images", "ascend");
        dataTable = [dataTable; trainData];
    end
end
clear data trainData;

saturatedTargets = apply_saturation(dataTable.Targets);
meanTargets = mean(saturatedTargets);
stdTargets = std(saturatedTargets);
C = meanTargets;
S = stdTargets;

disp(['Mean of targets: ', num2str(meanTargets)]);
disp(['Standard deviation of targets: ', num2str(stdTargets)]);

figure;
histogram(saturatedTargets);
xlabel('Steering angle in radian');
ylabel('Count');
title('Histogram of saturated targets');

normalizedTargets = (saturatedTargets - meanTargets) / stdTargets;
figure;
histogram(normalizedTargets);
xlabel('Steering angle in radian');
ylabel('Count');
title('Histogram of saturated and normalized (z-score) targets');

mainDirectory = [folder, '\train'];
folders = dir(fullfile(mainDirectory, '*'));

for i = 1:numel(folders)
    dark_table = [];
    bright_table = [];
    if folders(i).isdir && ~strcmp(folders(i).name, '.') && ~strcmp(folders(i).name, '..')
        currentFolder = fullfile(mainDirectory, folders(i).name);
        disp(['Processing folder: ' currentFolder]);
        data_path = currentFolder;
        data = readtable(fullfile(currentFolder, [folders(i).name '.csv']), 'Delimiter', ',');
        if ismember('Var2', data.Properties.VariableNames)
            targetValue = data.Var2;
        else
            targetValue = data.Targets;
        end
        
        imds = imageDatastore(fullfile(data_path), ...
            "IncludeSubfolders", true, "FileExtensions", ".jpg", "LabelSource", "foldernames");
        data = table(imds.Files, targetValue, 'VariableNames', {'Images', 'Targets'});
        data = sortrows(data, "Images", "ascend");

        filePaths = data.Images;

        for j = 1:size(data, 1)
            img = imread(filePaths{j});
            img_adjusted = jitterColorHSV(img, Brightness = [0.2, 0.4]);

            [~, filename, ext] = fileparts(filePaths{j});
            new_filename = [filename '_brighter' ext];
            augmented_folder = [currentFolder, '_augmented_brighter'];
            if ~exist(augmented_folder, 'dir')
                mkdir(augmented_folder);
            end
            new_filepath = fullfile(augmented_folder, new_filename);
            imwrite(img_adjusted, new_filepath);

            new_row = table({new_filepath}, data.Targets(j), 'VariableNames', {'Images', 'Targets'});
            bright_table = [bright_table; new_row];
        end
        writetable(bright_table, fullfile(augmented_folder, [folders(i).name, '_augmented_brighter.csv']));

        for j = 1:size(data, 1)
            img = imread(filePaths{j});
            img_adjusted = jitterColorHSV(img, Brightness = [-0.3, -0.1]);

            [~, filename, ext] = fileparts(filePaths{j});
            new_filename = [filename '_darker' ext];
            augmented_folder = [currentFolder, '_augmented_darker'];
            if ~exist(augmented_folder, 'dir')
                mkdir(augmented_folder);
            end
            new_filepath = fullfile(augmented_folder, new_filename);
            imwrite(img_adjusted, new_filepath);

            new_row = table({new_filepath}, data.Targets(j), 'VariableNames', {'Images', 'Targets'});
            dark_table = [dark_table; new_row];
        end
        writetable(dark_table, fullfile(augmented_folder, [folders(i).name, '_augmented_darker.csv']));
    end
end

mainDirectory = [folder, '\test'];
folders = dir(fullfile(mainDirectory, '*'));
combinedTestFeatures = [];
combinedTestLabels = [];

for i = 1:numel(folders)
    if folders(i).isdir && ~strcmp(folders(i).name, '.') && ~strcmp(folders(i).name, '..')
        currentFolder = fullfile(mainDirectory, folders(i).name);
        disp(['Processing folder: ' currentFolder]);
        data_path = currentFolder;
        data = readtable(fullfile(currentFolder, [folders(i).name '.csv']), 'Delimiter', ',');
        if ismember('Var2', data.Properties.VariableNames)
            targetValue = data.Var2;
        else
            targetValue = data.Targets;
        end
        
        targetValue = apply_saturation(targetValue);
        targetValue = (targetValue - C) / S;

        imds = imageDatastore(fullfile(data_path), ...
            "IncludeSubfolders", true, "FileExtensions", ".jpg", "LabelSource", "foldernames");
        testData = table(imds.Files, targetValue, 'VariableNames', {'Images', 'Targets'});
        testData = sortrows(testData, "Images", "ascend");

        combinedTestFeatures = [combinedTestFeatures; testData.Images];
        combinedTestLabels = [combinedTestLabels; testData.Targets];
    end
end

testDir = [folder, '\concatenated_images_', num2str(sequenceLength), '\test'];
if ~exist(testDir, 'dir')
    mkdir(testDir);
end

testImagePaths = cell(size(combinedTestFeatures, 1), 1);
testLabels = zeros(size(combinedTestFeatures, 1), 1);

for i = 1:size(combinedTestFeatures, 1)
    image = imread(combinedTestFeatures{i});
    filename = [num2str(i), '.jpg'];
    fullPath = fullfile(testDir, filename);
    testImagePaths{i} = fullPath;
    testLabels(i) = combinedTestLabels(i);
    imwrite(image, fullPath);
end

testData = table(testImagePaths, testLabels);
save(fullfile(testDir, 'conc_combinedTestFeatures.mat'), 'testData');

shuffledTrain = trainData(randperm(size(trainData, 1)), :);
shuffledTest = testData(randperm(size(testData, 1)), :);

trainRatio = 0.85;
valRatio = 0.15;

numTrain = round(height(shuffledTrain) * trainRatio);
numVal = round(height(shuffledTrain) * valRatio);

trainData = shuffledTrain(1:numTrain, :);
valData = shuffledTrain(numTrain + 1:numTrain + numVal - 1, :);
testData = shuffledTest;

dropout_rate = 0.5;
layers = [
    imageInputLayer([sequenceLength * 224, 224, 3], "Normalization", "zscore")
    convolution2dLayer(5, 24, 'Stride', [2, 2])
    eluLayer()
    convolution2dLayer(5, 36, 'Stride', [2, 2])
    eluLayer()
    convolution2dLayer(5, 48, 'Stride', [2, 2])
    eluLayer()
    convolution2dLayer(5, 64, 'Stride', [2, 2])
    eluLayer()
    dropoutLayer(dropout_rate)
    flattenLayer
    fullyConnectedLayer(100)
    fullyConnectedLayer(50)
    lstmLayer(64, 'OutputMode', 'last')
    fullyConnectedLayer(1)
    regressionLayer()
];

miniBatchSize = 256;
options = trainingOptions('adam', ...
    'MiniBatchSize', miniBatchSize, ...
    'MaxEpochs', 100, ...
    'InitialLearnRate', 1e-3, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', valData, ...
    'Plots', 'training-progress', ...
    'Verbose', true);

netCNN_LSTM = trainNetwork(trainData, layers, options);

YPredicted = predict(netCNN_LSTM, testData);
YPredicted = YPredicted * S + C;
testData.testLabels = testData.testLabels * S + C;

figure;
plot(YPredicted, 'LineWidth', 1.5);
hold on;
grid on;
plot(testData.testLabels, 'LineWidth', 1.5);
legend("Prediction", "Ground Truth");
xlabel("Test Data Image Index");
ylabel("Steering Angle [rad]");
title("Prediction vs Ground Truth");
saveas(gcf, 'prediction_vs_groundTruth_graph.fig');

mse = mean((testData.testLabels - YPredicted).^2);
disp(['MSE: ', num2str(mse)]);
rmse = sqrt(mse);
disp(['RMSE: ', num2str(rmse)]);
