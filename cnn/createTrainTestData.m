clear; clc;

%% Main Script

% Data paths and file names
train_folder1 = fullfile('data', 'resized_data_10_31_14_12');
train_csv1 = 'data_10_31_14_12.csv';

test_folder1 = fullfile('data', 'resized_data_10_31_14_14');
test_csv1 = 'data_10_31_14_14.csv';


augmented_train_folder = fullfile(pwd,'data', 'augmented_train');
augmented_test_folder = fullfile(pwd,'data', 'augmented_test');

%% Load Training Data
train_data1 = loadImageData(train_folder1, train_csv1);
train_data2 = loadImageData(train_folder2, train_csv2);

% Combine training data and saturate targets
train_table = [train_data1; train_data2];
%train_table.Targets = saturateTargets(train_table.Targets);

%% Load Testing Data
test_data1 = loadImageData(test_folder1, test_csv1);
test_data2 = loadImageData(test_folder2, test_csv2);

% Combine testing data and saturate targets
test_table = [test_data1; test_data2];
%test_table.Targets = saturateTargets(test_table.Targets);

%% Augment Data
train_table = augmentImages(train_table, augmented_train_folder);
test_table = augmentImages(test_table, augmented_test_folder);

%% Shuffle Data
shuffledTrain = train_table(randperm(size(train_table, 1)), :);
shuffledTest = test_table(randperm(size(test_table, 1)), :);

%% Split Data into Training, Validation, and Testing Sets
trainRatio = 0.85;
valRatio = 0.15;

numTrain = round(height(shuffledTrain) * trainRatio);
numVal = round(height(shuffledTrain) * valRatio);

trainData = shuffledTrain(1:numTrain, :);
valData = shuffledTrain(numTrain+1:numTrain+numVal-1, :);
testData = shuffledTest;

% Output dataset sizes for verification
fprintf('Training data size: %d\n', size(trainData, 1));
fprintf('Validation data size: %d\n', size(valData, 1));
fprintf('Test data size: %d\n', size(testData, 1));

% Save trainData, testData, and valData into the 'data' folder
save(fullfile('data', 'trainData.mat'), 'trainData');
save(fullfile('data', 'testData.mat'), 'testData');
save(fullfile('data', 'valData.mat'), 'valData');

% Display confirmation
disp('Data saved successfully:');
disp('trainData.mat');
disp('testData.mat');
disp('valData.mat');


%% Utility Functions

% Function to load image data and targets
function dataTable = loadImageData(data_path, csv_filename)
    csv_fullpath = fullfile(data_path, csv_filename);
    data = readtable(csv_fullpath, "Delimiter", 'comma');
    targetValue = data.Var2;
    
    imds = imageDatastore(fullfile(data_path), ...
        "IncludeSubfolders", true, "FileExtensions", ".jpg", "LabelSource", "foldernames");
    
    dataTable = table(imds.Files, targetValue, 'VariableNames', {'Images', 'Targets'});
end

% Function to saturate the target values between -0.34 and 0.34
function targets = saturateTargets(targets)
    targets(targets > 0.34) = 0.34;
    targets(targets < -0.34) = -0.34;
end

% Function to augment images with brightness adjustment
function augmentedTable = augmentImages(data_table, output_folder)
    if ~isfolder(output_folder)
        mkdir(output_folder);
    end
    
    augmentedTable = data_table;
    for i = 1:size(data_table, 1)
        img = imread(data_table.Images{i});
        
        % Brighter image
        img_bright = jitterColorHSV(img, 'Brightness', [0.3 0.5]);
        [~, filename, ext] = fileparts(data_table.Images{i});
        new_filepath_bright = fullfile(output_folder, [filename '_brighter' ext]);
        imwrite(img_bright, new_filepath_bright);
        new_row_bright = table({new_filepath_bright}, data_table.Targets(i), 'VariableNames', {'Images', 'Targets'});
        augmentedTable = [augmentedTable; new_row_bright];
        
        % Darker image
        img_dark = jitterColorHSV(img, 'Brightness', [-0.3 -0.1]);
        new_filepath_dark = fullfile(output_folder, [filename '_darker' ext]);
        imwrite(img_dark, new_filepath_dark);
        new_row_dark = table({new_filepath_dark}, data_table.Targets(i), 'VariableNames', {'Images', 'Targets'});
        augmentedTable = [augmentedTable; new_row_dark];
    end
end