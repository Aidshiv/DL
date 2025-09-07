% Program 2:
% Write a program to demonstrate the working of a deep neural network for classification task.
digitDatasetPath = fullfile(matlabroot, 'toolbox', 'nnet', 'nndemos', 'nndatasets', 'DigitDataset');
imds = imageDatastore(digitDatasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
disp("Total number of images: " + numel(imds.Files));
labelCount = countEachLabel(imds);
disp(labelCount);
figure;
perm = randperm(numel(imds.Files), 20);
for i = 1:20
    subplot(4,5,i);
    img = readimage(imds, perm(i));
    imshow(img);
    title(char(imds.Labels(perm(i))));
end
sgtitle('Sample Digits from the Dataset');
minSetCount = min(countEachLabel(imds).Count);
imds = splitEachLabel(imds, minSetCount, 'randomize');
imageSize = [28 28];
imds.ReadFcn = @(filename) imresize(imread(filename), imageSize);
[imdsTrainFull, imdsTest] = splitEachLabel(imds, 0.7, 'randomized');
[imdsTrain, imdsVal] = splitEachLabel(imdsTrainFull, 0.85, 'randomized');
layers = [
    imageInputLayer([28 28 1], 'Name', 'input')
    fullyConnectedLayer(256, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(128, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(64, 'Name', 'fc3')
    reluLayer('Name', 'relu3')
    fullyConnectedLayer(10, 'Name', 'fc4')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];
options = trainingOptions('adam',
    'MaxEpochs', 5,
    'MiniBatchSize', 64,
    'Shuffle', 'every-epoch',
    'ValidationData', imdsVal,
    'ValidationFrequency', 30,
    'Verbose', false,
    'Plots', 'training-progress');
net = trainNetwork(imdsTrain, layers, options);
YPredTest = classify(net, imdsTest);
YTest = imdsTest.Labels;
testAccuracy = sum(YPredTest == YTest) / numel(YTest);
fprintf('\nTest Accuracy: %.2f%%\n', testAccuracy * 100);
YPredVal = classify(net, imdsVal);
YVal = imdsVal.Labels;
valAccuracy = sum(YPredVal == YVal) / numel(YVal);
fprintf('Validation Accuracy: %.2f%%\n', valAccuracy * 100);
figure;
confusionchart(YTest, YPredTest);
title('Confusion Matrix Test Data');
