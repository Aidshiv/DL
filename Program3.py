
 #Program 3:Design and implement a Convolutional Neural Network(CNN) for classification of image dataset.
digitDatasetPath = fullfile(matlabroot, 'toolbox', 'nnet', 'nndemos', 'nndatasets', 'DigitDataset');
imds = imageDatastore(digitDatasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
minSetCount = min(countEachLabel(imds).Count);
imds = splitEachLabel(imds, minSetCount, 'randomize');
imageSize = [28 28];
imds.ReadFcn = @(filename)imresize(imread(filename), imageSize);
[imdsTrainFull, imdsTest] = splitEachLabel(imds, 0.7, 'randomized');
[imdsTrain, imdsVal] = splitEachLabel(imdsTrainFull, 0.85, 'randomized');
layers = [
    imageInputLayer([28 28 1], 'Name', 'input')
    convolution2dLayer(3, 8, 'Padding', 'same', 'Name', 'conv_1')
    batchNormalizationLayer('Name', 'batchnorm_1')
    reluLayer('Name', 'relu_1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_1')
    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv_2')
    batchNormalizationLayer('Name', 'batchnorm_2')
    reluLayer('Name', 'relu_2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_2')
    fullyConnectedLayer(64, 'Name', 'fc_1')
    reluLayer('Name', 'relu_fc')
    fullyConnectedLayer(10, 'Name', 'fc_out')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];
disp('Visualizing CNN Layers...');
analyzeNetwork(layers);
lgraph = layerGraph(layers);
figure;
plot(lgraph);
title('CNN Layer Connectivity');
options = trainingOptions('adam',
    'MaxEpochs', 6,
    'MiniBatchSize', 64,
    'Shuffle', 'every-epoch',
    'ValidationData', imdsVal,
    'ValidationFrequency', 30,
    'Verbose', false,
    'Plots', 'training-progress');
net = trainNetwork(imdsTrain, layers, options);
YPredTest = classify(net, imdsTest);
YTrueTest = imdsTest.Labels;
testAccuracy = sum(YPredTest == YTrueTest) / numel(YTrueTest);
fprintf('\nTest Accuracy: %.2f%%\n', testAccuracy * 100);
YPredVal = classify(net, imdsVal);
YTrueVal = imdsVal.Labels;
valAccuracy = sum(YPredVal == YTrueVal) / numel(YTrueVal);
fprintf('Validation Accuracy: %.2f%%\n', valAccuracy * 100);
figure;
confusionchart(YTrueTest, YPredTest);
title('Confusion Matrix CNN Test Data');
