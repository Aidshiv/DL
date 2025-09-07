% Program 4:
% Build and demonstrate an autoencoder network using neural layers for data compression on image dataset.
digitDatasetPath = fullfile(matlabroot, 'toolbox', 'nnet', 'nndemos', 'nndatasets', 'DigitDataset');
imds = imageDatastore(digitDatasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
minSetCount = min(countEachLabel(imds).Count);
imds = splitEachLabel(imds, minSetCount, 'randomized');
imds = shuffle(imds);
numImages = 1000;
X = zeros(28*28, numImages);
labels = strings(numImages, 1);
for i = 1:numImages
    img = readimage(imds, i);
    img = imresize(img, [28 28]);
    img = im2double(img);
    X(:, i) = img(:);
    labels(i) = string(imds.Labels(i));
end
X = X';
X = single(X);
trainRatio = 0.8;
numTrain = round(trainRatio * numImages);
XTrain = X(1:numTrain, :);
XVal = X(numTrain+1:end, :);
YTrain = XTrain;
YVal = XVal;
layers = [
    featureInputLayer(784, 'Name', 'input')
    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(64, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(32, 'Name', 'bottleneck')
    fullyConnectedLayer(64, 'Name', 'fc3')
    reluLayer('Name', 'relu3')
    fullyConnectedLayer(128, 'Name', 'fc4')
    reluLayer('Name', 'relu4')
    fullyConnectedLayer(784, 'Name', 'fc5')
    regressionLayer('Name', 'output')
];
analyzeNetwork(layers);
options = trainingOptions('adam',
    'MaxEpochs', 20,
    'MiniBatchSize', 64,
    'Shuffle', 'every-epoch',
    'ValidationData', {XVal, YVal},
    'ValidationFrequency', 30,
    'Plots', 'training-progress',
    'Verbose', false);
net = trainNetwork(XTrain, YTrain, layers, options);
YTrainPred = predict(net, XTrain);
YValPred = predict(net, XVal);
figure;
for i = 1:5
    original = reshape(XTrain(i, :), [28 28]);
    reconstructed = reshape(YTrainPred(i, :), [28 28]);
    subplot(2, 5, i);
    imshow(original);
    title(['Original: ' + labels(i)]);
    subplot(2, 5, i + 5);
    imshow(reconstructed);
    title('Reconstructed');
end
sgtitle('Autoencoder - Training Set Reconstruction');
mseTrain = mean((XTrain(:) - YTrainPred(:)).^2);
mseVal = mean((XVal(:) - YValPred(:)).^2);
fprintf('\nMean Squared Reconstruction Error: \n');
fprintf('Training Set: %.6f\n', mseTrain);
fprintf('Validation Set: %.6f\n', mseVal);
