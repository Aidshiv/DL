numTimeSteps = 200;
t = (1:numTimeSteps)';
data = sin(0.05*t) + 0.1*randn(numTimeSteps,1);

X = data(1:end-1)';
Y = data(2:end)';

X = {X};
Y = {Y};

totalSteps = numel(X{1});
numTrain = floor(0.8 * totalSteps);
numVal = floor(0.1 * totalSteps);

XTrain = {X{1}(:, 1:numTrain)};
YTrain = {Y{1}(:, 1:numTrain)};

XVal = {X{1}(:, numTrain+1:numTrain+numVal)};
YVal = {Y{1}(:, numTrain+1:numTrain+numVal)};

XTest = {X{1}(:, numTrain+numVal+1:end)};
YTest = {Y{1}(:, numTrain+numVal+1:end)};

numFeatures = 1;
numResponses = 1;
numHiddenUnits = 100;

layers = [
sequenceInputLayer(numFeatures)
lstmLayer(numHiddenUnits)
fullyConnectedLayer(numResponses)
regressionLayer
];

options = trainingOptions('adam', ...
'MaxEpochs', 200, ...
'GradientThreshold', 1, ...
'InitialLearnRate', 0.005, ...
'MiniBatchSize', 1, ...
'Shuffle', 'never', ...
'ValidationData', {XVal, YVal}, ...
'Verbose', 0, ...
'Plots', 'training-progress');

net = trainNetwork(XTrain, YTrain, layers, options);

YPredVal = predict(net, XVal, 'MiniBatchSize', 1);
rmseVal = sqrt(mean((YPredVal{1} - YVal{1}).^2));
fprintf('Validation RMSE: %.4f\n', rmseVal);

YPredTest = predict(net, XTest, 'MiniBatchSize', 1);
rmseTest = sqrt(mean((YPredTest{1} - YTest{1}).^2));
fprintf('Test RMSE: %.4f\n', rmseTest);
