data = readtable('text_classification_data.csv', 'TextType', 'string');
documents = data.Text;
labels = data.Label;
Y = categorical(labels);

cv = cvpartition(Y, 'HoldOut', 0.2);
idxTrain = training(cv);
idxVal = test(cv);

trainDocs = documents(idxTrain);
trainLabels = Y(idxTrain);
valDocs = documents(idxVal);
valLabels = Y(idxVal);

trainTokenized = tokenizedDocument(trainDocs);
valTokenized = tokenizedDocument(valDocs);

enc = wordEncoding(trainTokenized);
XTrain = doc2sequence(enc, trainTokenized);
XVal = doc2sequence(enc, valTokenized);

inputSize = 1;
embeddingDimension = 50;
numHiddenUnits = 100;
numClasses = numel(categories(Y));

layers = [
sequenceInputLayer(inputSize)
wordEmbeddingLayer(embeddingDimension, enc.NumWords)
lstmLayer(numHiddenUnits, 'OutputMode', 'last')
fullyConnectedLayer(numClasses)
softmaxLayer
classificationLayer
];

options = trainingOptions('adam', ...
'MaxEpochs', 30, ...
'MiniBatchSize', 4, ...
'ValidationData', {XVal, valLabels}, ...
'Shuffle', 'every-epoch', ...
'Verbose', 0, ...
'Plots', 'training-progress');

net = trainNetwork(XTrain, trainLabels, layers, options);

YPredTrain = classify(net, XTrain);
trainAccuracy = mean(YPredTrain == trainLabels);
fprintf('Training Accuracy: %.2f%%\n', trainAccuracy * 100);

YPredVal = classify(net, XVal);
valAccuracy = mean(YPredVal == valLabels);
fprintf('Validation Accuracy: %.2f%%\n', valAccuracy * 100);

testSamples = [
"He scored an amazing goal in the final match"
"The new AI processor improves smartphone performance"
"They won the football league this year"
"Innovative apps are changing the mobile industry"
"The athlete broke the world record in running"
];

testActualLabels = [
"sports"
"technology"
"sports"
"technology"
"sports"
];
testActualLabels = categorical(testActualLabels);

testTokenized = tokenizedDocument(testSamples);
XTest = doc2sequence(enc, testTokenized);
YPredTest = classify(net, XTest);

for i = 1:numel(testSamples)
fprintf('Sample %d: "%s"\n Actual: %s | Predicted: %s\n\n', ...
i, testSamples(i), string(testActualLabels(i)), string(YPredTest(i)));
end
