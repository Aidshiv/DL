datasetPath = 'D:\RNSIT\AIML\2025-26_ODD_sem\Deep Learning\Lab Programs\flowers';
imds = imageDatastore(datasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[imdsTrainRaw, imdsValRaw] = splitEachLabel(imds, 0.8, 'randomized');
numClasses = numel(categories(imdsTrainRaw.Labels));

inputSize = [224 224 3];
imdsTrain = augmentedImageDatastore(inputSize, imdsTrainRaw);
imdsVal = augmentedImageDatastore(inputSize, imdsValRaw);

net = resnet18();
lgraph = layerGraph(net);
lgraph = removeLayers(lgraph, {'fc1000','prob','ClassificationLayer_predictions'});

newLayers = [
fullyConnectedLayer(numClasses, 'Name', 'fcNew')
softmaxLayer('Name','softmax')
classificationLayer('Name','output')
];

lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'pool5', 'fcNew');

options = trainingOptions('adam', ...
'MiniBatchSize', 32, ...
'MaxEpochs', 5, ...
'Shuffle', 'every-epoch', ...
'ValidationData', imdsVal, ...
'ValidationFrequency', 20, ...
'Verbose', false, ...
'Plots', 'training-progress');

trainedNet = trainNetwork(imdsTrain, lgraph, options);

YPred = classify(trainedNet, imdsVal);
YTrue = imdsValRaw.Labels;
accuracy = mean(YPred == YTrue);
fprintf('Validation Accuracy: %.2f%%\n', accuracy * 100);

classes = categories(YTrue);
confMat = confusionmat(YTrue, YPred);
for i = 1:numel(classes)
fprintf('Class: %-15s Correct: %-3d Total: %-3d\n', string(classes{i}), confMat(i,i), sum(confMat(i,:)));
end

figure;
confusionchart(YTrue, YPred);
title('Confusion Matrix');

figure;
uniqueClasses = categories(YTrue);
shown = false(size(uniqueClasses));
shownCount = 0;
for i = 1:numel(imdsValRaw.Files)
img = readimage(imdsValRaw, i);
label = YTrue(i);
pred = classify(trainedNet, imresize(img, inputSize(1:2)));
idx = find(uniqueClasses == label);
if ~shown(idx)
shownCount = shownCount + 1;
subplot(ceil(numel(uniqueClasses)/3), 3, shownCount);
imshow(img);
title(sprintf('True: %s\nPred: %s', string(label), string(pred)));
shown(idx) = true;
end
if all(shown)
break;
end
end
