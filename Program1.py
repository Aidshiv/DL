% Program 1:
% Design and implement a neural-based network for generating word embedding for words in a document corpus.
textData = fileread('sample_corpus.txt');
textData = lower(textData);
textData = regexprep(textData, '[^a-z\s]', '');
words = strsplit(strtrim(textData));
stopWords = ["the", "is", "and", "to", "with", "this", "as","on", "for", "it", "i", "of", "a", "in", "be"];
words = words(~ismember(words, stopWords));
words = words(~cellfun(@isempty, words));
words = words(cellfun(@(w) length(w) > 1, words));
wordCountsMap = containers.Map();
for i = 1:length(words)
    word = words{i};
    if iskey(wordCountsMap, word)
        wordCountsMap(word) = wordCountsMap(word) + 1;
    else
        wordCountsMap(word) = 1;
    end
end
minFreq = 2;
vocabWords = keys(wordCountsMap);
counts = cell2mat(values(wordCountsMap));
keep = counts >= minFreq;
vocabWords = vocabWords(keep);
vocabSize = length(vocabWords);
word2idx = containers.Map(vocabWords, 1:vocabSize);
idx2word = vocabWords;
windowSize = 4;
X = [];
Y = [];
for i = 1:length(words)
    if ~iskey(word2idx, words{i})
        continue;
    end
    centerIdx = word2idx(words{i});
    for j = max(1, i - windowSize):min(length(words), i + windowSize)
        if j ~= i && iskey(word2idx, words{j})
            contextIdx = word2idx(words{j});
            X = [X; centerIdx];
            Y = [Y; contextIdx];
        end
    end
end
X_onehot = full(ind2vec(X', vocabSize));
Y_onehot = full(ind2vec(Y', vocabSize));
numSamples = size(X_onehot, 2);
idx = randperm(numSamples);
trainRatio = 0.8;
numTrain = round(trainRatio * numSamples);
trainIdx = idx(1:numTrain);
valIdx = idx(numTrain+1:end);
XTrain = X_onehot(:, trainIdx)';
YTrain = categorical(vec2ind(Y_onehot(:, trainIdx))', 1:vocabSize);
XVal = X_onehot(:, valIdx)';
YVal = categorical(vec2ind(Y_onehot(:, valIdx))', 1:vocabSize);
embeddingDim = 100;
layers = [
    featureInputLayer(vocabSize, "Name", "input")
    fullyConnectedLayer(embeddingDim, "Name", "embedding")
    fullyConnectedLayer(vocabSize, "Name", "fc")
    softmaxLayer("Name", "softmax")
    classificationLayer("Name", "output")
];
options = trainingOptions('adam',
    'MaxEpochs', 200,
    'MiniBatchSize', 256,
    'Shuffle', 'every-epoch',
    'ValidationData', {XVal, YVal},
    'Verbose', false,
    'Plots', 'training-progress');
net = trainNetwork(XTrain, YTrain, layers, options);
YPredVal = classify(net, XVal);
valAccuracy = sum(YPredVal == YVal) / numel(YVal);
fprintf('\nFinal Validation Accuracy: %.2f%%\n', valAccuracy * 100);
YValProb = predict(net, XVal);
YValProb = YValProb';
trueOneHot = full(ind2vec(double(YVal)', vocabSize));
valLoss = crossentropy(YValProb, trueOneHot);
fprintf('Final Validation Loss: %.4f\n', valLoss);
embeddingMatrix = net.Layers(2).Weights;
sampleWord = 'learning';
if iskey(word2idx, sampleWord)
    wordIdx = word2idx(sampleWord);
    fprintf('\nEmbedding vector for "%s": \n', sampleWord);
    disp(embeddingMatrix(:, wordIdx)');
else
    fprintf('\nWord "%s" not found in vocabulary.\n', sampleWord);
end
