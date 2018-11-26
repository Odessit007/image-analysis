clc, clear, close all
rootFolder = 'H:\PycharmProjects\ml36\Signal-Processing-Notebooks\PM5\cifar-10';
categories = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'};
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');

tbl = countEachLabel(imds)

minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category

% Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');

% Notice that each set now has exactly the same number of images.
countEachLabel(imds)

[trainingSet, validationSet] = splitEachLabel(imds, 5000, 'randomize');

% Find and plot the first instance of an image for each category
airplane = find(trainingSet.Labels == 'airplane', 1);
figure
imshow(readimage(trainingSet,airplane))

% BoVW
bag = bagOfFeatures(trainingSet);
img = readimage(imds, 1);
featureVector = encode(bag, img);

% Plot the histogram of visual word occurrences
figure
bar(featureVector)
title('Visual word occurrences')
xlabel('Visual word index')
ylabel('Frequency of occurrence')

% Train classifier
categoryClassifier = trainImageCategoryClassifier(trainingSet, bag);
confMatrix = evaluate(categoryClassifier, trainingSet)
confMatrix = evaluate(categoryClassifier, validationSet)
% Compute average accuracy
mean(diag(confMatrix));


% Open some image
img = imread(fullfile(rootFolder, 'airplane', '36.jpg'));
[labelIdx, scores] = predict(categoryClassifier, img);
% Display the string label
categoryClassifier.Labels(labelIdx)