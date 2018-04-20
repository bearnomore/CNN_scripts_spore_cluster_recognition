% Import combined data
dir = 'Y:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\OtherCode\CNNcode\New Flow\CombinedTrainingData\';
load([dir 'PropImgs_3Labels4trainingNetwork_comb'])
% load([dir 'PropImgs_3Labels4trainingNetwork_comb_bigger'])
% load([dir 'PropImgs_thickEdge_3Labels4trainingNetwork'])
%% Import semiauto data
% dir = 'Z:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\OtherCode\CNNcode\New Flow\CombinedTrainingData\SemiAutoLabeledData\';
% load([dir 'PropImgs_3Labels4trainingNetwork_semi'])
% %% Import manual data
% dir = 'Z:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\OtherCode\CNNcode\New Flow\CombinedTrainingData\ManualLabeledData\';
% load([dir 'PropImgs_3Labels4trainingNetwork_man'])
%% If use all PropImgs as trainImgs
trainImages = PropImgs;
%% Separate thte training imgs (processed) into train samples and cross validation samples
idx_crossVal = randperm(size(PropImgs,4),100000);
idx_train = setdiff(1:size(PropImgs,4), idx_crossVal);
valImages = PropImgs(:,:,:,idx_crossVal);
trainImages = PropImgs(:,:,:,idx_train);
valLabs = trainLabs(idx_crossVal);
trainLabs = trainLabs(idx_train);
%% Image augmentation by padding outboundary image with symmetric pixels
w = size(trainImages,1);
h = size(trainImages,2);
ImSize = size(trainImages,4);
% 5% for X shearing, 5% for Y shearing, 5% for rotation (45 degree
% clockwise); 5% for rotation (45 degree anti-clock wise); 
tform_Xshear = transformationMat2D('xshear', 0.45);
tform_Yshear = transformationMat2D('yshear', 0.45);
tform_Rotate = transformationMat2D('rotate', 45);
tform_Rotate_anti = transformationMat2D('rotate-anti', 45);

imID_Xshear = randperm(ImSize,round(ImSize*0.05));
imID_Yshear = randperm(ImSize,round(ImSize*0.05));
imID_Rotate = randperm(ImSize,round(ImSize*0.05));
imID_Rotate_anti = randperm(ImSize,round(ImSize*0.05));

train_Xshear = zeros(w,h,1,round(ImSize*0.05));
train_Yshear = zeros(w,h,1,round(ImSize*0.05));
train_Rotate = zeros(w,h,1,round(ImSize*0.05));
train_Rotate_anti = zeros(w,h,1,round(ImSize*0.05));

trainLab_rotate = trainLabs(imID_Rotate);
trainLab_rotate_anti = trainLabs(imID_Rotate_anti);
trainLab_Xshear = trainLabs(imID_Xshear);
trainLab_Yshear = trainLabs(imID_Yshear);
tic
for ii = 1:round(ImSize*0.05)
%     
    img1_rotate = trainImages(:,:,1,imID_Rotate(ii));
    img2_rotate = trainImages(:,:,1,imID_Rotate_anti(ii));
    img1_shear = trainImages(:,:,1,imID_Xshear(ii));
    img2_shear = trainImages(:,:,1,imID_Yshear(ii));
    temp_Rotate= imageTransform(img1_rotate, tform_Xshear, 'circular', 225);
    temp_Rotate_anti = imageTransform(img2_rotate, tform_Xshear, 'circular', 225);
    temp_Xshear = imageTransform(img1_rotate, tform_Xshear, 'circular', 225);
    temp_Yshear = imageTransform(img2_rotate, tform_Yshear, 'circular', 225);
    
    train_Rotate(:,:,1,ii) = temp_Rotate(7:33,:);
    train_Rotate_anti(:,:,1,ii) = temp_Rotate_anti(7:33,:);

    train_Xshear(:,:,1,ii) = temp_Xshear(7:33,:);
    train_Yshear(:,:,1,ii) = temp_Yshear(:,7:33);
end
toc
% trainImages = cat(4, trainImages, train_Rotate, train_Rotate_anti);
% trainLabs = [trainLabs; trainLab_rotate; trainLab_rotate_anti];

trainImages = cat(4, trainImages, train_Xshear, train_Yshear,train_Rotate, train_Rotate_anti);
trainLabs = [trainLabs; trainLab_Xshear; trainLab_Yshear, trainLab_rotate; trainLab_rotate_anti];

%% Create a simple CNN 
% Generate image augmenter 
% FillVal = median(trainImages(:));
% FillVal = max(trainImages(:));
imageAugmenter = imageDataAugmenter('RandXReflection', true,...
    'RandYReflection', true,...
    'RandXScale', [1 1.2],...
    'RandYScale', [1 1.2]);
%     'RandRotation', [-90 90],...
%     'RandXShear', [0 45],...
%     
%     'FillValue',FillVal);
    
%     'RandXTranslation', [0 20],...
%     'RandYTranslation', [0 20]);
    % 'RandRotation', [-90 90],...
    %
%         'RandYShear', [0 45]);
%         'FillValue',medianInt,...

% create image input layer for 27x27 image data
imSize = [size(trainImages,1), size(trainImages,2),1]; % one channel for the grayscale images
datasource = augmentedImageSource(imSize,trainImages,trainLabs,'DataAugmentation',imageAugmenter);
 
numImageCategories = 3; % two categories: being a germling or being a background


filterSize = [3 3];
numFilters = 64;
% Convolutional layer parameters
middleLayers = [
    
% The first convolutional layer has a bank of 64x [3x3] filters. (inPutSize
% - filterSize + 2* paddingSize)/stride + 1)
convolution2dLayer(filterSize, numFilters, 'Padding', 'same'); %16x[27x27]
% A batch normalization layer normalizes each input channel across a mini-batch. 
% The layer first normalizes the activations of each channel by subtracting the
% mini-batch mean and dividing by the mini-batch standard deviation. 
% Then, the layer shifts the input by a learnable offset ? and scales it by a learnable scale factor ?.
% Use batch normalization layers between convolutional layers and nonlinearities, 
% such as ReLU layers, to speed up training of convolutional neural networks
% and reduce the sensitivity to network initialization.
% 
batchNormalizationLayer
% Next add the ReLU layer:
reluLayer()
maxPooling2dLayer(2, 'Stride', 2, 'Padding',[0 1 0 1]); % [14x14]

% Next another convolutional layer with the same setting followed by relu

convolution2dLayer(filterSize, numFilters, 'Padding', 'same'); % 14x14
batchNormalizationLayer
reluLayer()
maxPooling2dLayer(2, 'Stride', 2, 'Padding',[0 0 0 0]); % 7x7 

convolution2dLayer(filterSize, numFilters, 'Padding', 'same'); %7x7
batchNormalizationLayer
reluLayer()
];

finalLayers = [

% Add a dropout layer randomly sets input elements to zero with a given probability.
dropoutLayer(0.5)
% Add a fully connected layer with 100 output neurons. The output size of
% this layer will be an array with a length of 100.
fullyConnectedLayer(100)

% Add an ReLU non-linearity.
reluLayer

% Add the last fully connected layer. At this point, the network must
% produce 2 signals that can be used to measure whether the input image
% belongs to one category or another. This measurement is made using the
% subsequent loss layers.
fullyConnectedLayer(numImageCategories)

% Add the softmax loss layer and classification layer. The final layers use
% the output of the fully connected layer to compute the categorical
% probability distribution over the image classes. During the training
% process, all the network weights are tuned to minimize the loss over this
% categorical distribution.
softmaxLayer
classificationLayer
];

%% Create a simple CNN with convolutional implementation of filters (in fully connected layers)
% Generate image augmenter 
% FillVal = median(trainImages(:));
% FillVal = max(trainImages(:));
imageAugmenter = imageDataAugmenter('RandXReflection', true,...
    'RandYReflection', true,...
    'RandXScale', [1 1.2],...
    'RandYScale', [1 1.2]);
%     'RandRotation', [-90 90],...
%     'RandXShear', [0 45],...
%     
%     'FillValue',FillVal);
    
%     'RandXTranslation', [0 20],...
%     'RandYTranslation', [0 20]);
    % 'RandRotation', [-90 90],...
    %
%         'RandYShear', [0 45]);
%         'FillValue',medianInt,...

% create image input layer for 27x27 image data
imSize = [size(trainImages,1), size(trainImages,2),1]; % one channel for the grayscale images
datasource = augmentedImageSource(imSize,trainImages,trainLabs,'DataAugmentation',imageAugmenter);
inputLayer = imageInputLayer(imSize); 
numImageCategories = 3; % two categories: being a germling or being a background


filterSize = [3 3];
numFilters = 64;
% Convolutional layer parameters
middleLayers = [
    
% The first convolutional layer has a bank of 64x [3x3] filters. (inPutSize
% - filterSize + 2* paddingSize)/stride + 1)
convolution2dLayer(filterSize, numFilters, 'Padding', 'same'); %16x[27x27]
% A batch normalization layer normalizes each input channel across a mini-batch. 
% The layer first normalizes the activations of each channel by subtracting the
% mini-batch mean and dividing by the mini-batch standard deviation. 
% Then, the layer shifts the input by a learnable offset ? and scales it by a learnable scale factor ?.
% Use batch normalization layers between convolutional layers and nonlinearities, 
% such as ReLU layers, to speed up training of convolutional neural networks
% and reduce the sensitivity to network initialization.
% 
batchNormalizationLayer
% Next add the ReLU layer:
reluLayer()
maxPooling2dLayer(2, 'Stride', 2, 'Padding',[0 1 0 1]); % [14x14]

% Next another convolutional layer with the same setting followed by relu

convolution2dLayer(filterSize, numFilters, 'Padding', 'same'); % 14x14x64
batchNormalizationLayer
reluLayer()
maxPooling2dLayer(2, 'Stride', 2, 'Padding',[0 0 0 0]); % 7x7 

convolution2dLayer(filterSize, numFilters); %5x5 x64
batchNormalizationLayer
reluLayer()

%Add 5x5 convolutional layer = connecting to 5x5x64 = 1600 units in a fully
%connected layer
convolution2dLayer([5,5], numFilters*25); %1x1x1600
dropoutLayer(0.5)
reluLayer()
convolution2dLayer([1,1], numFilters*25); %1x1x1600
dropoutLayer(0.5)
reluLayer()
];

finalLayers = [

% Add a dropout layer randomly sets input elements to zero with a given probability.

% Add a fully connected layer with 100 output neurons. The output size of
% this layer will be an array with a length of 100.
% fullyConnectedLayer(100)

% Add an ReLU non-linearity.


% Add the last fully connected layer. At this point, the network must
% produce 2 signals that can be used to measure whether the input image
% belongs to one category or another. This measurement is made using the
% subsequent loss layers.
fullyConnectedLayer(numImageCategories)

% Add the softmax loss layer and classification layer. The final layers use
% the output of the fully connected layer to compute the categorical
% probability distribution over the image classes. During the training
% process, all the network weights are tuned to minimize the loss over this
% categorical distribution.
softmaxLayer
classificationLayer
];
%% Combine layers
Layers = [
    inputLayer
    middleLayers
    finalLayers
    ];
%% Initialize 
Layers(2).Weights = 0.0001 * randn([[3 3] 1 numFilters]);
% Layers(2).Weights = normrnd(0,1,[3, 3, 1, 64])* sqrt(2/size(PropImgs,1)/size(PropImgs,2));

%%
miniBatchSize = 128;
% numValidationsPerEpoch = 10;
% validationFrequency = floor(size(trainImages,4)/miniBatchSize/numValidationsPerEpoch);
options = trainingOptions('sgdm',...
    'InitialLearnRate',0.001,...
    'Momentum', 0.9, ...
    'MaxEpochs',10,...
    'MiniBatchSize',miniBatchSize,...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 5, ...
    'L2Regularization', 0.0001, ...
    'Shuffle', 'every-epoch',...
    'ExecutionEnvironment', 'gpu',...
    'CheckpointPath','D:\CNN_temp',...
    'Plots','training-progress');
%     'ValidationFrequency',50,...
%     'ValidationData',{valImages,valLabs},...
%     'Plots','training-progress',...
%     'ValidationPatience',5000,...
%     'OutputFcn',@(info)stopIfAccuracyNotImproving(info,3));
    

    

% % Set the network training options
% opts = trainingOptions('sgdm', ...
%     'Momentum', 0.9, ...
%     'InitialLearnRate', 0.001, ...
%     'LearnRateSchedule', 'piecewise', ...
%     'LearnRateDropFactor', 0.1, ...
%     'LearnRateDropPeriod', 8, ...
%     'L2Regularization', 0.004, ...
%     'MaxEpochs', 20, ...
%     'MiniBatchSize', 32, ... % good between 16 and 128 (power of 2)
%     'ExecutionEnvironment', 'gpu',...
%     'Verbose', false);
%%
XuNet = trainNetwork(datasource,Layers,options);
%% temp saving 
temp_save = 'D:\CNN_temp\Short trains\';
save([temp_save 'CNN_spores_by3Lables_comb_bigger_s3_long_convImpOfFilter'], 'XuNet')
%% if stopped earlier , reume the taining
f = 'D:\CNN_temp\';
load([f 'convnet_checkpoint__1214240__2017_11_29__07_40_40'])
miniBatchSize = 128;
% numValidationsPerEpoch = 10;
% validationFrequency = floor(size(trainImages,4)/miniBatchSize/numValidationsPerEpoch);
options = trainingOptions('sgdm',...
    'InitialLearnRate',0.001,...
    'Momentum', 0.9, ...
    'MaxEpochs',5,...
    'MiniBatchSize',miniBatchSize,...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 5, ...
    'L2Regularization', 0.0001, ...
    'Shuffle', 'every-epoch',...
    'ExecutionEnvironment', 'gpu',...
    'CheckpointPath','D:\CNN_temp',...
    'Plots','training-progress');
XuNet2 = trainNetwork(datasource,Layers,options);
%%
% f = 'Z:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\OtherCode\CNNcode\New Flow\CombinedTrainingData\SemiAutoLabeledData\';
% temp_save = 'D:\CNN_temp\Short trains\';
% save([temp_save 'CNN_spores_by3Lables_comb_bigger_s3_15epoch_convImpOfFilter'], 'XuNet2')
% save([f 'CNN_spores_by3Lables_comb_v2'], 'XuNet')
