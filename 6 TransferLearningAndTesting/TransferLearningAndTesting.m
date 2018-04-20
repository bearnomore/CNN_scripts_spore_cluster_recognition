% Import combined data
% dir = 'Y:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\OtherCode\CNNcode\New Flow\CombinedTrainingData\';
% load([dir 'PropImgs_3Labels4trainingNetwork_comb'])
f = 'Y:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\OtherCode\CNNcode\New Flow\CombinedTrainingData\';
load([f, 'combinedTrainingData'])
%% test transfer training on a small set of imgs
trainImages = cellfun(@(x)cat(3,x,x,x), trainImgsPh,'UniformOutput', false);
trainImgs = cat(4,trainImages{:});

w = size(trainImgs,1); h = size(trainImgs,2);

w1 = 27;
w2 = 27;
slideSize = [w1 w2]; 

numChannels = 3;
startPx = 14;
endPx = 68;
pxSize = length(startPx:endPx)^2;
numFig = size(trainImages,2)-2000;

trainLabels = zeros(pxSize, numFig);
pxCoord = zeros(pxSize,2, numFig);
corners = [startPx, startPx; startPx, endPx; endPx, endPx; endPx, startPx; startPx, startPx];
xv = corners(:,1);
yv = corners(:,2);
% Pick up all contours, inset and the rest randomly pick from bkg
tic
for kk = 1:numFig

    mask = train_fullMasks{kk};
    con2rInd = find(mask == 1);
    [x1,y1] = ind2sub([w h], con2rInd);
    in = inpolygon(x1,y1, xv, yv);
    x1 = x1(in);
    y1 = y1(in);
   
%     bkgIndex = find(mask == 0);
    insetInd = find(mask == 2);
    [x2,y2] = ind2sub([w h], insetInd);
    in = inpolygon(x2,y2, xv, yv);
    x2 = x2(in);
    y2 = y2(in);
    
    numBkg = pxSize - length(x1)- length(x2);
    bkgIndex = find(mask == 0);
    [x0,y0] = ind2sub([w h], bkgIndex);
    in = inpolygon(x0,y0, xv, yv);
    x0 = x0(in);
    y0 = y0(in);
    permX = randperm(length(x0), numBkg)';
    permY = randperm(length(y0), numBkg)';
    x0 = x0(permX);
    y0 = y0(permY);
    
    xx = [x1;x2;x0];
    yy = [y1;y2;y0];
    pxCoord(:,:, kk) = [xx yy];
    for vv = 1:size(xx,1)
        trainLabels(vv,kk) = mask(xx(vv),yy(vv));
    end
end
toc
trainLabels = trainLabels(:);
trainLabs = categorical(trainLabels);

% substract images centered by rand selected pixels 
PropImgs = zeros(w1,w2,numChannels, numFig*pxSize, 'uint16');

tic
for ii = 1:numFig
    img = trainImgs(:,:,:,ii);
    xx = pxCoord(:,1,ii);
    yy = pxCoord(:,2,ii);
    
    PropImgs(:,:,1, 1+(ii-1)*pxSize:pxSize+(ii-1)*pxSize)= multiImCrop(img, xx, yy, (w1-1)/2, (w2-1)/2);  
    PropImgs(:,:,2, 1+(ii-1)*pxSize:pxSize+(ii-1)*pxSize)= multiImCrop(img, xx, yy, (w1-1)/2, (w2-1)/2); 
    PropImgs(:,:,3, 1+(ii-1)*pxSize:pxSize+(ii-1)*pxSize)= multiImCrop(img, xx, yy, (w1-1)/2, (w2-1)/2); 
end
toc

%%
net = vgg16;

lgraph = layerGraph(net.Layers);
figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
plot(lgraph)
%%
layersTransfer = net.Layers(2:18);
numClasses = 3;
imageAugmenter = imageDataAugmenter('RandXReflection', true,...
    'RandYReflection', true,...
    'RandXScale', [1 1.2],...
    'RandYScale', [1 1.2]);
imSize = [size(PropImgs,1), size(PropImgs,2),3]; 
datasource = augmentedImageSource(imSize,PropImgs,trainLabs,'DataAugmentation',imageAugmenter);
inputLayer = imageInputLayer(imSize); 
layers = [
    inputLayer
    layersTransfer
    convolution2dLayer([3,3], 2304, 'name', 'conv4_1')
    dropoutLayer(0.5, 'name', 'drop1_1')
    reluLayer('name', 'relu4_1')
    convolution2dLayer([1,1], 2304, 'name', 'conv4_2')
    dropoutLayer(0.5, 'name', 'drop1_2')
    reluLayer('name', 'relu4_2')
    fullyConnectedLayer(numClasses, 'name','full1_1')
    softmaxLayer('name', 'softmax')
    classificationLayer('name', 'crossentropy')];

miniBatchSize = 128;
% numValidationsPerEpoch = 10;
% validationFrequency = floor(size(trainImages,4)/miniBatchSize/numValidationsPerEpoch);
options = trainingOptions('sgdm',...
    'InitialLearnRate',0.0001,...
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

%%
XuNet = trainNetwork(datasource,layers,options);
% temp_save = 'D:\CNN_temp\Short trains\';
% save([temp_save 'CNN_spores_by3Lables_comb_bigger_s3_10epoch_Vgg16NetTransfer'], 'XuNet')
%% Validation

w1 = 27;
w2 = 27;
slideSize = [w1 w2];
numChannels = 1;
startPx = 14;
endPx = 68;
pxSize = length(startPx:endPx)^2;



classMetric = struct();
Ycontour = cell(1, size(valImgsPh,2));

for testNum =1:size(valImgsPh,2)
    testImg = valImgsPh{testNum};
    [w,h]= size(testImg);
    rect = [];
    for ii = startPx:endPx
        for jj =startPx:endPx
            rect = cat(1, rect, [ii-13, jj-13, 26, 26]);
        end
    end
    testIm = cat(3, testImg, testImg, testImg);
    for kk =1:pxSize
        testIm(:,:,1,kk) = imcrop(testImg, rect(kk,:));
        testIm(:,:,2,kk) = imcrop(testImg, rect(kk,:));
        testIm(:,:,3,kk) = imcrop(testImg, rect(kk,:));
    end
    YTest = classify(XuNet, testIm);
    
    testContour = val_fullMasks{testNum};
    testLabel = zeros(length(slideSize(1):w)^2, length(testNum));
    for kk = 1:length(testNum)
        contour_ = testContour(startPx:endPx, startPx:endPx);
        testLabel(:,kk) = contour_(:);
    end
    testLab = categorical(testLabel(:));
    
    Ycontour{testNum} = reshape(double(YTest), sqrt(size(testLabel,1)),sqrt(size(testLabel,1)));
   
    % Calculate the F1 score.
    [F1, precision, recall,performance,confusion] = classificationMetric(testLab,YTest);
    classMetric(testNum).F1 = F1;
    classMetric(testNum).precision = precision;
    classMetric(testNum).recall = recall;
    classMetric(testNum).performance = performance;
    classMetric(testNum).confisionMat = confusion;
    
end
testImg = cellfun(@(x) x(startPx:endPx, startPx:endPx),  valImgsPh,'UniformOutput',false);
testContour = cellfun(@(x) x(startPx:endPx, startPx:endPx),  val_fullMasks,'UniformOutput',false);
%% Compare with labels and score the prediction
scores = zeros(1,size(testContour,2));
for ii = 1:size(testContour,2)
    figure(ii);
    subplot(1,3,1)
    imagesc(testImg{ii})
    axis square
    title(['phase img sp' num2str(ii)]) 
    subplot(1,3,2)
    imagesc(testContour{ii})
    axis square
    title('true labels')
    subplot(1,3,3)
    imagesc(Ycontour{ii}, 'ButtonDownFcn',@ClickImgStatus)
    axis square
    title('CNN labels')
    waitforbuttonpress
    scores(ii) = get(gcf, 'UserData');
    close gcf   
end

%%
function ClickImgStatus(src,eventdata, ii)
    clickType =  get(gcf,'SelectionType');
    switch clickType
        case 'normal' %left click on the correct image = correct segmented spores
            rectangle('Position', [1, 1, 54, 54], 'EdgeColor', 'green', 'LineWidth',2);
            set(src, 'UserData', 1);
                
        case 'alt' %right click on 1st image = incorrect segmented spores
            rectangle('Position', [1, 1, 54, 54], 'EdgeColor', 'red', 'LineWidth',2);
            set(src, 'UserData', 0);
        
    end       
    set(gcf, 'UserData',src.UserData)
end

