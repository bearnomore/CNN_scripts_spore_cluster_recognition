f = 'Y:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\OtherCode\CNNcode\New Flow\CombinedTrainingData\';
% load([f, 'combinedTrainingData'])
% load([f, 'combinedTrainingData_thickEdge'])
load([f, 'combinedTrainingDataThickEdge'])
%% Import only Manual labeled data
f = 'Y:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\OtherCode\CNNcode\New Flow\CombinedTrainingData\ManualLabeledData\';
load([f 'TrainingDataFromManualLabels.mat'])
trainImgsPh = trainImgsPh_man;
train_fullMasks = train_fullMasks_man;

%% Import only Semiauto labeled data
f = 'Y:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\OtherCode\CNNcode\New Flow\CombinedTrainingData\SemiAutoLabeledData\';
load([f 'TrainingDataFromSemiAutoLabels.mat'])
trainImgsPh = trainImgsPh_semi;
train_fullMasks = train_fullMasks_semi;
%% Generate subimage proposals by sliding a window of 21x21 centering on all pixels in the input image
w = size(trainImgsPh{1},1);
h = size(trainImgsPh{1},2);
% w1 = 21;
w1 = 27;
% w2 = 21;
w2 = 27;
slideSize = [w1 w2]; % a 21x21 window
numChannels = 1;
% startPx = 11;
% endPx = 71;
startPx = 14;
endPx = 68;
pxSize = length(startPx:endPx)^2;
numFig = size(trainImgsPh,2);
% randomly pick up 3000 pixels from each figure 
% trainLabels = zeros(3000, numFig);
% randPx_x = randi([startPx, endPx], 3000, numFig);
% randPx_y = randi([startPx, endPx], 3000, numFig);
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
PropImgs = zeros(w1,w2,1, numFig*pxSize, 'uint16');

tic
for ii = 1:numFig
    img = trainImgsPh{ii};
    xx = pxCoord(:,1,ii);
    yy = pxCoord(:,2,ii);
    
    PropImgs(:,:,1, 1+(ii-1)*pxSize:pxSize+(ii-1)*pxSize)= multiImCrop(img, xx, yy, (w1-1)/2, (w2-1)/2);  
    %PropImgs(:,:,1, 1+(ii-1)*3000:3000+(ii-1)*3000) = bsxfun(@imcrop,...
    %img,xx,yy);
   
end
toc
%%
save_dir = 'Y:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\OtherCode\CNNcode\New Flow\CombinedTrainingData\';
save([save_dir 'PropImgs_thickEdge_3Labels4trainingNetwork'], 'PropImgs', 'trainLabs', '-v7.3')
%%
save_dir = 'Z:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\OtherCode\CNNcode\New Flow\CombinedTrainingData\SemiAutoLabeledData\';
save([save_dir 'PropImgs_3Labels4trainingNetwork_semi'], 'PropImgs', 'trainLabs', '-v7.3')
%%
save_dir = 'Z:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\OtherCode\CNNcode\New Flow\CombinedTrainingData\ManualLabeledData\';
save([save_dir 'PropImgs_3Labels4trainingNetwork_man'], 'PropImgs', 'trainLabs', '-v7.3')
%%
save_dir = 'Y:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\OtherCode\CNNcode\New Flow\CombinedTrainingData\';
save([save_dir 'PropImgs_3Labels4trainingNetwork_comb'], 'PropImgs', 'trainLabs', '-v7.3')
%% Combined bigger cut of imgs
save_dir = 'Y:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\OtherCode\CNNcode\New Flow\CombinedTrainingData\';
save([save_dir 'PropImgs_3Labels4trainingNetwork_comb_bigger'], 'PropImgs', 'trainLabs', '-v7.3')
%%
save_dir = 'Y:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\OtherCode\CNNcode\New Flow\CombinedTrainingData\';
save([save_dir 'PropImgs_thickEdge_3Labels4trainingNetwork_bigger'], 'PropImgs', 'trainLabs', '-v7.3')