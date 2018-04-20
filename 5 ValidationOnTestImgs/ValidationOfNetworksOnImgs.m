%% Load validation data
f = 'Z:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\OtherCode\CNNcode\New Flow\CombinedTrainingData\';
load([f, 'combinedTrainingData'])
%% Load comparing network
f = 'D:\CNN_temp\Short trains\';
% load([f 'CNN_spores_by3Lables_comb_bigger_s3_long'])
% net = XuNet;
% load([f 'CNN_spores_by3Lables_comb_bigger_s3_10epoch_Vgg16NetTransfer'])
load([f 'CNN_spores_by3Lables_comb_bigger_s3_long_convImpOfFilter'])
net = XuNet;
% load([f 'CNN_spores_by3Lables_comb_bigger_s3_15epoch_convImpOfFilter'])
% net3 = XuNet2;
% net = XuNet2;
%% Apply networks to all test figures;
singles =  classfiedValImgs.singlesValPh;
singlesMasks = classfiedValImgs.singlesValFullMasks;
doublets = classfiedValImgs.doubletValPh;
doubletsMasks = classfiedValImgs.doubletValFullMasks;
multiplets = classfiedValImgs.multipletValPh;
multipletsMasks = classfiedValImgs.multipletValFullMasks;

w1 = 27;
w2 = 27;
slideSize = [w1 w2];
numChannels = 1;
startPx = 14;
endPx = 68;
pxSize = length(startPx:endPx)^2;

classMetric1 = struct();
Ycontour1 = cell(1, size(singles,2));
for testNum =1:size(singles,2)
    testImg = singles{testNum};
    [w,h]= size(testImg);
    rect = [];
    for ii = startPx:endPx
        for jj =startPx:endPx
            rect = cat(1, rect, [ii-13, jj-13, 26, 26]);
        end
    end
    testIm = zeros(w1,w2,1, pxSize);
    for kk =1:pxSize
        testIm(:,:,1,kk) = imcrop(testImg, rect(kk,:));
%         testIm(:,:,2,kk) = imcrop(testImg, rect(kk,:));
%         testIm(:,:,3,kk) = imcrop(testImg, rect(kk,:));
    end
    
    YTest1 = classify(net, testIm);
    
    
    testContour = singlesMasks{testNum};
    testLabel = zeros(length(slideSize(1):w)^2, length(testNum));
    for kk = 1:length(testNum)
        contour_ = testContour(startPx:endPx, startPx:endPx);
        testLabel(:,kk) = contour_(:);
    end
    testLab = categorical(testLabel(:));
    
    Ycontour1{testNum} = reshape(double(YTest1), sqrt(size(testLabel,1)),sqrt(size(testLabel,1)));
    % Calculate the F1 score.
    [F1, precision, recall,performance,confusion] = classificationMetric(testLab,YTest1);
    classMetric1(testNum).F1 = F1;
    classMetric1(testNum).precision = precision;
    classMetric1(testNum).recall = recall;
    classMetric1(testNum).performance = performance;
    classMetric1(testNum).confisionMat = confusion;
end

classMetric2 = struct();
Ycontour2 = cell(1, size(doublets,2));
for testNum =1:size(doublets,2)
    testImg = doublets{testNum};
    [w,h]= size(testImg);
    rect = [];
    for ii = startPx:endPx
        for jj =startPx:endPx
            rect = cat(1, rect, [ii-13, jj-13, 26, 26]);
        end
    end
    testIm = zeros(w1,w2,1, pxSize);
    for kk =1:pxSize
        testIm(:,:,1,kk) = imcrop(testImg, rect(kk,:));
%         testIm(:,:,2,kk) = imcrop(testImg, rect(kk,:));
%         testIm(:,:,3,kk) = imcrop(testImg, rect(kk,:));
    end
    YTest2 = classify(net, testIm);
    
    
    testContour = doubletsMasks{testNum};
    testLabe2 = zeros(length(slideSize(1):w)^2, length(testNum));
    for kk = 1:length(testNum)
        contour_ = testContour(startPx:endPx, startPx:endPx);
        testLabe2(:,kk) = contour_(:);
    end
    testLab = categorical(testLabe2(:));
    
    Ycontour2{testNum} = reshape(double(YTest2), sqrt(size(testLabe2,1)),sqrt(size(testLabe2,1)));
    % Calculate the F1 score.
    [F1, precision, recall,performance,confusion] = classificationMetric(testLab,YTest1);
    classMetric2(testNum).F1 = F1;
    classMetric2(testNum).precision = precision;
    classMetric2(testNum).recall = recall;
    classMetric2(testNum).performance = performance;
    classMetric2(testNum).confisionMat = confusion;
end

classMetric3 = struct();
Ycontour3 = cell(1, size(multiplets,2));
for testNum =1:size(multiplets,2)
    testImg = multiplets{testNum};
    [w,h]= size(testImg);
    rect = [];
    for ii = startPx:endPx
        for jj =startPx:endPx
            rect = cat(1, rect, [ii-13, jj-13, 26, 26]);
        end
    end
    testIm = zeros(w1,w2,1, pxSize);
    for kk =1:pxSize
        testIm(:,:,1,kk) = imcrop(testImg, rect(kk,:));
%         testIm(:,:,2,kk) = imcrop(testImg, rect(kk,:));
%         testIm(:,:,3,kk) = imcrop(testImg, rect(kk,:));
    end
    YTest3 = classify(net, testIm);
    
    testContour = val_fullMasks{testNum};
    testLabe3 = zeros(length(slideSize(1):w)^2, length(testNum));
    for kk = 1:length(testNum)
        contour_ = testContour(startPx:endPx, startPx:endPx);
        testLabel(:,kk) = contour_(:);
    end
    testLab = categorical(testLabel(:));

    Ycontour3{testNum} = reshape(double(YTest3), sqrt(size(testLabel,1)),sqrt(size(testLabel,1)));
    % Calculate the F1 score.
    [F1, precision, recall,performance,confusion] = classificationMetric(testLab,YTest3);
    classMetric3(testNum).F1 = F1;
    classMetric3(testNum).precision = precision;
    classMetric3(testNum).recall = recall;
    classMetric3(testNum).performance = performance;
    classMetric3(testNum).confisionMat = confusion;
end
testImg1 = cellfun(@(x) x(startPx:endPx, startPx:endPx),  singles,'UniformOutput',false);
testContour1 = cellfun(@(x) x(startPx:endPx, startPx:endPx),  singlesMasks,'UniformOutput',false);

testImg2 = cellfun(@(x) x(startPx:endPx, startPx:endPx),  doublets,'UniformOutput',false);
testContour2 = cellfun(@(x) x(startPx:endPx, startPx:endPx),  doubletsMasks,'UniformOutput',false);

testImg3 = cellfun(@(x) x(startPx:endPx, startPx:endPx),  multiplets,'UniformOutput',false);
testContour3 = cellfun(@(x) x(startPx:endPx, startPx:endPx),  multipletsMasks,'UniformOutput',false);
%% Compare with labels and score the prediction
scores1 = zeros(1,size(testContour1,2));
for ii = 1:size(testContour1,2)
    figure(ii);
    subplot(1,3,1)
    imagesc(testImg1{ii})
    axis square
    title(['phase img sp' num2str(ii)]) 
    subplot(1,3,2)
    imagesc(testContour1{ii})
    axis square
    title('true labels')
    subplot(1,3,3)
    imagesc(Ycontour1{ii}, 'ButtonDownFcn',@ClickImgStatus)
    axis square
    title('CNN labels')
    waitforbuttonpress
    scores1(ii) = get(gcf, 'UserData');
    close gcf   
end
%%
scores2 = zeros(1,size(testContour2,2));
for ii = 1:size(testContour2,2)
    figure(ii);
    subplot(1,3,1)
    imagesc(testImg2{ii})
    axis square
    title(['phase img sp' num2str(ii)]) 
    subplot(1,3,2)
    imagesc(testContour2{ii})
    axis square
    title('true labels')
    subplot(1,3,3)
    imagesc(Ycontour2{ii}, 'ButtonDownFcn',@ClickImgStatus)
    axis square
    title('CNN labels')
    waitforbuttonpress
    scores2(ii) = get(gcf, 'UserData');
    close gcf   
end
%%
scores3 = zeros(1,size(testContour3,2));
for ii = 1:size(testContour3,2)
    figure(ii);
    subplot(1,3,1)
    imagesc(testImg3{ii})
    axis square
    title(['phase img sp' num2str(ii)]) 
    subplot(1,3,2)
    imagesc(testContour3{ii})
    axis square
    title('true labels')
    subplot(1,3,3)
    imagesc(Ycontour3{ii}, 'ButtonDownFcn',@ClickImgStatus)
    axis square
    title('CNN labels')
    waitforbuttonpress
    scores3(ii) = get(gcf, 'UserData');
    close gcf   
end
%%
accuracy_seg1 = sum(scores1)/numel(scores1);
accuracy_seg2 = sum(scores2)/numel(scores2);
accuracy_seg3 = sum(scores3)/numel(scores3);

save_dir = 'D:\CNN_temp\Short trains\val on s3\';
% save([save_dir 'performance of comb_bigger_s3_long_convImpOfFilter'], 'accuracy_seg1', 'accuracy_seg2','accuracy_seg3')
% save([save_dir 'performance of comb_bigger_s3_10epoch_Vgg16NetTransfer'], 'accuracy_seg1', 'accuracy_seg2','accuracy_seg3')
% save([save_dir 'performance of comb_bigger_s3_15epoch_convImpOfFilter'], 'accuracy_seg1', 'accuracy_seg2','accuracy_seg3')
save([save_dir 'performance of comb_bigger_s3_long'], 'accuracy_seg1', 'accuracy_seg2','accuracy_seg3')

%% Check on training data accuracy
% Apply networks to training figures;
singles =  classfiedTrainImgs.singlesTrainPh;
singlesMasks = classfiedTrainImgs.singlesTrainFullMasks;
doublets = classfiedTrainImgs.doubletTrainPh;
doubletsMasks = classfiedTrainImgs.doubletTrainFullMasks;
multiplets = classfiedTrainImgs.multipletTrainPh;
multipletsMasks = classfiedTrainImgs.multipletTrainFullMasks;

w1 = 27;
w2 = 27;
slideSize = [w1 w2];
numChannels = 1;
startPx = 14;
endPx = 68;
pxSize = length(startPx:endPx)^2;

classMetric1 = struct();
Ycontour1 = cell(1, size(singles,2));
for testNum =1:size(singles,2)
    testImg = singles{testNum};
    [w,h]= size(testImg);
    rect = [];
    for ii = startPx:endPx
        for jj =startPx:endPx
            rect = cat(1, rect, [ii-13, jj-13, 26, 26]);
        end
    end
    testIm = zeros(w1,w2,1, pxSize);
    for kk =1:pxSize
        testIm(:,:,1,kk) = imcrop(testImg, rect(kk,:));
%         testIm(:,:,2,kk) = imcrop(testImg, rect(kk,:));
%         testIm(:,:,3,kk) = imcrop(testImg, rect(kk,:));
    end
    
    YTest1 = classify(net, testIm);
    
    
    testContour = singlesMasks{testNum};
    testLabel = zeros(length(slideSize(1):w)^2, length(testNum));
    for kk = 1:length(testNum)
        contour_ = testContour(startPx:endPx, startPx:endPx);
        testLabel(:,kk) = contour_(:);
    end
    testLab = categorical(testLabel(:));
    
    Ycontour1{testNum} = reshape(double(YTest1), sqrt(size(testLabel,1)),sqrt(size(testLabel,1)));
    % Calculate the F1 score.
    [F1, precision, recall,performance,confusion] = classificationMetric(testLab,YTest1);
    classMetric1(testNum).F1 = F1;
    classMetric1(testNum).precision = precision;
    classMetric1(testNum).recall = recall;
    classMetric1(testNum).performance = performance;
    classMetric1(testNum).confisionMat = confusion;
end

classMetric2 = struct();
Ycontour2 = cell(1, size(doublets,2));
for testNum =1:size(doublets,2)
    testImg = doublets{testNum};
    [w,h]= size(testImg);
    rect = [];
    for ii = startPx:endPx
        for jj =startPx:endPx
            rect = cat(1, rect, [ii-13, jj-13, 26, 26]);
        end
    end
    testIm = zeros(w1,w2,1, pxSize);
    for kk =1:pxSize
        testIm(:,:,1,kk) = imcrop(testImg, rect(kk,:));
%         testIm(:,:,2,kk) = imcrop(testImg, rect(kk,:));
%         testIm(:,:,3,kk) = imcrop(testImg, rect(kk,:));
    end
    YTest2 = classify(net, testIm);
    
    
    testContour = doubletsMasks{testNum};
    testLabe2 = zeros(length(slideSize(1):w)^2, length(testNum));
    for kk = 1:length(testNum)
        contour_ = testContour(startPx:endPx, startPx:endPx);
        testLabe2(:,kk) = contour_(:);
    end
    testLab = categorical(testLabe2(:));
    
    Ycontour2{testNum} = reshape(double(YTest2), sqrt(size(testLabe2,1)),sqrt(size(testLabe2,1)));
    % Calculate the F1 score.
    [F1, precision, recall,performance,confusion] = classificationMetric(testLab,YTest1);
    classMetric2(testNum).F1 = F1;
    classMetric2(testNum).precision = precision;
    classMetric2(testNum).recall = recall;
    classMetric2(testNum).performance = performance;
    classMetric2(testNum).confisionMat = confusion;
end

classMetric3 = struct();
Ycontour3 = cell(1, size(multiplets,2));
for testNum =1:size(multiplets,2)
    testImg = multiplets{testNum};
    [w,h]= size(testImg);
    rect = [];
    for ii = startPx:endPx
        for jj =startPx:endPx
            rect = cat(1, rect, [ii-13, jj-13, 26, 26]);
        end
    end
    testIm = zeros(w1,w2,1, pxSize);
    for kk =1:pxSize
        testIm(:,:,1,kk) = imcrop(testImg, rect(kk,:));
%         testIm(:,:,2,kk) = imcrop(testImg, rect(kk,:));
%         testIm(:,:,3,kk) = imcrop(testImg, rect(kk,:));
    end
    YTest3 = classify(net, testIm);
    
    testContour = train_fullMasks{testNum};
    testLabe3 = zeros(length(slideSize(1):w)^2, length(testNum));
    for kk = 1:length(testNum)
        contour_ = testContour(startPx:endPx, startPx:endPx);
        testLabel(:,kk) = contour_(:);
    end
    testLab = categorical(testLabel(:));

    Ycontour3{testNum} = reshape(double(YTest3), sqrt(size(testLabel,1)),sqrt(size(testLabel,1)));
    % Calculate the F1 score.
    [F1, precision, recall,performance,confusion] = classificationMetric(testLab,YTest3);
    classMetric3(testNum).F1 = F1;
    classMetric3(testNum).precision = precision;
    classMetric3(testNum).recall = recall;
    classMetric3(testNum).performance = performance;
    classMetric3(testNum).confisionMat = confusion;
end
testImg1 = cellfun(@(x) x(startPx:endPx, startPx:endPx),  singles,'UniformOutput',false);
testContour1 = cellfun(@(x) x(startPx:endPx, startPx:endPx),  singlesMasks,'UniformOutput',false);

testImg2 = cellfun(@(x) x(startPx:endPx, startPx:endPx),  doublets,'UniformOutput',false);
testContour2 = cellfun(@(x) x(startPx:endPx, startPx:endPx),  doubletsMasks,'UniformOutput',false);

testImg3 = cellfun(@(x) x(startPx:endPx, startPx:endPx),  multiplets,'UniformOutput',false);
testContour3 = cellfun(@(x) x(startPx:endPx, startPx:endPx),  multipletsMasks,'UniformOutput',false);

%% Compare with labels and score the prediction for the training imgs
max_fig = get(0, 'ScreenSize');
scores1 = zeros(1,size(testContour1,2));
%%
for ii = 1:size(testContour1,2)
    figure(ii); set(gcf, 'Position', max_fig);
    subplot(1,3,1)
    imagesc(testImg1{ii})
    axis square
    title(['phase img sp' num2str(ii)]) 
    subplot(1,3,2)
    imagesc(testContour1{ii})
    axis square
    title('true labels')
    subplot(1,3,3)
    imagesc(Ycontour1{ii}, 'ButtonDownFcn',@ClickImgStatus)
    axis square
    title('CNN labels')
    waitforbuttonpress
    scores1(ii) = get(gcf, 'UserData');
    close gcf   
end
singlet_accuracy_train = sum(scores1)/numel(scores1);
%%
scores2 = zeros(1,size(testContour2,2));
%%
for ii = 1164:size(testContour2,2)
    figure(ii);set(gcf, 'Position', max_fig);
    subplot(1,3,1)
    imagesc(testImg2{ii})
    axis square
    title(['phase img sp' num2str(ii)]) 
    subplot(1,3,2)
    imagesc(testContour2{ii})
    axis square
    title('true labels')
    subplot(1,3,3)
    imagesc(Ycontour2{ii}, 'ButtonDownFcn',@ClickImgStatus)
    axis square
    title('CNN labels')
    waitforbuttonpress
    scores2(ii) = get(gcf, 'UserData');
    close gcf   
end
doublet_accuracy_train = sum(scores2)/numel(scores2);
%%
scores3 = zeros(1,size(testContour3,2));
%%
for ii = 1:size(testContour3,2)
    figure(ii);set(gcf, 'Position', max_fig);
    subplot(1,3,1)
    imagesc(testImg3{ii})
    axis square
    title(['phase img sp' num2str(ii)]) 
    subplot(1,3,2)
    imagesc(testContour3{ii})
    axis square
    title('true labels')
    subplot(1,3,3)
    imagesc(Ycontour3{ii}, 'ButtonDownFcn',@ClickImgStatus)
    axis square
    title('CNN labels')
    waitforbuttonpress
    scores3(ii) = get(gcf, 'UserData');
    close gcf   
end
multiplet_accuracy_train = sum(scores3)/numel(scores3);
%%
save_dir = 'D:\CNN_temp\Short trains\val on s3\';
% save([save_dir 'performance of comb_bigger_s3_long_convImpOfFilter'], 'accuracy_seg1', 'accuracy_seg2','accuracy_seg3')
% save([save_dir 'performance of comb_bigger_s3_10epoch_Vgg16NetTransfer'], 'accuracy_seg1', 'accuracy_seg2','accuracy_seg3')
% save([save_dir 'performance of comb_bigger_s3_15epoch_convImpOfFilter'], 'accuracy_seg1', 'accuracy_seg2','accuracy_seg3')
% save([save_dir 'performance of comb_bigger_s3_long'], 'singlet_accuracy_train', 'doublet_accuracy_train','multiplet_accuracy_train', '-append')

%%
function ClickImgStatus(src,eventdata, ii)
    clickType =  get(gcf,'SelectionType');
    switch clickType
        case 'normal' %left click on the correct image = correct segmented spores
            rectangle('Position', [1, 1, 54, 54], 'EdgeColor', 'green', 'LineWidth',2);
            set(src, 'UserData', 1);
                
        case 'alt' %right click on the image = incorrect segmented spores
            rectangle('Position', [1, 1, 54, 54], 'EdgeColor', 'red', 'LineWidth',2);
            set(src, 'UserData', 0);
        
    end       
    set(gcf, 'UserData',src.UserData)
end


