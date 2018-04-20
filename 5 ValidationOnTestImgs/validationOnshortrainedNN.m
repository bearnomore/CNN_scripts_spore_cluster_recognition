%% Validation on short trained network to adjust hyper parameters and other factors
%% Load validation data
f = 'Y:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\OtherCode\CNNcode\New Flow\CombinedTrainingData\';
load([f, 'combinedTrainingData'])
%% Load short trained network
f = 'D:\CNN_temp\Short trains\';
% load([f 'CNN_spores_by3Lables_comb_bigger_s1'])
% load([f 'CNN_spores_by3Lables_comb_bigger_s2'])
% load([f 'CNN_spores_by3Lables_comb_bigger_s3'])
% load([f 'CNN_spores_by3Lables_comb_bigger_s4'])
% load([f 'CNN_spores_by3Lables_comb_bigger_s5'])
% load([f 'CNN_spores_by3Lables_comb_bigger_s6'])
% load([f 'CNN_spores_by3Lables_comb_bigger_s7'])
%load([f 'CNN_spores_by3Lables_comb_bigger_s8'])
% load([f 'CNN_spores_by3Lables_comb_bigger_s9'])
% load([f 'CNN_spores_by3Lables_comb_bigger_s9'])
% load([f 'CNN_spores_by3Lables_comb_bigger_s3_long'])
load([f 'CNN_spores_by3Lables_comb_bigger_s3_long_convImpOfFilter'])
%% Find accuracy of all test figures;
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
%             rect = cat(1, rect, [ii-10, jj-10, 20, 20]);
            rect = cat(1, rect, [ii-13, jj-13, 26, 26]);
        end
    end
    testIm = zeros(w1,w2,1, pxSize);
    for kk =1:pxSize
        testIm(:,:,1,kk) = imcrop(testImg, rect(kk,:));
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

% short_version = s3;

% save_dir = ['D:\CNN_temp\Short trains\val on s' num2str(short_version) '\'];
% save([save_dir 'Classification Metric by s' num2str(short_version)], 'classMetric', 'Ycontour')
save_dir = 'D:\CNN_temp\Short trains\val on s3\';
save([save_dir 'Classification Metric by s3_long_conImple_filter'], 'classMetric', 'Ycontour')

%%
% check the performance
% testNum = randi(size(valImgsPh,2),1);
testNum =271;
testImg = valImgsPh{testNum}(startPx:endPx, startPx:endPx);
testContour = val_fullMasks{testNum}(startPx:endPx, startPx:endPx);
CNNcontour = Ycontour{testNum};

figure; 
subplot(1,3,1)
imagesc(testImg)
axis square
title(['phase img sp' num2str(testNum)]) 
subplot(1,3,2)
imagesc(testContour)
axis square
title('true labels')
subplot(1,3,3)
imagesc(CNNcontour)
axis square
title('CNN labels')
%% Plot F1 precision and recall for bkg, border and inside pixels

 F1 = cat(1,classMetric.F1);
 Precision = cat(1,classMetric.precision);   
 Recall = cat(1, classMetric.recall);
 Performance = cat(1, classMetric.performance);

 figure; 
 h1 = histogram(F1(:,1),'facealpha',.8,'edgecolor','none')
 hold on
 h2 = histogram(F1(:,2),'facealpha',.8,'edgecolor','none')
 hold on
 h3 = histogram(F1(:,3),'facealpha',.8,'edgecolor','none')
 axis tight
 legend([h1 h2 h3], 'bkg', 'border', 'inside')
 legend boxoff
 xlabel('F1 score')
 title('F1 scores of bkg border and inside pixels by current CNN')
 
 figure; 
 h1 = histogram(Precision(:,1),'facealpha',.8,'edgecolor','none')
 hold on
 h2 = histogram(Precision(:,2),'facealpha',.8,'edgecolor','none')
 hold on
 h3 = histogram(Precision(:,3),'facealpha',.8,'edgecolor','none')
 axis tight
 legend([h1 h2 h3], 'bkg', 'border', 'inside')
 legend boxoff
 xlabel('Precision percentage')
 title('Precision of bkg border and inside pixels by current CNN')
 
 figure;
 h1 = histogram(Recall(:,1),'facealpha',.8,'edgecolor','none')
 hold on
 h2 = histogram(Recall(:,2),'facealpha',.8,'edgecolor','none')
 hold on
 h3 = histogram(Recall(:,3),'facealpha',.8,'edgecolor','none')
 axis tight
 legend([h1 h2 h3], 'bkg', 'border', 'inside')
 legend boxoff
 xlabel('Recall percentage')
 title('Recall of bkg border and inside pixels by current CNN')
 
 figure;
 plot(Performance,'ko')
 ylabel('Preformance of CNN')
 title(['Average performance = ', num2str(mean(Performance))])
 %% find the worst F1 scored border images
[sortF1_border, Index] = sort(F1(:,2), 'ascend');
testNum = Index(1:16);
for ii = 1:size(testNum,1)
    
    testImg = valImgsPh{testNum(ii)}(startPx:endPx, startPx:endPx);
    testContour = val_fullMasks{testNum(ii)}(startPx:endPx, startPx:endPx);
    CNNcontour = Ycontour{testNum(ii)};
    figure(1); colormap(gray)
    subplot(4,4,ii)
    imagesc(testImg)
    axis square
    figure(2); colormap(gray)
    subplot(4,4,ii)
    imagesc(testContour)
    axis square
    figure(3); colormap(gray)
    subplot(4,4,ii)
    imagesc(CNNcontour)
    axis square
end

%% find the images of worst Performace 
[sortPerformance, Index] = sort(Performance, 'ascend');
testNum = Index(1:16);
for ii = 1:size(testNum,1)
    
    testImg = valImgsPh{testNum(ii)}(startPx:endPx, startPx:endPx);
    testContour = val_fullMasks{testNum(ii)}(startPx:endPx, startPx:endPx);
    CNNcontour = Ycontour{testNum(ii)};
    figure(1); colormap(gray)
    subplot(4,4,ii)
    imagesc(testImg)
    axis square
    figure(2); colormap(gray)
    subplot(4,4,ii)
    imagesc(testContour)
    axis square
    figure(3); colormap(gray)
    subplot(4,4,ii)
    imagesc(CNNcontour)
    axis square
end