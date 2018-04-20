%% Load validation data
f = 'Y:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\OtherCode\CNNcode\New Flow\CombinedTrainingData\';
load([f, 'combinedTrainingData'])
%% Load short trained network
f = 'D:\CNN_temp\Short trains\';
% load([f 'CNN_spores_by3Lables_comb_bigger_s1'])
% load([f 'CNN_spores_by3Lables_comb_bigger_s2'])
load([f 'CNN_spores_by3Lables_comb_bigger_s3'])
% load([f 'CNN_spores_by3Lables_comb_bigger_s4'])
% load([f 'CNN_spores_by3Lables_comb_bigger_s5'])
% load([f 'CNN_spores_by3Lables_comb_bigger_s6'])
% load([f 'CNN_spores_by3Lables_comb_bigger_s7'])
% load([f 'CNN_spores_by3Lables_comb_bigger_s8'])
% load([f 'CNN_spores_by3Lables_comb_bigger_s9'])
% load([f 'CNN_spores_by3Lables_comb_bigger_s10'])
%%
% Pick up a few training data set and compare the border classification
% accuracy on train with that on validation(dev)set
train_valIds = randperm(size(trainImgsPh,2), size(valImgsPh,2));
valImgsPh_tr = trainImgsPh(train_valIds);
val_fullMasks_tr = train_fullMasks(train_valIds);

%%
w1 = 27;
w2 = 27;
slideSize = [w1 w2];
numChannels = 1;
startPx = 14;
endPx = 68;
pxSize = length(startPx:endPx)^2;

classMetric_ValTr = struct();
classMetric_Val = struct();
Ycontour_ValTr = cell(1, size(valImgsPh_tr ,2));
Ycontour_Val = cell(1, size(valImgsPh ,2));
for testNum =1:size(valImgsPh_tr,2)
    testImg_tr = valImgsPh_tr{testNum};
    testImg = valImgsPh{testNum};
    [w,h]= size(testImg);
    rect = [];
    for ii = startPx:endPx
        for jj =startPx:endPx
%             rect = cat(1, rect, [ii-10, jj-10, 20, 20]);
            rect = cat(1, rect, [ii-13, jj-13, 26, 26]);
        end
    end
    testIm_tr = zeros(w1,w2,1, pxSize);
    testIm = zeros(w1,w2,1, pxSize);
    for kk =1:pxSize
        testIm_tr(:,:,1,kk) = imcrop(testImg_tr, rect(kk,:));
        testIm(:,:,1,kk) = imcrop(testImg, rect(kk,:));
    end
    YTest_tr = classify(XuNet, testIm_tr);
    YTest = classify(XuNet, testIm);
    
    testContour_tr = val_fullMasks_tr{testNum};
    testContour = val_fullMasks{testNum};
    
    testLabel_tr = zeros(length(slideSize(1):w)^2, length(testNum));
    testLabel = zeros(length(slideSize(1):w)^2, length(testNum));
    for kk = 1:length(testNum)
        contour_tr = testContour_tr(startPx:endPx, startPx:endPx);
        testLabel_tr(:,kk) = contour_tr(:);
        
        contour = testContour(startPx:endPx, startPx:endPx);
        testLabel(:,kk) = contour(:);
    end
    testLab_tr = categorical(testLabel_tr(:));
    testLab = categorical(testLabel(:));
    
    Ycontour_ValTr{testNum} = reshape(double(YTest_tr), sqrt(size(testLabel_tr,1)),sqrt(size(testLabel_tr,1)));
    Ycontour_Val{testNum} = reshape(double(YTest), sqrt(size(testLabel,1)),sqrt(size(testLabel,1)));

    % Calculate the confusion mat 
    [F1, precision, recall,performance,confusion] = classificationMetric(testLab_tr,YTest_tr);
    classMetric_ValTr(testNum).F1 = F1;
    classMetric_ValTr(testNum).precision = precision;
    classMetric_ValTr(testNum).recall = recall;
    classMetric_ValTr(testNum).performance = performance;
    classMetric_ValTr(testNum).confusionMat = confusion;
    
    [F1, precision, recall,performance,confusion] = classificationMetric(testLab,YTest);
    classMetric_Val(testNum).F1 = F1;
    classMetric_Val(testNum).precision = precision;
    classMetric_Val(testNum).recall = recall;
    classMetric_Val(testNum).performance = performance;
    classMetric_Val(testNum).confusionMat = confusion;
end

%%
% check the accuracy of border pixel classification 
Cmat_tr = cat(3, classMetric_ValTr.confusionMat);
Cmat_val = cat(3, classMetric_Val.confusionMat);
Bkg_tr_true = Cmat_tr(1,1,:);
Border_tr_true = Cmat_tr(2,2,:);
Inside_tr_true = Cmat_tr(3,3,:);
tot_tr = sum(Cmat_tr,2);
accu_Bkg_tr = Bkg_tr_true./tot_tr(1,:,:); accu_Bkg_tr = accu_Bkg_tr(:);
accu_Border_tr = Border_tr_true./tot_tr(2,:,:); accu_Border_tr = accu_Border_tr(:);
accu_Inside_tr = Inside_tr_true./tot_tr(3,:,:); accu_Inside_tr = accu_Inside_tr(:);

Bkg_val_true = Cmat_val(1,1,:);
Border_val_true = Cmat_val(2,2,:);
Inside_val_true = Cmat_val(3,3,:);
tot_val = sum(Cmat_val,2);
accu_Bkg_val = Bkg_val_true./tot_val(1,:,:); accu_Bkg_val = accu_Bkg_val(:);
accu_Border_val = Border_val_true./tot_val(2,:,:); accu_Border_val = accu_Border_val(:);
accu_Inside_val = Inside_val_true./tot_val(3,:,:); accu_Inside_val = accu_Inside_val(:);

%% Plot and compare the accuracy between the train and val data sets
mean_accu_tr_bkg =  mean(accu_Bkg_tr);
mean_accu_tr_border =  mean(accu_Border_tr);
mean_accu_tr_inside =  mean(accu_Inside_tr);

mean_accu_val_bkg =  mean(accu_Bkg_val);
mean_accu_val_border =  mean(accu_Border_val);
mean_accu_val_inside =  mean(accu_Inside_val);

figure; 
h1 = histogram(accu_Bkg_tr, 'facealpha',.5); hold on
h2 = histogram(accu_Bkg_val, 'facealpha',.5); 
legend([h1 h2], 'train', 'val','Location','NorthWest' )
title('Bkg Pixel accuracy comparison')

figure; 
h1 = histogram(accu_Border_tr, 'facealpha',.5); hold on
h2 = histogram(accu_Border_val, 'facealpha',.5); 
legend([h1 h2], 'train', 'val','Location','NorthWest' )
title('Border Pixel accuracy comparison')

figure; 
h1 = histogram(accu_Inside_tr, 'facealpha',.5); hold on
h2 = histogram(accu_Inside_val, 'facealpha',.5); 
legend([h1 h2], 'train', 'val','Location','NorthWest' )
title('Inside Pixel accuracy comparison')
%%
% Check the figures of least accurately classified border
[sort_border_tr, Index_tr] = sort(accu_Border_tr, 'ascend');
[sort_border_val, Index_val] = sort(accu_Border_val, 'ascend');

ind_tr = Index_tr(1:16);
ind_val = Index_val(1:16);

for ii = 1:length(ind_tr)
    ImgTr = valImgsPh_tr{ind_tr(ii)}(startPx:endPx, startPx:endPx);
    ImgVal = valImgsPh{ind_val(ii)}(startPx:endPx, startPx:endPx);
    ContourTr = val_fullMasks_tr{ind_tr(ii)}(startPx:endPx, startPx:endPx);
    ContourVal = val_fullMasks{ind_val(ii)}(startPx:endPx, startPx:endPx);
    
    CNNcontour_tr = Ycontour_ValTr{ind_tr(ii)};
    CNNcontour_val = Ycontour_Val{ind_val(ii)};
    
    figure(1); colormap(gray)
    subplot(4,4,ii)
    imagesc(ImgTr)
    axis square
    figure(2); colormap(gray)
    subplot(4,4,ii)
    imagesc(ContourTr)
    axis square
    figure(3); colormap(gray)
    subplot(4,4,ii)
    imagesc(CNNcontour_tr)
    axis square
    
    figure(4); colormap(gray)
    subplot(4,4,ii)
    imagesc(ImgVal)
    axis square
    figure(5); colormap(gray)
    subplot(4,4,ii)
    imagesc(ContourVal)
    axis square
    figure(6); colormap(gray)
    subplot(4,4,ii)
    imagesc(CNNcontour_val)
    axis square
    
end
%% 
% Manually Check the accuracy of segmentation of train set and val set
score_TrVal = zeros(1,size(valImgsPh,2));
score_Val = zeros(1,size(valImgsPh,2));
%%
for ii = 224:size(valImgsPh,2)
    ImgTr = valImgsPh_tr{ii}(startPx:endPx, startPx:endPx);
    ImgVal = valImgsPh{ii}(startPx:endPx, startPx:endPx);
%     ContourTr = val_fullMasks_tr{ii}(startPx:endPx, startPx:endPx);
%     ContourVal = val_fullMasks{ii}(startPx:endPx, startPx:endPx);
    
    CNNcontour_tr = Ycontour_ValTr{ii};
    CNNcontour_val = Ycontour_Val{ii};
    
    figure(1);
    subplot(1,2,1)
    imagesc(ImgTr);axis image; axis off
    subplot(1,2,2)
    h1=imagesc(CNNcontour_tr); axis image; axis off; hold on  
    set(h1, 'UserData', 0); 
    set(h1,'ButtonDownFcn',{@ClickCNNSegmentedSp, h1});
    waitfor(h1,'UserData')
    score_TrVal(ii) = get(h1, 'UserData');
    close(gcf);
    
    
    figure(2);
    subplot(1,2,1)
    imagesc(ImgVal); axis image; axis off
    subplot(1,2,2)
    h2 = imagesc(CNNcontour_val); axis image; axis off; hold on    
    set(h2, 'UserData', 0); 
    set(h2,'ButtonDownFcn',{@ClickCNNSegmentedSp, h2});
    waitfor(h2,'UserData')
    score_Val(ii) = get(h2, 'UserData');
    close(gcf);
    
end
accuracy_tr = sum(score_TrVal==1)/numel(score_TrVal);
accuracy_val = sum(score_Val==1)/numel(score_Val);
%% Save the metric for neural net work
% shortTrainDir = 'D:\CNN_temp\Short trains\';
% nn = 6;
% save([shortTrainDir 'val on s' num2str(nn) '\MetricComparisonBetweenTrainAndVal'],'classMetric_ValTr','classMetric_Val',...
%     'Ycontour_ValTr', 'Ycontour_Val','accuracy_tr','accuracy_val','score_TrVal','score_Val' )  