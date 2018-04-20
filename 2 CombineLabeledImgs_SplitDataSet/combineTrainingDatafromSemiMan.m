
dir = 'Y:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\OtherCode\CNNcode\New Flow\AllInputImgDataFromManualMarkedFigs\';
load([dir 'InputImgData'])
man_data = InputImgData;
man_masks = [man_data.separateMasks];
man_con2rs = [man_data.contourLab];
man_inputFigs = [man_data.inputFig]; 


inside_maskLab_man = cell(1, length(man_con2rs));
full_Lab_man = cell(1, length(man_con2rs));
for ii = 1:length(man_con2rs)
    mask = man_masks{ii};
    comb_mask = sum(mask,3);
    con2r = man_con2rs{ii};
%     con2r_dilate = imdilate(con2r, se);
    insideLab = xor(comb_mask, con2r); 
%     insideLab = xor(comb_mask, con2r_dilate);  
    full = uint8(insideLab);
    full(insideLab)=2;
    full(con2r==1) = 1;
%     full(con2r_dilate==1) = 1;
    inside_maskLab_man{ii} = uint8(insideLab);
    full_Lab_man{ii} = full;
end

%% for thick edge generation:
se = strel('disk',1,4);% this will add 1 pixel on both sides of the original con2r, and 1 pixel at the boundary of mask
inside_maskLab_man = cell(1, length(man_con2rs));
full_Lab_man = cell(1, length(man_con2rs));
for ii = 1:length(man_con2rs)
    
    con2r = man_con2rs{ii};
    con2r_dilate = imdilate(con2r, se);
    
    mask = man_masks{ii};
    comb_mask = sum(mask,3);
    comb_mask(comb_mask>0) = 1;
    
    mask_dilate = imdilate(comb_mask,se);
    mask_dilate(mask_dilate>0) = 1;
    
    insideLab_dilate = xor(mask_dilate, con2r_dilate); 

    full = uint8(comb_mask);
    full(insideLab_dilate)=2;% eventually generate 2 pixel con2rs (erase 
    
    inside_maskLab_man{ii} = uint8(insideLab_dilate);
    full_Lab_man{ii} = full;
end


% Check the full_Lab_man
figure; colormap(gray); n = 80;
for ii = 1:9
    subplot(3,3,ii); imagesc(full_Lab_man{ii+9*n})
end
%%
% Split the dataset into train and test
trainSize_man = round(length(man_inputFigs)*0.9);
trainIDs_man = randperm(length(man_inputFigs), trainSize_man);
testIDs_man = setdiff(1:length(man_inputFigs), trainIDs_man);

trainImgsPh_man = man_inputFigs(:,trainIDs_man,1);
trainImgsdT_man = man_inputFigs(:,trainIDs_man,2);
train_contours_man = man_con2rs(trainIDs_man);
train_insideLabs_man = inside_maskLab_man(trainIDs_man);
train_fullMasks_man = full_Lab_man(trainIDs_man);

testImgsPh_man = man_inputFigs(:,testIDs_man,1);
testImgsdT_man = man_inputFigs(:,testIDs_man,2);
test_contours_man = man_con2rs(testIDs_man);
test_insideLabs_man = inside_maskLab_man(testIDs_man);
test_fullMasks_man = full_Lab_man(testIDs_man);

% dir_save = 'Y:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\OtherCode\CNNcode\New Flow\CombinedTrainingData\ManualLabeledData\';
% save([dir_save 'TrainingDataFromManualLabels.mat'],'trainImgsPh_man', 'trainImgsdT_man', 'train_contours_man',...
%     'train_insideLabs_man', 'train_fullMasks_man','testImgsPh_man','testImgsdT_man','test_contours_man',...
%     'test_insideLabs_man','test_fullMasks_man', '-v7.3')

dir_save = 'Y:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\OtherCode\CNNcode\New Flow\CombinedTrainingData\ManualLabeledData\';
save([dir_save 'TrainingDataFromManualLabelsThickEdge.mat'],'trainImgsPh_man', 'trainImgsdT_man', 'train_contours_man',...
    'train_insideLabs_man', 'train_fullMasks_man','testImgsPh_man','testImgsdT_man','test_contours_man',...
    'test_insideLabs_man','test_fullMasks_man', '-v7.3')

%%
d = 'Z:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\OtherCode\CNNcode\New Flow\AllInputImgDataFromSemiAutoMarkedFigs\';
load([d 'InputImgData'])
semi_data = InputImgData;
semi_con2rs = [semi_data.contourLab];
semi_masks = [semi_data.maskLab];
semi_inputFigs = [semi_data.inputFig];

inside_maskLab_semi = cell(1, length(semi_con2rs));
full_Lab_semi = cell(1, length(semi_con2rs));
for ii = 1:length(semi_con2rs)
    con2r = semi_con2rs{ii};
%     con2r_dilate = imdilate(con2r, se);
%     whole_mask = imfill(con2r, 'holes');
%     whole_mask = imfill(con2r_dilate, 'holes');
    mask = semi_masks{ii};
%     insideLab  = xor(whole_mask, con2r);    
%     insideLab  = xor(whole_mask, con2r_dilate); 
    insideLab = xor(mask, con2r);
    full = uint8(insideLab);
    full(insideLab)=2;
    full(con2r==1) = 1;
%     full(con2r_dilate==1) = 1;
    inside_maskLab_semi{ii} = uint8(insideLab);
    full_Lab_semi{ii} = full;
end
%% for thick edge
se = strel('disk',1,4);% this will add 1 pixel on both sides of the original con2r, and 1 pixel at the boundary of mask
semi_data = InputImgData;
semi_con2rs = [semi_data.contourLab];
semi_masks = [semi_data.maskLab];
semi_inputFigs = [semi_data.inputFig];

inside_maskLab_semi = cell(1, length(semi_con2rs));
full_Lab_semi = cell(1, length(semi_con2rs));
for ii = 1:length(semi_con2rs)
    
    con2r = semi_con2rs{ii};
    con2r_dilate = imdilate(con2r, se);
    whole_mask = imfill(con2r_dilate, 'holes');
    insideLab_dilate = xor(whole_mask, con2r_dilate); 
    
    mask = semi_masks{ii};
%     mask_dilate=imdilate(mask,se);
    full = uint8(mask);
%     full = uint8(mask_dilate);
    full(insideLab_dilate)=2;% eventually generate 2 pixel con2rs (erase 
    
    inside_maskLab_semi{ii} = uint8(insideLab_dilate);
    full_Lab_semi{ii} = full;
end


% Check the full_Lab_man
figure; colormap(gray); n = 100;
for ii = 1:9
    subplot(3,3,ii); imagesc(full_Lab_semi{ii+9*n})
end
%%
% Split the dataset into train and test
trainSize_semi = round(length(semi_inputFigs)*0.9);
trainIDs_semi = randperm(length(semi_inputFigs), trainSize_semi);
testIDs_semi = setdiff(1:length(semi_inputFigs), trainIDs_semi);

trainImgsPh_semi = semi_inputFigs(:,trainIDs_semi,1);
trainImgsdT_semi = semi_inputFigs(:,trainIDs_semi,2);
train_contours_semi = semi_con2rs(trainIDs_semi);
train_insideLabs_semi = inside_maskLab_semi(trainIDs_semi);
train_fullMasks_semi = full_Lab_semi(trainIDs_semi);

testImgsPh_semi = semi_inputFigs(:,testIDs_semi,1);
testImgsdT_semi = semi_inputFigs(:,testIDs_semi,2);
test_contours_semi = semi_con2rs(testIDs_semi);
test_insideLabs_semi = inside_maskLab_semi(testIDs_semi);
test_fullMasks_semi = full_Lab_semi(testIDs_semi);

dir_save = 'Y:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\OtherCode\CNNcode\New Flow\CombinedTrainingData\SemiAutoLabeledData\';

% save([dir_save, 'TrainingDataFromSemiAutoLabels.mat'],'trainImgsPh_semi', 'trainImgsdT_semi', 'train_contours_semi',...
%     'train_insideLabs_semi', 'train_fullMasks_semi','testImgsPh_semi','testImgsdT_semi','test_contours_semi',...
%     'test_insideLabs_semi','test_fullMasks_semi', '-v7.3')

save([dir_save, 'TrainingDataFromSemiAutoLabelsThickEdge.mat'],'trainImgsPh_semi', 'trainImgsdT_semi', 'train_contours_semi',...
    'train_insideLabs_semi', 'train_fullMasks_semi','testImgsPh_semi','testImgsdT_semi','test_contours_semi',...
    'test_insideLabs_semi','test_fullMasks_semi', '-v7.3')
%% Combine the two labeled resources


comb_con2rLabs = [semi_con2rs, man_con2rs];
comb_insideLabs = [inside_maskLab_semi, inside_maskLab_man];
comb_fullLabs = [full_Lab_semi, full_Lab_man];
comb_inputFigs = [semi_data.inputFig, man_data.inputFig];


% Alternatively, take 95% of each dataset of species
trainSize = round(length(comb_inputFigs)*0.95);

trainIDs = randperm(length(comb_inputFigs), trainSize);
valIDs = setdiff(1:length(comb_inputFigs), trainIDs);


trainImgsPh = comb_inputFigs(:,trainIDs,1);
trainImgsdT = comb_inputFigs(:,trainIDs,2);
train_contours = comb_con2rLabs(trainIDs);
train_insideLabs = comb_insideLabs(trainIDs);
train_fullMasks = comb_fullLabs(trainIDs);


valImgsPh = comb_inputFigs(:,valIDs,1);
valImgsdT = comb_inputFigs(:,valIDs,2);
val_contours = comb_con2rLabs(valIDs);
val_insideLabs = comb_insideLabs(valIDs);
val_fullMasks = comb_fullLabs(valIDs);


save_dir = 'Y:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\OtherCode\CNNcode\New Flow\CombinedTrainingData\';
% save([save_dir, 'combinedTrainingData_thickEdge'],'comb_con2rLabs','comb_insideLabs','comb_fullLabs','comb_inputFigs',...
%     'trainImgsPh', 'trainImgsdT', 'train_contours', 'train_insideLabs','train_fullMasks',...
%     'valImgsPh', 'valImgsdT', 'val_contours', 'val_insideLabs', 'val_fullMasks','-v7.3')
% save([save_dir, 'combinedTrainingData'],'comb_con2rLabs','comb_insideLabs','comb_fullLabs','comb_inputFigs',...
%     'trainImgsPh', 'trainImgsdT', 'train_contours', 'train_insideLabs','train_fullMasks',...
%     'valImgsPh', 'valImgsdT', 'val_contours', 'val_insideLabs', 'val_fullMasks','-v7.3')
save([save_dir, 'combinedTrainingDataThickEdge'],'comb_con2rLabs','comb_insideLabs','comb_fullLabs','comb_inputFigs',...
    'trainImgsPh', 'trainImgsdT', 'train_contours', 'train_insideLabs','train_fullMasks',...
    'valImgsPh', 'valImgsdT', 'val_contours', 'val_insideLabs', 'val_fullMasks','-v7.3')