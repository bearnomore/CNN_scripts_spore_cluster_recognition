m = 'combinedTrainingData';
f = 'Z:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\OtherCode\CNNcode\New Flow\CombinedTrainingData\';
load([f m])
%% for valImgs
score = zeros(1, size(valImgsPh,2));
for ii =170:size(valImgsPh,2)
    figure;
    set(gcf, 'KeyPressFcn', {@ClickImg})
    imagesc(valImgsPh{ii})
    waitforbuttonpress
    score(ii) = get(gcf, 'UserData');
    close gcf
       
end

classfiedValImgs.singlesValPh = valImgsPh(score == 1);
classfiedValImgs.singlesValdT = valImgsdT(score == 1);
classfiedValImgs.singlesValFullMasks = val_fullMasks(score == 1);


classfiedValImgs.doubletValPh = valImgsPh(score == 2);
classfiedValImgs.doubletValdT = valImgsdT(score == 2);
classfiedValImgs.doubletValFullMasks = val_fullMasks(score == 2);



classfiedValImgs.multipletValPh = valImgsPh(score == 3);
classfiedValImgs.multipletValdT = valImgsdT(score == 3);
classfiedValImgs.multipletValFullMasks = val_fullMasks(score == 3);

classfiedValImgs.badValPh = valImgsPh(score == 0);
classfiedValImgs.badValdT = valImgsdT(score == 0);
classfiedValImgs.badValFullMasks = val_fullMasks(score == 0);
%% For train imgs
score = zeros(1, size(trainImgsPh,2));
%%
for ii =1:size(trainImgsPh,2)
    figure;
    set(gcf, 'KeyPressFcn', {@ClickImg})
    imagesc(trainImgsPh{ii})
    waitforbuttonpress
    score(ii) = get(gcf, 'UserData');
    close gcf
       
end
%%
classfiedTrainImgs.singlesTrainPh = trainImgsPh(score == 1);
classfiedTrainImgs.singlesTraindT = trainImgsdT(score == 1);
classfiedTrainImgs.singlesTrainFullMasks = train_fullMasks(score == 1);


classfiedTrainImgs.doubletTrainPh = trainImgsPh(score == 2);
classfiedTrainImgs.doubletTraindT = trainImgsdT(score == 2);
classfiedTrainImgs.doubletTrainFullMasks = train_fullMasks(score == 2);



classfiedTrainImgs.multipletTrainPh = trainImgsPh(score == 3);
classfiedTrainImgs.multipletTraindT = trainImgsdT(score == 3);
classfiedTrainImgs.multipletTrainFullMasks = train_fullMasks(score == 3);

classfiedTrainImgs.badTrainPh = trainImgsPh(score == 0);
classfiedTrainImgs.badTraindT = trainImgsdT(score == 0);
classfiedTrainImgs.badTrainFullMasks = train_fullMasks(score == 0);
%%
% save([f m], 'classfiedTrainImgs', '-append', '-v7.3')



