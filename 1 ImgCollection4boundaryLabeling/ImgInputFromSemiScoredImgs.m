f = 'Z:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\OtherCode\CNNcode\New Flow\';
Species_folder = {'B1511', 'GrRd7', 'ISP5230', 'M145'};
InputImgData = struct();
for ii = 1:length(Species_folder)
    d = [f Species_folder{ii} '\IdentifiedStreptomycesspores.mat'];
    load(d)
    markedIDs = ~cellfun('isempty', Mask); 
    masks = Mask(markedIDs);
    contours = ContourMap(markedIDs);
    imgF1 = imgs1(1,markedIDs,:);
    mapInfo = MapInfo(markedIDs);
    InputImgData(ii).maskLab = masks;
    InputImgData(ii).contourLab = contours;
    InputImgData(ii).inputFig = imgF1;
    InputImgData(ii).dataInfo = mapInfo;
end

% Check the img, contour and mask
SpeciesSet = 2;
imgNum = 15;
figure;
subplot(2,2,1)
imagesc(InputImgData(SpeciesSet).inputFig{1,imgNum,1})
subplot(2,2,2)
imagesc(InputImgData(SpeciesSet).inputFig{1,imgNum,1})
subplot(2,2,3)
imagesc(InputImgData(SpeciesSet).contourLab{imgNum})
subplot(2,2,4)
imagesc(InputImgData(SpeciesSet).maskLab{imgNum})

save_dir = 'Z:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\OtherCode\CNNcode\New Flow\AllInputImgDataFromSemiAutoMarkedFigs\';
save([save_dir 'InputImgData'], 'InputImgData')