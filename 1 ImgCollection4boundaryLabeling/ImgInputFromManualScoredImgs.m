%% Input image data
f = 'Y:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\OtherCode\CNNcode\New Flow\';
Species_folder = {'B1511spores', 'GrRd7spores', 'ISP5230spores', 'M145spores'};
InputImgData = struct();
for ii = 1:length(Species_folder)
    tempid = findstr(Species_folder{ii}, 'spores');
    SpeciesName = Species_folder{ii}(1:tempid-1);
    d = [f Species_folder{ii} '\' SpeciesName '_1st_frame_spores'];
    load(d)
    markedIDs = ~cellfun('isempty', Mask); 
    masks = Mask(markedIDs);
    con2rs = cell(1, length(masks));
    imgs = cell(1, length(masks),2);
    for jj = 1:size(masks,2)
        mm = masks{jj};
        contourmap = zeros(size(mm));
        for tt = 1:size(mm,3)
            contourmap(:,:,tt) = bwperim(mm(:,:,tt));            
        end
        c2 = sum(contourmap,3);
        c2(c2>0) = 1;
        con2rs{jj} = c2;
        
        imgs{1,jj,1} = im{jj};
        imgs{1,jj,2} = im_red{jj};
    end
    InputImgData(ii).separateMasks = masks;
    InputImgData(ii).contourLab = con2rs;
    InputImgData(ii).inputFig = imgs;
    InputImgData(ii).sporeNums = cellfun(@(x) size(x,3), masks);
end
%%
% Check the img, contour and mask
SpeciesSet = 3;
imgNum = 50;
figure;colormap(gray)
subplot(1,3,1)
imagesc(InputImgData(SpeciesSet).inputFig{1,imgNum,1})
axis square
subplot(1,3,2)
imagesc(InputImgData(SpeciesSet).inputFig{1,imgNum,2})
axis square
subplot(1,3,3)
imagesc(InputImgData(SpeciesSet).contourLab{imgNum})
axis square
%%
save_dir = 'Z:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\OtherCode\CNNcode\New Flow\AllInputImgDataFromManualMarkedFigs\';
save([save_dir 'InputImgData'], 'InputImgData')