% d = 'Z:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\OtherCode\CNN_Scripts_4_Kalin\4 StartTraining\';
% load([d, 'CNN_spores_by3Lables_comb_bigger_s3_long_convImpOfFilter'])
d = 'Z:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\OtherCode\CNNcode\New Flow\CombinedTrainingData\';
load([d 'CNN_spores_by3Lables_thickEdge_comb1'])
% load([d 'CNN_spores_by3Lables_thickEdge_comb2'])

d = 'Z:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\interaction_HP\SefInhibition\B1511\20150120\40xD\';
fname = '20150120b1511dtday1_xy1 - Aligned.nd2';
% fname = '20150120b1511dtday1_xy6 - Aligned.nd2';

% d = 'Z:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\interaction_HP\SefInhibition\M145\20141110\D3_5xD\';
% fname = '20141110m145dtd3_5xd_xy8 - Aligned.nd2';

% d = 'Z:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\interaction_HP\SefInhibition\ISP5230\20150304_sup_on0.02NBAgar\0.02NB\rep1\';
% fname = '20150304_nb_xy5 - Aligned.nd2';

% d = 'Z:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\interaction_HP\SefInhibition\M145\20141110\D3_5xD\';
% fname = '20141110m145dtd3_5xd_xy6 - Aligned.nd2';

% d = 'Z:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\interaction_HP\SefInhibition\M145\20150428M145dTvsB1511eG\B1511eGcontrol\';
% fname = '20150428m145dtvsb1511eg_b1511con_xy2 - Aligned.nd2';

% d = 'Z:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\interaction_HP\SefInhibition\ISP5230\20150320_sup_on0.02NBAgar\0.02NB\rep1\';
% fname = '20150320isp5230dt_nb_xy5 - Aligned.nd2';

% Erik's dirty figure
% d = 'ws/sysbio/vetsigiangroup/shared/Erik/EX4d/multipoints/xy1/161118_strains_22001_xy1t\';
% d = 'Y:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\large_image\Erik dirty imgs\';
% fname = '161118_strains_22001_xy1_crop2.nd2';
nd2_fname = [d fname];
t = 5:7;
im_ph = imreadBF(nd2_fname, 1, t(1), 2);
im_flu = imreadBF(nd2_fname,1, t(1), 1);
figure; imagesc(im_ph); colormap(gray)
figure; imagesc(im_flu); colormap(gray)

[rowSize, colSize] = size(im_ph);
slideHeight = 21;
slideWidth = 21;
w = (slideWidth-1)/2;
h = (slideHeight-1)/2;

xx = w + 1 : rowSize-w; %drop left and right sides
yy = h + 1 : colSize-h; % drop up and down sides
slideVol =  numel(xx)*numel(yy);


crop_stack    = uint16(zeros(slideWidth,slideHeight,1,slideVol));
img_contour = nan*zeros(rowSize, colSize);

tic
for i = xx
    crop_rows = uint16(zeros(slideWidth,slideHeight,1,numel(yy)));
    for j = yy
        %// get the cropped image from the original image
        idxI = i-w:i+w;
        idxJ = j-h:j+h;
        crop_rows(:,:,1,j-h) = im_ph(idxI,idxJ);
    end
    k = i-w;    
    crop_stack(:,:,1,1+(k-1)*numel(yy):k*numel(yy)) = crop_rows;
end
toc
tic
TestResults = classify(XuNet, crop_stack);
toc
TestResults = uint8(TestResults);
contour_map = reshape(TestResults, [numel(yy), numel(xx)]);
figure;
imagesc(contour_map')

% find boundary pixels of contour_map
bb = contour_map' == 2;
crop = im_ph(xx,yy)';
[row,col] = size(bb);
figure;
imagesc(crop);colormap(gray); hold on
for ii = 1:row
    for jj = 1:col
        if bb(ii,jj) 
            plot(ii,jj,'r.', 'MarkerSize',8); hold on
        end
        
    end
end

d = 'G:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\OtherCode\CNNcode\New Flow\Full Image samples\testFullImgCodes\';
save([d 'FullImageSporeContourTest.mat'], 'Imgs','Imgs_dT','testContour', '-v7.3')