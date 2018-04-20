%% for small images
dd = 'Z:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\interaction_HP\SefInhibition\M145\20141110\D3_5xD\';
load([dd 'IMGs11'])
load([dd 'results_with_contour'])
%%
dd = 'Z:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\interaction_HP\SefInhibition\GrRd7\20141129\Day1_20xD\';
load([dd 'IMGs11'])
load([dd 'results_with_contour'])

%%
Img = cell(1,length(results));
Img_red = cell(1, length(results));
timepoint = 1;
for ii = 1:length(results)
    contourmap = results(ii).contourBWmask;
    im = IMG{ii};
    emptycell = cellfun('isempty',contourmap);
    emptyim = im(:,emptycell,:);
    good = sum(~cellfun('isempty', emptyim)) == size(emptyim,1);
    good = good(:,:,1);
    img = emptyim(:,good,1);
    img_red = emptyim(:,good,2);
    Img{ii} = img(timepoint,:);
    Img_red{ii} = img_red(timepoint,:);
end
Img = [Img{:}];
Img_red = [Img_red{:}];
%%
dd = 'Z:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\interaction_HP\SefInhibition\ISP5230\20141119\D2_237sp\';
load([dd 'IMGs11'])

%%
dd = 'Z:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\interaction_HP\SefInhibition\B1511\20150120\40xD\';
load([dd 'IMGs11'])
load([dd 'results_with_manual_doublets'])
%%
Img = cell(1,length(IMG));
Img_red = cell(1, length(IMG));
timepoint = 4;
for ii = 1:length(IMG)
    
    im = IMG{ii}(timepoint,:,1);
    im_red = IMG{ii}(timepoint,:,2);
  
    good = ~cellfun('isempty', im);
    
    Img{ii} = im(good);
    Img_red{ii} = im_red(good);
end

Img = [Img{:}];
Img_red = [Img_red{:}];
%% Randomely scored 300 images
randID = randperm(size(Img,2),300);
Mask = cell(1,300);
im = Img(randID);
im_red = Img_red(randID);
%%
for xx = randID(1:end)
    id = find(randID==xx);
    A = double(im{id});
    R = double(im_red{id});
    minA = min(A(:));
    maxA = max(A(:));
    maxR = max(R(:));
    medR = median(R(:));
    stdR = std(R(:));

    A = A - minA;
    A = A/maxA/1;
   
    R = R - medR - 0.5*stdR;
    R(R<0) = 0;
    R = R/(4*stdR);
   
    A = cat(3, A,A,A);
    A(:,:,1) = A(:,:,1) + 0.5*R;
    figure; imagesc(A)
    
    n = input('How many spores');
    if n ~=0
        bw = zeros(size(R,1),size(R,2),n);
        for yy =1:n
            bw(:,:,yy) = roipoly;
        end
        
        Mask{id} = bw;
        
    else
        Mask{id} = zeros(size(R,1),size(R,2),1);
    end
    close all
end


%%
% figure; 
% for i = 201:300
%     subplot(10,10,i-200)
%     imagesc(Img_red{randID(i)})
%     title(num2str(randID(i)))
% end
%%
% save_dir = 'Z:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\OtherCode\CNNcode\New Flow\M145spores\';
% save_dir = 'Z:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\OtherCode\CNNcode\New Flow\GrRd7spores\';
% save_dir = 'Z:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\OtherCode\CNNcode\New Flow\ISP5230spores\';
save_dir = 'Z:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\OtherCode\CNNcode\New Flow\B1511spores\';
save([save_dir 'B1511_1st_frame_spores.mat'], 'im', 'im_red', 'Mask', 'randID') 

%% Try split the touchings

Mask_combined = cell(1,length(Mask));
for xx = 1:length(Mask)
    bw = sum(Mask{xx},3);
    D = bwdist(~bw);
    D = -D;
    D(~bw) = Inf;
         
    L = watershed(D);
    L(~bw) = 0;
    BW = L;
    
    A = double(im{xx});
    R = double(im_red{xx});
    
    minA = min(A(:));
    maxA = max(A(:));
    maxR = max(R(:));
    medR = median(R(:));
    stdR = std(R(:));

    A = A - minA;
    A = A/maxA/1.5;
   
    R = R - medR - 0.5*stdR;
    R(R<0) = 0;
    R = R/(4*stdR);
   
    A = cat(3, A,A,A);
    A(:,:,1) = A(:,:,1) + 0.6*R;
    
    
    if max(L(:)) == size(Mask{xx},3)
        figure; colormap(gray)
        
%         subplot(2,2,1)
%         imagesc(bw); axis square
        subplot(1,2,1)
        imagesc(BW); axis square
        subplot(1,2,2)
        imagesc(A); axis square
%         subplot(2,2,4)
%         imagesc(R); axis square
        n = input('Correct?');
        if n == 1
            Mask_combined{xx} = BW;
        end
        close all
    else
        continue
    end
end
%%
good_split = ~cellfun('isempty', Mask_combined);
im_split = im(good_split);
im_red_split = im_red(good_split);
Spore_mask = Mask_combined(good_split);

save([save_dir 'ISP5230_1st_frame_spores.mat'], 'im_split', 'im_red_split', 'Spore_mask') 
