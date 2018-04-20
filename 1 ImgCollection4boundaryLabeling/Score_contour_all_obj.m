%% 
% f = 'Z:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\interaction_HP\SefInhibition\B1511\20150120\40xD\';
% f = 'Z:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\interaction_HP\SefInhibition\M145\20141110\D3_10xD\';
% load ([f 'results_with_speed_BIG']);
% r = results;
% n = length(dir([f 'IMGs_BIG_*.mat']));
% imgs = cell(1,n);
% for p = 1:n
%     S=load([f 'IMGs_BIG_' num2str(p)]);
%     imgs{p} = S.IMG;
% end
% imgs = [imgs{:}];
% [TimeFrames, SporeIDs, Channels] = size(imgs);
% ti = 1;
% species = 'M145';
% data_dir = 'Z:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\OtherCode\CNNcode\New Flow\';
%%
f = 'Z:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\interaction_HP\SefInhibition\B1511\20150120\40xD\';
% f = 'Z:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\interaction_HP\SefInhibition\M145\20141110\D3_5xD\';
% f = 'Z:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\interaction_HP\SefInhibition\GrRd7\20141201\Day3_20xD\';
%f = 'Z:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\interaction_HP\SefInhibition\ISP5230\20150306_sup_on0.02NBAgar\0.02NB\rep1\';

load([f 'IMGs11'])
imgs = [IMG{:}];
[TimeFrames, SporeIDs, Channels] = size(imgs);
ti = 5;
species = 'B1511';
% species = 'GrRd7';
% species = 'M145';
% species = 'ISP5230';
data_dir = 'Z:\Dropbox (Vetsigian lab)\Vetsigian lab Team Folder\Ye\OtherCode\CNNcode\New Flow\';

%%
good_im_ids = sum(~cellfun('isempty', imgs(:,:,1))) == TimeFrames;
imgs1 = imgs(ti,good_im_ids,:);
sporeIDs = size(imgs1,2);
contourmap = {};
mask = {};
mapinfo = struct('NumObj',[],'TimeFrame',[],'CentralSporeId',[], 'Species',[],'RawDataPath',[],'ScoreStatus',[], 'ThresholdingFactor',[]);
mid = 41;


%%
% If pause in the middle, record the sp and continue with the next 
max_fig = get(0, 'ScreenSize');
for sp = 1:sporeIDs
    A =  double(imgs1{1,sp, 1});
    R =  double(imgs1{1,sp, 2});
    TreshFold = 0.9:0.1:1.4;
    figure('Position', max_fig); imshowpair(A,R, 'montage');
    [contourmap{sp},mask{sp},score_status, NumObj, Thresh] = DrawContour4StreptomycesAll(A,R, TreshFold);
    mapinfo(sp).NumObj = NumObj;
    mapinfo(sp).TimeFrame = ti;
    mapinfo(sp).CentralSporeId = sp;
    mapinfo(sp).Species = species;
    mapinfo(sp).RawDataPath = f;
    mapinfo(sp).ScoreStatus = score_status;
    mapinfo(sp).ThresholdingFactor = Thresh;
    close(gcf)
end
%%
ContourMap = contourmap;
Mask = mask;
MapInfo = mapinfo;
save([data_dir 'ISP5230\IdentifiedStreptomycesSpores.mat'], 'imgs1','ContourMap','Mask', 'MapInfo') 

%% 
load([data_dir 'IdentifiedStreptomycesSpores.mat'])

% Repeat block 1 and 2 and add the new marked data 
ContourMap = [Contoumap contourmap];
Mask = [Mask mask];
MapInfo = [MapInfo mapinfo];
save([data_dir 'IdentifiedStreptomycesSpores.mat'], 'ContourMap','Mask', 'MapInfo') 
