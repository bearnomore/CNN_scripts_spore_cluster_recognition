function [ContourMap,Mask, Score_status, numObj, thresh] = DrawContour4StreptomycesAll(A,R, ThreshFactor)

full = size(A,1);

minA = min(A(:));
maxA = max(A(:));
maxR = max(R(:));
medR = median(R(:));
stdR = std(R(:));

a = A - minA;
a = a/maxA/1.5;

r = R - medR - 0.5*stdR;
r(r<0) = 0;
r = r/(4*stdR);

a = cat(3, a,a,a);
a(:,:,1) = a(:,:,1) + 0.6*r;

Zmin_red = 3;
Area_min_red = 10;

hg = fspecial('gaussian', [3 3], 0.8);
R1 = imfilter(R, hg);

max_fig = get(0, 'ScreenSize');
figure('Position', max_fig); colormap(jet)
hl = zeros(1,length(ThreshFactor));
Lb = zeros(size(A,1), size(A,2), length(ThreshFactor));
Bw = zeros(size(A,1), size(A,2), length(ThreshFactor));
for ii = 1:length(ThreshFactor)
    I = mat2gray(R1);
%     Thresh = graythresh(I);
    Thresh = median(R1(:))+Zmin_red*std(R1(:));
    fold = ThreshFactor(ii);
    out_ = I<fold*Thresh;
    I = -I;
    I(out_) = inf;
    
    L = watershed(I);
    L(out_) = 0;
    STATS = regionprops(L, 'Centroid', 'Area',  'MajorAxisLength', 'Perimeter');
    Area = [STATS.Area];
    non_sp = find(Area < Area_min_red);
    if ~isempty(non_sp)
        STATS(non_sp) = [];
        L(non_sp) = 0;
    end
    B = L > 10000;
    for i = 1 : max(L(:))
        B = B | edge(L==i);
    end
    
    Bw(:,:,ii) = B;
    Lb(:,:,ii) = L;
    
    [x,y] = ind2sub(size(B),find(B==1));
    subplot(2,3,ii)
    
    hl(ii) = imagesc(a); hold on
    plot(y,x,'w.','LineWidth', 2)
    
    set(hl(ii), 'UserData', 0); axis image; axis off
    
    set(hl(ii),'ButtonDownFcn',{@ClickRightSegment,ii});
    
end

waitfor(gcf,'UserData')
close(gcf);



    function ClickRightSegment(src, eventdata, ii)
        clickType =  get(gcf,'SelectionType');
        %          ContourMap = 5;
%         Mask = 34; Score_status = 23; numObj = 4; thresh = 12;
        switch clickType
            case 'normal' %left click on the correct image = the correct segmented image
                rectangle('Position', [1, 1, 80, 80], 'EdgeColor', 'green', 'LineWidth',2);
                %  set(src, 'UserData', ii);
                ll = Lb(:,:,ii);
                Mask = logical(ll);
                ContourMap = Bw(:,:,ii);
                Score_status = 'Correct';
                numObj = max(ll(:));
                thresh = ThreshFactor(ii);
            case 'alt' %right click on 1st image = cannot segment even by eyes
                rectangle('Position', [1, 1, 80, 80], 'EdgeColor', 'red', 'LineWidth',2);
                Mask = [];
                ContourMap = [];
                Score_status = 'Unidentifiable';
                numObj = [];
                thresh =[];
                % set(src, 'UserData', -1);
            case 'extend'  %middle click on 1st image = incorrect segment but doable by manual segmentation
                rectangle('Position', [1, 1, 80, 80], 'EdgeColor', 'yellow', 'LineWidth',2);
                Mask = [];
                ContourMap = [];
                Score_status = 'Wait4ManualScore';
                numObj = [];
                thresh =[];
                %set(src, 'UserData', -2);
        end
        
        
        set(gcf, 'UserData',1)
    end
end