function CropIms = multiImCrop(im, x,y,w,h)
% Given an image and multiple center coordinates x's and y's, and the half width and half height
% of the crop, output cropped image in 4D array
    CropIms = zeros(w*2+1, h*2+1,1, numel(x));
    for ii = 1:numel(x)
        CropIms(:,:,1,ii) = im(x(ii)-w:x(ii)+w, y(ii)-h:y(ii)+h);
    end
end