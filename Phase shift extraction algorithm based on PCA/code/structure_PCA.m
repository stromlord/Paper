function [Center,Left,Right,f_Center,f_Left,f_Right] = structure_PCA(I1,I2,I3,mod)
[M,N] = size(I1);
F_ALL = fftshift(fft2(I1));
I1 = reshape(I1,[M*N,1]);
I2 = reshape(I2,[M*N,1]);
I3 = reshape(I3,[M*N,1]);

X = [I1,I2,I3];
Xm = repmat(((I1+I2+I3)./3),1,3);
Xf = Xm-X;

% Covmat = Xf'*Xf;
Covmat = cov(Xf);
[Q,V] = eig(Covmat);

P = Q*Xf';

P1 = P(1,:);
P2 = P(2,:);
P3 = P(3,:);

P1 = reshape(P1,[M,N]);
P2 = reshape(P2,[M,N]);
P3 = reshape(P3,[M,N]);

fft_P1 = fftshift(fft2(P1));
fft_P2 = fftshift(fft2(P2));
fft_P3 = fftshift(fft2(P3));

% figure;
% subplot(131),imshow(100*abs(fft_P1)./max(max(F_ALL)));
% subplot(132),imshow(100*abs(fft_P2)./max(max(F_ALL)));
% subplot(133),imshow(100*abs(fft_P2)./max(max(F_ALL)));

fs = fft_P2;
fc = 1i*fft_P3;

if strcmp(mod,'L')
    [xLmax,yLmax]=find(fft_P3==max(max(fft_P3(:,1:N/2))));
    [xRmax,yRmax]=find(fft_P3==max(max(fft_P3(:,N/2:N))));
elseif strcmp(mod, 'V')
    [xLmax,yLmax]=find(fft_P3==max(max(fft_P3(1:M/2,:))));
    [xRmax,yRmax]=find(fft_P3==max(max(fft_P3(M/2:M,:))));
end


scale1 = fs(xLmax,yLmax)/fc(xLmax,yLmax);
scale2 = fs(xRmax,yRmax)/fc(xRmax,yRmax);

% scale1 = abs(scale1);scale2 = abs(scale2);

f_Left = fs - fc*scale2; 
f_Right = fs - fc*scale1;

scale3 = F_ALL(xLmax,yLmax)/f_Left(xLmax,yLmax);
scale4 = F_ALL(xRmax,yRmax)/f_Right(xRmax,yRmax);
f_Center = F_ALL - f_Left*scale4 - f_Right*scale3;

Left = ifft2(ifftshift(f_Left));
Right = ifft2(ifftshift(f_Right));
Center = ifft2(ifftshift(f_Center));

