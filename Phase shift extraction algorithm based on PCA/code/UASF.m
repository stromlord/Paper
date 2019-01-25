clear all;clc;
close all;
% 加载图片
% startnum = 141;
% for i = 1:7
%     n = i-1;
%     eval(['I' num2str(n) '=double(imread(''../experinment_data/1.22/Image' num2str(startnum+n, '%04d') '.bmp''));']);
% %     eval(['I' num2str(n) '=double(imread(''../experinment_data/r3/Image' num2str(startnum+n, '%04d') '.bmp''));']);
%     filename = num2str(startnum+n, '%04d')
% end

I0 = double(imread('Image0141.bmp'));
I1 = double(imread('Image0142.bmp'));
I2 = double(imread('Image0143.bmp'));
I3 = double(imread('Image0144.bmp'));
I4 = double(imread('Image0145.bmp'));
I5 = double(imread('Image0146.bmp'));
I6 = double(imread('Image0147.bmp'));
[M,N] = size(I1);

figure;imshow(abs(I1),[]);
figure;imshow(abs(I1(256:381,441:560)),[]);

f_I0 = fftshift(fft2(I0));
f_I1 = fftshift(fft2(I1));f_I2 = fftshift(fft2(I2));f_I3 = fftshift(fft2(I3));
f_I4 = fftshift(fft2(I4));f_I5 = fftshift(fft2(I5));f_I6 = fftshift(fft2(I6));
figure;imshow(500.*abs(f_I5)./max(max(f_I0)));title('full fft');

dy = 120;
dx = 170;
% 截取全息图一极
[first_order_Xm,first_order_Ym] = find(f_I1==max(max(f_I1(1:M/2-100,N/2+100:N-1))));   % 获取一级最大值
f_Ir0 = f_I0(first_order_Xm-dy:first_order_Xm+dy,first_order_Ym-dx:first_order_Ym+dx);
f_Ir1 = f_I1(first_order_Xm-dy:first_order_Xm+dy,first_order_Ym-dx:first_order_Ym+dx);
f_Ir2 = f_I2(first_order_Xm-dy:first_order_Xm+dy,first_order_Ym-dx:first_order_Ym+dx);
f_Ir3 = f_I3(first_order_Xm-dy:first_order_Xm+dy,first_order_Ym-dx:first_order_Ym+dx);
f_Ir4 = f_I4(first_order_Xm-dy:first_order_Xm+dy,first_order_Ym-dx:first_order_Ym+dx);
f_Ir5 = f_I5(first_order_Xm-dy:first_order_Xm+dy,first_order_Ym-dx:first_order_Ym+dx);
f_Ir6 = f_I6(first_order_Xm-dy:first_order_Xm+dy,first_order_Ym-dx:first_order_Ym+dx);

[f_Ir1, f_Ir2, f_Ir3] = Freq_Intensity(f_Ir0, f_Ir1, f_Ir2, f_Ir3);
[f_Ir4, f_Ir5, f_Ir6] = Freq_Intensity(f_Ir0, f_Ir4, f_Ir5, f_Ir6);

figure;
subplot(231);imshow(500.*abs(f_Ir1)./max(max(f_I1)));
subplot(232);imshow(500.*abs(f_Ir2)./max(max(f_I2)));
subplot(233);imshow(500.*abs(f_Ir3)./max(max(f_I3)));
subplot(234);imshow(500.*abs(f_Ir4)./max(max(f_I4)));
subplot(235);imshow(500.*abs(f_Ir5)./max(max(f_I5)));
subplot(236);imshow(500.*abs(f_Ir6)./max(max(f_I6)));

Ir0 = ifft2(ifftshift(f_Ir0));
Ir1 = ifft2(ifftshift(f_Ir1));Ir2 = ifft2(ifftshift(f_Ir2));Ir3 = ifft2(ifftshift(f_Ir3));
Ir4 = ifft2(ifftshift(f_Ir4));Ir5 = ifft2(ifftshift(f_Ir5));Ir6 = ifft2(ifftshift(f_Ir6));

% Ir0 = abs(Ir0);Ir1 = abs(Ir1);Ir2 = abs(Ir2);Ir3 = abs(Ir3);Ir4 = abs(Ir4);Ir5 = abs(Ir5);Ir6 = abs(Ir6);
% figure;imshow(abs(Ir0),[]);
% figure;imshow(angle(Ir0),[]);
%% PCA消除像差
% [Ir0_res,shift_compention0] = PCA_Aberration(Ir0);
% [Ir1_res,shift_compention1] = PCA_Aberration(Ir1);
% [Ir2_res,shift_compention2] = PCA_Aberration(Ir2);
% [Ir3_res,shift_compention3] = PCA_Aberration(Ir3);
% [Ir4_res,shift_compention4] = PCA_Aberration(Ir4);
% [Ir5_res,shift_compention5] = PCA_Aberration(Ir5);
% [Ir6_res,shift_compention6] = PCA_Aberration(Ir6);
% 
% figure;
% subplot(231);imshow(angle(shift_compention1), []);
% subplot(232);imshow(angle(shift_compention2), []);
% subplot(233);imshow(angle(shift_compention3), []);
% subplot(234);imshow(angle(shift_compention4), []);
% subplot(235);imshow(angle(shift_compention5), []);
% subplot(236);imshow(angle(shift_compention6), []);

%%  频率分解
% tradition
% theta = 2/3*pi;
% [Center_x,Left,Right,f_Center_x,f_Left,f_Right] = structure_tradition(Ir1,Ir2,Ir3, theta);
% [Center_y,Down,Up,f_Center_y,f_Down,f_Up] = structure_tradition(Ir4,Ir5,Ir6, theta);

% % % PCA
[Center_x,Left,Right,f_Center_x,f_Left,f_Right] = structure_PCA(Ir1,Ir2,Ir3, 'L');
[Center_y,Down,Up,f_Center_y,f_Down,f_Up] = structure_PCA(Ir4,Ir5,Ir6, 'V');

% % orthogonal
% [Center_x,Left,Right,f_Center_x,f_Left,f_Right] = structure_orthogonal(Ir1_res,Ir2_res);
% [Center_y,Down,Up,f_Center_y,f_Down,f_Up] = structure_orthogonal(Ir4_res,Ir5_res);

figure;
subplot(231),imshow(100*abs(f_Left)./max(max(f_I1)));title('Left order');
subplot(232),imshow(100*abs(f_Right)./max(max(f_I1)));title('Right order');
subplot(233),imshow(100*abs(f_Center_x)./max(max(f_I1)));title('Center order');
subplot(234),imshow(100*abs(f_Down)./max(max(f_I1)));title('Down order');
subplot(235),imshow(100*abs(f_Up)./max(max(f_I1)));title('Up order');
subplot(236),imshow(100*abs(f_Center_y)./max(max(f_I1)));title('Center order');

% [f_Left,f_Right,f_Down,f_Up] = Freq_Normalization(f_Center_x,f_Left,f_Right,f_Down,f_Up);

%% 频移
f_Left_pad = zeros(M,N);
f_Right_pad = zeros(M,N);
f_Up_pad = zeros(M,N);
f_Down_pad = zeros(M,N);
f_Center_pad = zeros(M,N);

f_Left_pad(M/2-dy:M/2+dy,N/2-dx:N/2+dx) = f_Left;
f_Right_pad(M/2-dy:M/2+dy,N/2-dx:N/2+dx) = f_Right;
f_Up_pad(M/2-dy:M/2+dy,N/2-dx:N/2+dx) = f_Up;
f_Down_pad(M/2-dy:M/2+dy,N/2-dx:N/2+dx) = f_Down;
f_Center_pad(M/2-dy:M/2+dy,N/2-dx:N/2+dx) = f_Ir0;

Left_pad = ifft2(ifftshift(f_Left_pad));
Right_pad = ifft2(ifftshift(f_Right_pad));
Up_pad = ifft2(ifftshift(f_Up_pad));
Down_pad = ifft2(ifftshift(f_Down_pad));
Center_pad = ifft2(ifftshift(f_Center_pad));

%% PCA移频
% [Left_res,shift_Left] = PCA_Aberration(Left_pad,1);
% [Right_res,shift_Right] = PCA_Aberration(Right_pad,1);
% [Up_res,shift_Up] = PCA_Aberration(Up_pad,1);
% [Down_res,shift_Down] = PCA_Aberration(Down_pad,1);
% [Center_res,shift_Center] = PCA_Aberration(Center_pad,1);
% 
% f_Left_res = fftshift(fft2(Left_res));
% f_Right_res = fftshift(fft2(Right_res));
% f_Up_res = fftshift(fft2(Up_res));
% f_Down_res = fftshift(fft2(Down_res));
% 
% f_Ir0_res = fftshift(fft2(Ir0));
% f_Ir0_pad = zeros(M,N);
% f_Ir0_pad(M/2-dy:M/2+dy,N/2-dx:N/2+dx) = f_Ir0_res;
% Ir0_pad = ifft2(ifftshift(f_Ir0_pad));
% 
% f_s = f_Left_res+f_Down_res+f_Right_res+f_Up_res;
% figure;imshow(255.*abs(f_s)./max(max(f_s)));title('fft_structure');
% s = ifft2(ifftshift(f_s));
% figure;imshow(abs(s),[]);
% figure;imshow(angle(s),[]);title('structure angle');

%% 手动移频
% y = linspace(-960/2,960/2,960);
% x = linspace(-1280/2,1280/2,1280);
% [x,y]=meshgrid(x,y);
% 
% [yC,xC] = find(f_Center_pad == max(max(f_Center_pad)));
% [yL,xL] = find(f_Left_pad == max(max(f_Left_pad)));
% [yR,xR] = find(f_Right_pad == max(max(f_Right_pad)));
% [yU,xU] = find(f_Up_pad == max(max(f_Up_pad)));
% [yD,xD] = find(f_Down_pad == max(max(f_Down_pad)));
% 
% dLx = xC - xL; dLy = yC - yL;
% dRx = xC - xR; dRy = yC - yR;
% dUx = xC - xU; dUy = yC - yU;
% dDx = xC - xD; dDy = yC - yD;
% 
% Left_pad = Left_pad.*exp(1i*(2*dLx*pi.*x + 2*dLy*pi.*y));
% Right_pad = Right_pad.*exp(1i*(2*dRx*pi.*x + 2*dRy*pi.*y));
% Up_pad = Up_pad.*exp(1i*(2*dUx*pi.*x + 2*dUy*pi.*y));
% Down_pad = Down_pad.*exp(1i*(2*dDx*pi.*x + 2*dDy*pi.*y));
% 
% f_Left_pad = fftshift(fft2(Left_pad));
% f_Right_pad = fftshift(fft2(Right_pad));
% f_Up_pad = fftshift(fft2(Up_pad));
% f_Down_pad = fftshift(fft2(Down_pad));
% 
% [xC,yC] = find(f_Center_pad == max(max(f_Center_pad)));
% [xL,yL] = find(f_Left_pad == max(max(f_Left_pad)));
% [xR,yR] = find(f_Right_pad == max(max(f_Right_pad)));
% [xU,yU] = find(f_Up_pad == max(max(f_Up_pad)));
% [xD,yD] = find(f_Down_pad == max(max(f_Down_pad)));
% 
% figure;
% subplot(221),imshow(500*abs(f_Left_pad)./max(max(f_I1)));title('Left order');
% subplot(222),imshow(500*abs(f_Right_pad)./max(max(f_I1)));title('Right order');
% subplot(223),imshow(500*abs(f_Down_pad)./max(max(f_I1)));title('Down order');
% subplot(224),imshow(500*abs(f_Up_pad)./max(max(f_I1)));title('Up order');
% 
% sf = f_Up_pad + f_Down_pad+f_Left_pad+f_Right_pad;
% figure;imshow(500*abs(sf)./max(max(f_I1)));
% s = (ifft2(sf));
% figure;imshow(abs(s),[]);
% figure;imshow(abs(Center_pad),[]);title('raw');

%% 少辉
d_Left = exp(1i*angle(Center_pad./Left_pad));
d_Right = exp(1i*angle(Center_pad./Right_pad));
d_Up = exp(1i*angle(Center_pad./Up_pad));
d_Down = exp(1i*angle(Center_pad./Down_pad));

Left = Left_pad.*d_Left;
Right = Right_pad.*d_Right;
Up = Up_pad.*d_Up;
Down = Down_pad.*d_Down;

f_Left = fftshift(fft2(Left));
f_Right = fftshift(fft2(Right));
f_Up = fftshift(fft2(Up));
f_Down = fftshift(fft2(Down));

sf = f_Up + f_Down+f_Left+f_Right;
figure;imshow(200*abs(sf)./max(max(f_I1)));
s = ifft2(sf);

A_s = abs(s);
A_Center = abs(Center_pad);

figure;imshow(abs(s),[40,120]);
figure;imshow(abs(Center_pad),[5,30]);

s_p = A_s(508:526,571);
c_p = A_Center(508:526,571);

figure;
plot(s_p);hold on;plot(c_p, 'r'); legend('Our method','Normal microscopy'); xlabel('Pixel');ylabel('Intensity');

figure;imshow(A_s(453:641,415:613),[40,120]);
figure;imshow(A_Center(453:641,415:613),[5,30]);
