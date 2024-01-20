function [ASI, sigma, phi, tau] = DNN_ASIRotD50(input_Mw,input_Rrup,input_Vs30,input_Frv,input_Fnm)

% Created by Mao-Xin Wang (dr.maoxin.wang@gmail.com or maoxin.wang@polyu.edu.hk)
% December 2023
%
% Predict RotD50 value of ASI
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUT
%
%   input_Mw     = scalar or matrix of inputted moment magnitude
%   input_Rrup   = scalar or matrix of inputted rupture distance
%   input_Vs30   = scalar or matrix of inputted upper 30 m shear-wave velocity
%   input_Frv    = scalar or matrix of inputted reverse-faulting indicator
%   input_Fnm    = scalar or matrix of inputted normal-faulting indicator
%                  (Note: the above inputs must be in the same dimension)
%
% OUTPUT
%
%   ASI          = median acceleration spectrum intensity
%   sigma        = logarithmic total standard deviation
%   phi          = logarithmic within-event standard deviation
%   tau          = logarithmic between-event standard deviation
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% specify model coefficients
X_min = [3.0500 	-1.5606 	4.4922 	0.0000 	0.0000];
X_max = [7.9000 	5.9904 	7.2889 	1.0000 	1.0000];

W1 = [
    0.7282 	2.7201 	-1.0177 	-0.2295 	-0.0594 	-0.1214 	0.0718 	0.6300 	-0.4850 	1.3265 	-0.1074 	-0.8452 	1.5998 	0.3595 	-0.7995 	0.7398
    0.4374 	0.8216 	0.2023 	-0.3810 	0.2712 	-1.1407 	0.6613 	-1.5204 	-0.8328 	-0.0967 	-1.1473 	1.2804 	1.1908 	-0.4458 	1.5291 	-1.3435
    0.3588 	-0.1395 	-0.1683 	-1.4832 	0.9413 	-0.1387 	0.2785 	0.4056 	0.2108 	-0.2011 	-0.3115 	0.4911 	0.0207 	-0.8691 	-0.2438 	0.2378
    0.4210 	-0.3544 	-0.7035 	-0.0093 	-0.4425 	-0.5552 	-0.5112 	-0.9960 	0.1048 	-0.6671 	-0.7239 	0.0138 	-0.0627 	-0.1336 	-0.1139 	0.5720
    0.4270 	-0.7704 	-0.8573 	-0.3740 	-0.3376 	0.3280 	-0.5244 	0.3359 	0.0801 	0.3461 	0.1706 	1.0015 	0.7355 	0.9796 	1.1034 	0.7562
    ];

b1 = [
    -0.2045 	-0.2695 	0.5343 	0.3119 	-0.1288 	-0.0345 	0.3956 	-0.0208 	0.1334 	-0.0302 	-0.0026 	-0.1439 	-0.9245 	-0.4338 	-0.2680 	0.3796
    ];

W2 = [
    -0.0827 	0.0931 	0.0298 	-0.0734 	0.2599 	0.6386 	-0.6554 	-0.1093 	-0.5708 	0.1026 	-0.7865 	-0.4032 	-0.0192 	0.4656 	-0.3552 	-0.3250
    -0.2609 	-1.3672 	0.6947 	-0.1669 	0.0542 	0.1736 	1.0543 	-0.6690 	-1.7000 	-0.2696 	1.3266 	0.4779 	1.8308 	0.3023 	1.2417 	-1.1360
    0.0212 	-0.6565 	-0.5063 	0.9922 	-0.6834 	-0.3916 	1.5811 	-0.2990 	-0.0227 	-0.4648 	1.0461 	1.4004 	0.3327 	-0.1379 	0.4959 	-0.4455
    -0.3909 	-0.0159 	0.6686 	-0.1290 	0.0575 	-0.2106 	-0.3050 	0.7842 	0.2704 	-0.4796 	0.2268 	-0.2060 	-0.4817 	0.4061 	0.6710 	0.0758
    0.4152 	-0.2413 	-0.2453 	0.4987 	-0.0126 	-0.2409 	0.7182 	-0.1506 	-0.3423 	0.3900 	-0.1150 	0.4068 	0.2948 	-0.5512 	0.4336 	-0.3393
    -0.3035 	1.2831 	0.3114 	-0.4278 	0.1146 	-0.3245 	-1.2691 	0.3098 	0.0652 	0.6994 	-0.0304 	-0.7845 	-0.3710 	0.2430 	-0.7185 	0.0539
    0.0946 	-0.7590 	-0.0203 	0.6705 	-0.6938 	0.2700 	1.0702 	0.1696 	-0.4017 	0.8245 	1.1327 	0.7660 	0.8030 	-0.2798 	0.8375 	0.0472
    -0.2141 	1.6068 	0.2664 	-0.2605 	-0.5955 	-0.7434 	-1.2118 	-0.3568 	-0.5147 	1.4151 	0.6344 	-2.2486 	-0.2761 	0.2533 	-2.2411 	-0.4759
    -0.0479 	0.1385 	0.2990 	-0.6484 	0.3832 	-1.0495 	-0.0926 	-0.4215 	0.3682 	-1.1446 	0.0234 	-0.2096 	-0.6026 	0.3623 	-0.1678 	0.3548
    -0.3371 	0.1699 	-0.3377 	0.0053 	-0.6876 	0.1645 	-0.6502 	-0.3419 	-0.2578 	0.9605 	-0.7660 	-0.7350 	-0.4811 	-0.4512 	-0.4710 	-0.3098
    -0.4163 	0.3663 	-0.2529 	-0.1449 	-0.1180 	-0.4354 	-0.6736 	0.1571 	-0.0511 	0.7995 	-0.0874 	-0.2538 	-0.6179 	-0.2294 	-0.7030 	-0.1114
    1.1162 	0.9514 	-1.1610 	1.8216 	-0.4773 	1.2518 	-0.1479 	-1.4481 	1.9499 	-1.0676 	0.2817 	-0.5247 	-0.5694 	-0.9764 	-1.2556 	1.6978
    0.7900 	0.0587 	-0.4833 	-0.0059 	-0.8837 	1.8151 	-0.4981 	-0.4639 	-0.6289 	2.4025 	-1.0897 	-1.3104 	0.0048 	-0.9078 	-0.0390 	0.6200
    -0.1383 	1.1453 	0.3212 	-0.0560 	0.9928 	0.5972 	-1.3789 	-0.6637 	0.2876 	0.2492 	-0.4901 	-0.9108 	-0.5806 	0.1655 	-1.1153 	0.5543
    1.6252 	0.8957 	-1.5328 	1.5814 	-0.4950 	2.8428 	-0.0799 	-1.5766 	1.6301 	-0.4387 	0.7441 	-0.7202 	-0.4758 	-0.7055 	-0.5744 	1.9277
    -0.5673 	0.8093 	0.8447 	-0.9301 	1.4870 	0.3470 	-1.2850 	-0.1092 	0.0599 	-0.7572 	-0.3708 	-0.7465 	-0.6305 	-0.0450 	-0.2909 	0.2893
    ];

b2 = [
    0.2448 	-0.5007 	-0.2693 	0.5109 	-1.2300 	0.3200 	1.4636 	-0.0162 	0.0389 	-0.5757 	1.7827 	0.6450 	0.8743 	-0.3774 	0.3346 	0.0381
    ];

W_out = [
    -0.3134 	0.7055 	0.4024 	-0.6142 	0.8695 	-0.9143 	-0.9680 	1.0706 	-0.6712 	0.8753 	-1.0291 	-1.0152 	-0.8094 	0.3829 	-0.8530 	-0.7900
    ]';

b_out = [-0.4396];

%% perform median prediction
[n_row,n_col] = size(input_Mw);
n_data = n_row*n_col;
X_Mw = reshape(input_Mw,n_data,1);
X_lnR = reshape(log(input_Rrup),n_data,1);
X_lnVs30 = reshape(log(input_Vs30),n_data,1);
X_Frv = reshape(input_Frv,n_data,1);
X_Fnm = reshape(input_Fnm,n_data,1);

X_norm = 2*([X_Mw,X_lnR,X_lnVs30,X_Frv,X_Fnm]-repmat(X_min,[n_data,1]))./(repmat(X_max-X_min,[n_data,1]))-1;
Y = tanh(tanh(X_norm*W1+b1)*W2+b2)*W_out+b_out;
ASI = reshape(exp(Y),n_row,n_col);

%% estimate standard deviations
Mw1 = 4;  Mw2 = 6.5;
c = [16.114 6.171 -24.575];
phi = 0.637*ones(size(input_Mw));
tau = 1./(c(1)+c(2).*input_Mw.*log(input_Mw)+c(3).*(log(input_Mw)).^2);
tau(input_Mw<Mw1) = 1./(c(1)+c(2).*Mw1.*log(Mw1)+c(3).*(log(Mw1)).^2);
tau(input_Mw>Mw2) = 1./(c(1)+c(2).*Mw2.*log(Mw2)+c(3).*(log(Mw2)).^2);
sigma = (phi.^2+tau.^2).^0.5;
