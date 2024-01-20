function [SI, sigma, phi, tau] = DNN_SIRotD50(input_Mw,input_Rrup,input_Vs30,input_Frv,input_Fnm)

% Created by Mao-Xin Wang (dr.maoxin.wang@gmail.com or maoxin.wang@polyu.edu.hk)
% December 2023
%
% Predict RotD50 value of SI
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
%   SI           = median spectrum intensity
%   sigma        = logarithmic total standard deviation
%   phi          = logarithmic within-event standard deviation
%   tau          = logarithmic between-event standard deviation
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% specify model coefficients
X_min = [3.0500 	-1.3093 	4.4922 	0.0000 	0.0000];
X_max = [7.9000 	5.9910 	7.2889 	1.0000 	1.0000];

W1 = [
    0.3052 	-2.0869 	1.9520 	-0.8189 	-1.1026 	0.4218 	0.6327 	-0.9708 	0.3206 	-2.9081 	4.1863 	-1.8935 	2.0161 	1.8429 	-0.2293 	0.2324
    -1.8073 	0.1737 	0.3466 	1.4316 	1.7526 	1.6698 	-1.3398 	1.3483 	-1.7239 	1.2666 	0.6961 	-1.9828 	-1.7837 	-1.4158 	0.6649 	-0.6337
    0.3511 	-0.2679 	0.9878 	0.0864 	0.1145 	2.7297 	-0.9148 	-0.0347 	-0.0036 	0.7641 	0.2376 	1.2266 	0.2337 	-0.2887 	1.9012 	1.6900
    0.0589 	-0.4555 	0.3315 	-0.1928 	0.5407 	0.3701 	0.3771 	-0.0023 	-0.1145 	-0.3999 	0.2762 	0.4454 	-0.8349 	-0.2996 	-0.6296 	-0.8210
    0.5231 	0.1577 	0.6491 	0.3933 	0.2180 	0.2409 	0.3193 	0.2530 	-0.2913 	-1.1376 	0.2382 	0.0342 	0.0251 	0.4186 	0.4818 	1.0226
    ];

b1 = [
    0.8017 	-1.3883 	-1.3690 	-0.4635 	-1.0684 	-1.2495 	0.7158 	-0.6190 	0.5580 	-0.3019 	1.4863 	0.7041 	0.6866 	0.4516 	-1.0598 	-1.4118
    ];

W2 = [
    -0.2252 	-0.0739 	-0.7251 	0.2374 	0.7933 	1.3777 	-0.3172 	0.2455 	-0.3968 	1.0800 	-3.8325 	0.7078 	-0.7106 	0.1834 	-0.5039 	1.6234
    1.6920 	0.9047 	0.6475 	-0.1492 	-0.7166 	-0.6705 	0.6679 	-0.2996 	0.8662 	-0.2494 	-0.0769 	-0.3194 	0.6697 	-0.8689 	0.6038 	0.3066
    0.3346 	-0.1460 	-0.1319 	0.2137 	-0.0840 	0.5263 	-0.1650 	-0.1533 	0.9115 	-0.8223 	0.0173 	-0.2983 	-0.2037 	-0.4519 	-0.3123 	0.4248
    1.0778 	0.3849 	0.1638 	-0.5591 	-0.2257 	-1.1891 	0.2528 	-0.1756 	1.2102 	-0.9176 	1.5659 	-1.0176 	0.4943 	-0.3358 	1.0420 	-0.6108
    1.0921 	0.6613 	0.5289 	-0.8886 	-0.5057 	-1.1790 	1.0108 	-0.3788 	1.0071 	-0.9808 	1.9902 	-0.8123 	0.3888 	-0.9385 	0.9188 	-0.5778
    1.3021 	0.1878 	0.9743 	-0.8346 	-0.8224 	-0.0019 	0.5287 	-0.9239 	0.9346 	-1.4177 	2.4520 	-0.4100 	0.9448 	-0.7824 	0.6254 	-1.4797
    -1.3098 	-0.8710 	-0.5070 	0.2845 	0.3522 	0.6092 	-1.0960 	0.6362 	-0.4780 	0.3057 	-2.9327 	0.3711 	-0.7516 	0.2940 	-0.9730 	2.2006
    1.1030 	-0.0279 	0.2621 	-0.5342 	-0.5910 	-1.1384 	0.5245 	-0.5811 	1.0432 	-0.8489 	1.0311 	-0.6502 	0.0574 	-0.5623 	0.9227 	-0.6402
    -1.3406 	-1.0168 	-0.3014 	0.3450 	0.4500 	0.7143 	-1.0747 	0.6673 	-0.3561 	0.3181 	-3.2361 	0.3172 	-0.4587 	0.4074 	-0.0517 	2.6655
    -0.0404 	0.3917 	0.4289 	-0.4541 	0.3268 	-1.8495 	0.2087 	-0.2201 	0.4074 	-0.9844 	-0.1683 	-1.2545 	-0.0483 	-0.0809 	1.3571 	-0.0452
    -3.0378 	-1.9352 	-1.2952 	1.3972 	1.8102 	0.1895 	-2.5045 	1.2157 	-0.1690 	0.4301 	0.2142 	0.7879 	-1.2854 	0.7502 	-0.5285 	1.7229
    -0.4332 	-0.3380 	0.0432 	-0.1959 	0.1148 	0.5077 	0.2618 	-0.3890 	-1.0314 	1.3642 	-4.4898 	-0.1055 	0.3673 	-0.1294 	-0.8373 	1.9523
    -1.2751 	-1.2124 	-0.6261 	0.3223 	1.2835 	1.0812 	-1.1219 	1.1951 	0.0225 	-0.0611 	-2.8451 	0.8149 	-1.2358 	0.8333 	-0.3758 	2.0762
    -1.4129 	-0.7797 	-0.2338 	0.7960 	0.6371 	1.1890 	-1.3796 	0.6642 	-0.3969 	-0.1990 	-1.8091 	0.8131 	-1.0292 	0.3066 	-0.1334 	2.1070
    0.4134 	0.0414 	0.3866 	-0.3129 	-0.2422 	-1.0658 	-0.0023 	-0.5579 	0.4226 	-0.7975 	1.1233 	-0.1347 	0.2192 	0.0874 	0.7019 	-0.6610
    0.3307 	0.0783 	0.1804 	-0.1540 	0.1228 	-0.7185 	-0.0840 	0.0013 	0.2188 	-0.2875 	-0.0361 	-0.5354 	-0.0869 	-0.1098 	0.4862 	-0.0897
    ];

b2 = [
    -0.2421 	-0.1699 	-0.0088 	0.1395 	0.3395 	-0.1228 	-0.1159 	0.1742 	0.2215 	-0.3711 	-0.8153 	-0.1449 	-0.3331 	0.0345 	0.1076 	0.7987
    ];

W_out = [
    -1.2011 	-1.0236 	-0.9144 	0.6766 	1.1267 	1.2093 	-1.3268 	0.7632 	-0.9300 	0.6278 	-3.4280 	0.8366 	-1.1505 	0.4557 	-1.0971 	0.7172
    ]';

b_out = [-0.2583];

%% perform median prediction
[n_row,n_col] = size(input_Mw);
n_data = n_row*n_col;
X_Mw = reshape(input_Mw,n_data,1);
X_lnR = reshape(log(input_Rrup),n_data,1);
X_lnVs30 = reshape(log(input_Vs30),n_data,1);
X_Frv = reshape(input_Frv,n_data,1);
X_Fnm = reshape(input_Fnm,n_data,1);

X_norm = 2*([X_Mw,X_lnR,X_lnVs30,X_Frv,X_Fnm]-repmat(X_min,[n_data,1]))./(repmat(X_max-X_min,[n_data,1]))-1;
Y = sigmoid(sigmoid(X_norm*W1+b1)*W2+b2)*W_out+b_out;
SI = reshape(exp(Y),n_row,n_col);

%% estimate standard deviations
Mw1 = 4;  Mw2 = 6.5;
c = [9.993 	3.030 	-12.383];
phi = 0.566*ones(size(input_Mw));
tau = 1./(c(1)+c(2).*input_Mw.*log(input_Mw)+c(3).*(log(input_Mw)).^2);
tau(input_Mw<Mw1) = 1./(c(1)+c(2).*Mw1.*log(Mw1)+c(3).*(log(Mw1)).^2);
tau(input_Mw>Mw2) = 1./(c(1)+c(2).*Mw2.*log(Mw2)+c(3).*(log(Mw2)).^2);
sigma = (phi.^2+tau.^2).^0.5;


function y = sigmoid(x)
y = 1./(exp(-x)+1);