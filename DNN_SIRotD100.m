function [SI, sigma, phi, tau] = DNN_SIRotD100(input_Mw,input_Rrup,input_Vs30,input_Frv,input_Fnm)

% Created by Mao-Xin Wang (dr.maoxin.wang@gmail.com or maoxin.wang@polyu.edu.hk)
% December 2023
%
% Predict RotD100 value of SI
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
    0.7340 	-2.2698 	0.9115 	-0.5274 	-1.6197 	-2.8664 	3.6947 	-3.5887 	1.5983 	-1.9150 	-1.0938 	-1.6591
    -1.7975 	2.2681 	1.7211 	-2.4711 	-2.4678 	2.3958 	0.3233 	-0.4054 	-1.0868 	1.8799 	0.6945 	1.9176
    -0.7421 	0.5464 	3.1111 	0.2289 	0.9762 	-0.3777 	0.4274 	0.4701 	0.4958 	1.0082 	0.8928 	0.1718
    0.2760 	-0.6933 	0.2197 	-0.7147 	0.2531 	1.1148 	0.4981 	0.2857 	0.7205 	0.7108 	-1.5920 	0.3009
    0.0894 	-0.2124 	0.3186 	0.2324 	-0.1666 	-0.2479 	0.3259 	-0.0011 	-0.2775 	0.5200 	0.2564 	-0.0475
    ];

b1 = [
    1.3951 	-0.9000 	-1.2861 	1.6258 	0.9664 	-1.6473 	1.5363 	-1.3422 	1.1293 	-1.3295 	-0.6612 	-1.3883
    ];

W2 = [
    -1.5493 	2.1660 	-1.4660 	-0.9445 	0.3043 	-0.6454 	-0.5077 	-0.2430 	0.7155 	2.9960 	4.1763 	2.3082
    -0.2441 	0.3243 	0.4279 	0.5092 	-1.7010 	1.2950 	1.0927 	1.6677 	-1.8137 	0.1749 	-1.2036 	-0.2909
    0.8942 	-1.0754 	0.6484 	1.0487 	-2.0323 	0.6156 	0.7628 	0.2613 	-0.7045 	-2.1276 	-3.3102 	-1.5742
    -0.7234 	2.1105 	-0.1355 	-0.8739 	1.7854 	-0.3751 	-0.7969 	-0.3116 	0.3749 	1.8683 	4.4560 	2.8325
    -0.5304 	1.7718 	0.4460 	0.0909 	2.5884 	0.2593 	0.3953 	0.3689 	0.4305 	2.5470 	5.3853 	2.9072
    1.4594 	-1.2623 	0.2975 	0.4102 	-1.6865 	1.5739 	0.7584 	1.8513 	-2.0998 	-1.0526 	-2.5810 	-1.4771
    -3.2087 	2.5304 	-1.9040 	-1.8052 	0.0750 	-1.2299 	-0.9875 	-1.2493 	0.0440 	1.7268 	0.1194 	1.9326
    1.7934 	-0.5343 	0.9479 	1.2512 	1.8344 	0.1369 	0.6373 	1.2378 	0.4270 	0.0447 	1.9556 	0.2028
    -2.0356 	1.5396 	-1.6863 	-0.7275 	0.1551 	-0.1675 	-0.6441 	0.0688 	0.5523 	2.1539 	1.6234 	2.1508
    0.7649 	-0.6997 	1.5201 	0.8705 	-0.7241 	0.7068 	0.8585 	0.7216 	-0.5485 	-1.1277 	-1.0980 	-1.0574
    -0.2176 	-0.1766 	0.2987 	-0.2907 	-1.7195 	0.3272 	0.0977 	0.9847 	-1.4553 	-0.3375 	-1.0547 	-0.2781
    0.9517 	-0.9421 	1.2235 	0.5218 	-1.8906 	0.9967 	0.0175 	0.8952 	-0.4838 	-0.5414 	-1.1830 	-0.9161
    ];

b2 = [
    -0.7222 	0.8155 	-0.4879 	-0.5071 	-0.5558 	-0.2540 	-0.4475 	-0.1220 	-0.1195 	0.8233 	0.4774 	0.9052
    ];

W_out = [
    -2.4000 	1.1497 	-1.9121 	-1.3466 	1.3722 	-1.5274 	-1.3034 	-1.6032 	0.7224 	1.3311 	1.4629 	1.0500
    ]';

b_out = [-0.4442];

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
c = [9.632 	3.086 	-12.407];
phi = 0.568*ones(size(input_Mw));
tau = 1./(c(1)+c(2).*input_Mw.*log(input_Mw)+c(3).*(log(input_Mw)).^2);
tau(input_Mw<Mw1) = 1./(c(1)+c(2).*Mw1.*log(Mw1)+c(3).*(log(Mw1)).^2);
tau(input_Mw>Mw2) = 1./(c(1)+c(2).*Mw2.*log(Mw2)+c(3).*(log(Mw2)).^2);
sigma = (phi.^2+tau.^2).^0.5;


function y = sigmoid(x)
y = 1./(exp(-x)+1);