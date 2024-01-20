function [DSI, sigma, phi, tau] = DNN_DSIRotD100(input_Mw,input_Rrup,input_Vs30,input_Frv,input_Fnm)

% Created by Mao-Xin Wang (dr.maoxin.wang@gmail.com or maoxin.wang@polyu.edu.hk)
% December 2023
%
% Predict RotD100 value of DSI
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
%   DSI          = median displacement spectrum intensity
%   sigma        = logarithmic total standard deviation
%   phi          = logarithmic within-event standard deviation
%   tau          = logarithmic between-event standard deviation
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% specify model coefficients
X_min = [3.0500 	-1.5606 	4.4922 	0.0000 	0.0000];
X_max = [7.9000 	5.9910 	7.2889 	1.0000 	1.0000];

W1 = [
    -1.7054 	-2.8524 	-0.5671 	1.9115 	-2.0893 	-0.0560 	-0.0514 	1.4082 	-4.1941 	-3.0810 	2.0675 	2.4879
    1.6790 	1.0807 	-0.2434 	-2.0099 	1.4779 	-2.0608 	5.8507 	-1.4938 	-1.5346 	2.5792 	-1.6029 	-1.9817
    -0.0188 	-0.4601 	-4.5841 	1.0644 	-0.4126 	-1.7196 	-0.2457 	0.2158 	-0.6044 	1.4871 	-0.1104 	-0.0450
    0.2158 	1.0438 	0.0849 	-0.9368 	-0.1487 	-0.9852 	-0.2898 	1.2201 	-0.8023 	-1.5448 	-0.3502 	-0.8980
    0.4147 	0.0356 	0.0819 	0.0551 	0.3475 	-0.1451 	0.0163 	-0.5517 	-0.4423 	0.0878 	0.4024 	0.0833
    ];

b1 = [
    -0.9924 	-0.1413 	0.4796 	0.4813 	-1.1861 	1.0449 	-1.4158 	0.7929 	-1.0180 	-0.5753 	0.7856 	1.1586
    ];

W2 = [
    -1.5017 	0.8564 	-0.5286 	-1.3209 	-2.0554 	0.5437 	-0.7373 	0.9284 	0.2945 	0.7148 	0.3898 	-0.6898
    -2.2858 	0.3283 	-1.9674 	-1.5702 	-0.4609 	0.2975 	-0.6857 	0.3983 	0.8935 	0.2152 	1.0387 	-1.5859
    0.0572 	-0.6825 	1.6980 	1.6346 	0.1638 	-1.4655 	0.7047 	-0.4072 	-1.4374 	-1.2110 	-1.0543 	1.9180
    1.6838 	-0.7671 	-0.6903 	-0.1022 	4.5136 	-0.3349 	1.7707 	-2.0356 	-0.2917 	-2.0494 	0.0464 	1.2126
    -1.7677 	1.3980 	-0.7561 	-1.2242 	-1.8653 	0.9325 	-0.5781 	1.3634 	1.2347 	0.6482 	0.9433 	-0.7911
    0.4863 	-0.4937 	1.4432 	1.7825 	2.4879 	-1.2001 	1.4126 	-1.9221 	-1.2649 	-1.7727 	-0.8935 	1.4641
    -0.6980 	0.9136 	-1.6298 	-2.4547 	-0.6593 	0.2653 	-0.0811 	0.0509 	-0.0186 	0.5742 	1.1435 	-0.3671
    0.1297 	-1.3651 	-0.3448 	0.3943 	2.6123 	-1.9293 	2.1506 	-2.5809 	-2.0484 	-2.0287 	-0.6209 	-0.0384
    -1.5716 	0.6101 	-4.0836 	-5.3982 	-0.5457 	2.3842 	-1.3398 	0.8088 	1.8820 	0.2303 	4.7319 	-1.9665
    -0.3938 	1.6004 	-1.5659 	-2.1067 	-0.3686 	0.9157 	0.0352 	0.1631 	0.1409 	-0.0079 	1.9502 	-0.9716
    1.2329 	-0.5601 	0.4296 	0.4729 	2.7050 	-1.7095 	1.9734 	-2.3403 	-1.2502 	-2.0353 	-0.0256 	1.3391
    1.8510 	-0.7104 	0.2101 	0.0833 	4.2703 	-1.1504 	2.1061 	-2.6037 	-1.4034 	-2.5259 	-0.2846 	1.9918
    ];

b2 = [
    -0.1052 	0.1329 	-1.1316 	-0.6612 	0.6784 	-0.2240 	0.6805 	-0.8114 	-0.1276 	-0.5945 	0.1978 	-0.1053
    ];

W_out = [
    1.1819 	-1.6094 	0.8149 	1.2282 	1.0959 	-1.6428 	0.9412 	-2.0519 	-1.5769 	-1.6325 	-1.4891 	1.5194
    ]';

b_out = [-0.4389];

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
DSI = reshape(exp(Y),n_row,n_col);

%% estimate standard deviations
Mw1 = 4;  Mw2 = 6.5;
c = [6.071 	1.637 	-6.625];
phi = 0.575*ones(size(input_Mw));
tau = 1./(c(1)+c(2).*input_Mw.*log(input_Mw)+c(3).*(log(input_Mw)).^2);
tau(input_Mw<Mw1) = 1./(c(1)+c(2).*Mw1.*log(Mw1)+c(3).*(log(Mw1)).^2);
tau(input_Mw>Mw2) = 1./(c(1)+c(2).*Mw2.*log(Mw2)+c(3).*(log(Mw2)).^2);
sigma = (phi.^2+tau.^2).^0.5;


function y = sigmoid(x)
y = 1./(exp(-x)+1);
