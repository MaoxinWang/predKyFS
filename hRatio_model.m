function [h_H,h_ht,sigma_lnh] = hRatio_model(c_gh,tanPhi,tanBeta,D,ru)

% Created by Mao-Xin Wang (dr.maoxin.wang@gmail.com or wangmx@whu.edu.cn)
% December 2023
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUT
%
%   c_gh         = c/gama/H (scalar or matrix)
%                 (where c is soil cohesion, kPa; gama is soil unit weight, kN/m3;
%                  H is slope height, m)
%   tanPhi       = tangent of soil friction angle (scalar or matrix)
%   tanBeta      = tangent of slope angle (scalar or matrix)
%   D            = slope depth ratio (scalar or matrix)
%   ru           = tangent of slope angle (scalar or matrix)
%   Note: the above inputs must be in the same matrix dimension
%
% OUTPUT
%
%   h_H          = ratio of failure mass thickness (h_mass) to slope height (H)
%   h_ht         = ratio of failure mass thickness (h_mass) to total thickness (h_tot)
%   sigma_lnh    = standard deviation of ln(h_mass/H) or ln(h_mass/h_tot)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Range values of inputted parameters
X_min = [0.0017 	0.0875 	0.1763 	1.0000 	0.0000];
X_max = [1.1111 	1.7321 	2.7475 	3.5000 	0.5000];

%% specify model coefficients
w1 = [
    11.8570 	5.9724 	-0.0062 	6.0533 	8.4701 	6.0996 	5.0413 	2.0454 	4.0911 	9.1811 	2.2294 	2.6305 	7.3479 	5.3205 	21.0785 	7.0294 	4.9286 	1.6985
    -0.1202 	-2.4829 	0.0068 	-0.0617 	-0.2323 	-0.3825 	-0.1268 	0.2859 	0.0621 	-0.9523 	-0.0642 	-0.3219 	-0.3073 	-1.7789 	-2.8052 	-2.9924 	0.1225 	1.0547
    0.2285 	-2.0912 	0.0114 	-0.2680 	-0.3334 	-0.8521 	-2.7572 	-0.0040 	-0.2204 	-4.4213 	1.9942 	-0.3753 	-0.9743 	-1.9333 	-5.1784 	-0.9546 	-0.1310 	0.7745
    -0.0463 	0.4589 	0.4011 	0.0263 	0.1073 	-0.1511 	0.0536 	-0.5320 	-0.4642 	-1.2525 	-0.1190 	-0.4244 	-0.2227 	0.4826 	-0.0415 	0.5738 	-0.0645 	-0.4735
    0.0031 	0.1411 	-0.0044 	-0.0190 	-0.0181 	-0.0130 	0.0076 	-0.0197 	-0.0170 	0.0808 	-0.0542 	0.0061 	-0.0151 	0.1150 	0.0464 	0.0002 	-0.0266 	-0.0814
    ];
w2_T = [
    8.4332 	-1.4767 	4.9902 	-12.6722 	5.0755 	-3.7614 	-0.2741 	-2.8895 	2.7156 	0.1620 	-2.5400 	1.8756 	2.3508 	1.7377 	0.2846 	0.4618 	6.2947 	0.6444
    7.9129 	-1.4462 	-0.0999 	-12.8053 	4.9540 	-3.9002 	-0.2770 	-3.0908 	3.1723 	0.1805 	-2.7243 	1.8650 	2.5025 	1.6697 	0.2743 	0.4811 	6.3410 	0.6490
    ];
b1 = [
    13.0498 	1.2239 	1.2445 	5.1111 	7.1797 	2.2909 	2.3777 	2.2185 	3.9936 	3.5911 	4.6192 	1.5529 	2.6388 	1.3474 	9.9336 	2.2743 	3.9724 	2.8342
    ];
b2 = [-9.8103 	-5.6786];

%% predict factor of safety using neural network
% ensure predictors are in vector form
[n_row,n_col] = size(c_gh);
n_data = n_row*n_col;
x_c_gh = reshape(c_gh,n_data,1);
x_tanPhi = reshape(tanPhi,n_data,1);
x_tanBeta = reshape(tanBeta,n_data,1);
x_D = reshape(D,n_data,1);
x_ru = reshape(ru,n_data,1);

% normalize predictors (equation 4)
X_norm = 2*([x_c_gh,x_tanPhi,x_tanBeta,x_D,x_ru]-repmat(X_min,[n_data,1]))./(repmat(X_max-X_min,[n_data,1]))-1;

% make prediction (equation 5)
Y1 = exp((2./(1+exp(-2*(X_norm*w1+repmat(b1,[n_data,1]))))-1)*w2_T'+repmat(b2,[n_data,1]));
Y1_min = [zeros(size(x_D)),zeros(size(x_D))];
Y1_max = [x_D,ones(size(x_D))];
Y1(Y1<Y1_min) = Y1_min(Y1<Y1_min);
Y1(Y1>Y1_max) = Y1_max(Y1>Y1_max);

%% predict standard deviation using polynomial (equation 9)
Y2 = (0.102-0.622.*x_c_gh.^0.5+1.26.*x_c_gh-0.723.*x_c_gh.^1.5)./(1-4.354.*x_c_gh.^0.5+5.111.*x_c_gh-0.26.*x_c_gh.^1.5);

%% make dimension of outputs consistent with that of inputs
h_H = reshape(Y1(:,1),n_row,n_col);
h_ht = reshape(Y1(:,2),n_row,n_col);
sigma_lnh = reshape(Y2,n_row,n_col);
