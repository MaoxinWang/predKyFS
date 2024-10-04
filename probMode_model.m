function [P_shallow,P_toe,P_deep] = probMode_model(c_gh,tanPhi,tanBeta,D,ru,kh)

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
%   kh           = seismic coefficient in horizontal direction (scalar or matrix)
%   Note: the above inputs must be in the same matrix dimension
%
% OUTPUT
%
%   P_shallow    = Probability of shallow failure (with same dimension with input)
%   P_toe        = Probability of toe failure (with same dimension with input)
%   P_deep       = Probability of deep failure (with same dimension with input)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Range values of inputted parameters
X_min = [0.0017 	0.0875 	0.1763 	1.0000 	0.0000 	0.0000];
X_max = [1.1111 	1.7321 	2.7475 	3.5000 	0.5000 	0.5000];

%% specify model coefficients
w1 = [
    -0.2127 	1.3161 	1.1420 	-6.9054 	-9.1770 	1.4814 	-2.6430 	1.5472 	0.6946 	-2.6603 	0.1116 	-1.0190 	1.3726 	3.4426 	7.2579 	-6.7728 	-0.5108 	1.5213 	2.0188 	2.0342 	-0.6586 	6.7379 	-0.5085 	0.2723 	2.6888 	-6.5065
    0.0991 	0.6772 	-2.8687 	3.5249 	0.4823 	-1.8774 	2.6742 	-2.7168 	0.3360 	0.3479 	1.7254 	1.2633 	-0.1309 	-0.3863 	-2.8165 	-0.2821 	0.9990 	0.8570 	-0.4556 	-2.3926 	0.6420 	-3.3105 	0.6392 	0.2115 	4.6777 	0.1229
    -2.1619 	-0.0779 	-0.6340 	-1.2804 	1.8485 	0.1764 	3.2589 	-0.0219 	0.5137 	6.1863 	0.2774 	0.3185 	2.1619 	-1.4790 	1.9708 	-0.2921 	0.9331 	0.6182 	0.1657 	-0.5882 	6.0880 	-0.8686 	0.4750 	-7.5764 	-0.6892 	0.1112
    2.3834 	-0.3538 	-0.0436 	0.0661 	3.9711 	0.1612 	1.5136 	0.0313 	0.4602 	0.1234 	0.2199 	3.4269 	0.0125 	-0.3198 	0.0187 	-0.1763 	3.1364 	0.3849 	2.5816 	-0.1653 	-0.0374 	-0.1526 	6.7085 	0.0056 	0.0749 	0.0105
    0.0385 	-0.2722 	0.4029 	-0.2192 	-0.0786 	-0.1918 	-0.2716 	0.2830 	-0.1335 	0.0103 	-0.2918 	-0.1326 	0.2225 	-0.0267 	0.2444 	-0.0034 	-0.1072 	-0.2867 	0.3778 	0.3100 	-0.0538 	0.1798 	-0.0395 	-0.0718 	0.0498 	-0.0330
    -0.4199 	-1.1281 	2.7597 	-0.2529 	0.0646 	0.0018 	-0.9452 	2.2813 	-0.7093 	0.3906 	-1.4976 	-0.6029 	-0.6654 	0.0210 	0.3035 	0.0743 	-0.3137 	-0.1173 	1.3036 	1.7577 	-0.7436 	0.1407 	-0.2737 	0.3098 	-0.2069 	0.0230
    ];
w2 = [
    -0.0810 	-3.0983 	0.7343 	-0.0391 	-7.6069 	-1.3631 	-3.0305 	-1.7579 	-2.3972 	-0.0052 	2.0994 	1.4867 	0.5526 	-5.1202 	-1.3261 	4.3478 	-2.9144 	0.2310 	-2.9008 	-1.8802 	-1.7808 	1.1310 	-4.4707 	1.3105 	0.8686 	1.4953
    -1.0888 	2.3563 	1.8090 	-4.9560 	2.6325 	0.5174 	0.8854 	-1.8130 	-0.7598 	0.8134 	0.3603 	-3.2834 	0.2942 	4.3450 	-0.5201 	-4.2109 	2.7695 	0.5672 	-0.2027 	-0.3609 	-1.7298 	-6.0943 	-1.6923 	-4.2518 	5.3404 	3.6008
    1.1637 	2.2963 	-2.9751 	5.5495 	2.6375 	-0.7338 	3.8521 	3.5795 	2.9330 	-1.9851 	-1.3801 	-0.0905 	-0.8177 	2.9816 	2.4905 	-0.9331 	-0.3063 	-1.1589 	2.8315 	3.4884 	2.8550 	3.1850 	4.9362 	1.9215 	-4.4133 	-4.7543
    ]';
b1 = [
    1.9015 	-3.0732 	-3.1082 	-4.2950 	-2.4332 	0.7289 	4.5397 	-2.1808 	0.7265 	1.8101 	0.0339 	1.8717 	-0.3759 	2.6616 	4.7251 	-5.8846 	2.4657 	0.7480 	-3.4496 	-0.9666 	5.9118 	2.7047 	6.7680 	-7.1852 	7.5601 	-6.2168
    ];
b2 = [3.9603 	-0.2189 	-2.0553];

%% predict factor of safety using neural network
% ensure predictors are in vector form
[n_row,n_col] = size(c_gh);
n_data = n_row*n_col;
x_c_gh = reshape(c_gh,n_data,1);
x_tanPhi = reshape(tanPhi,n_data,1);
x_tanBeta = reshape(tanBeta,n_data,1);
x_D = reshape(D,n_data,1);
x_ru = reshape(ru,n_data,1);
x_kh = reshape(kh,n_data,1);

% normalize predictors (equation 4)
X_norm = 2*([x_c_gh,x_tanPhi,x_tanBeta,x_D,x_ru,x_kh]-repmat(X_min,[n_data,1]))./(repmat(X_max-X_min,[n_data,1]))-1;

% make prediction (equation 5)
Y0 = (2./(1+exp(-2*(X_norm*w1+repmat(b1,[n_data,1]))))-1)*w2+repmat(b2,[n_data,1]);
Y1 = exp(Y0)./repmat(sum(exp(Y0),2),1,3);

%% make dimension of outputs consistent with that of inputs
P_shallow = reshape(Y1(:,1),n_row,n_col);
P_toe = reshape(Y1(:,2),n_row,n_col);
P_deep = reshape(Y1(:,3),n_row,n_col);
