clear;
clc;

load('../data/ETH80/Gallery_ETH80_split10.mat'); 
load('../data/ETH80/Probe_ETH80_split10.mat');  

cd manopt;
addpath(genpath(pwd));
cd  ..;

addpath spd_metric_forback;


%%-------------------superparameter
global inputspd_size;
global subspace_size;
global mask_step;
global subspace_num;
global project_size;
global batch_size;
global ratio;
global beta;
global r;
global WLR;
global MLR;
global ALR;
global iteration;
global max_margin;
global min_margin;

inputspd_size=120;
project_size=120;

subspace_size=16;
mask_step=16;

subspace_num=((project_size-subspace_size)/mask_step)+1;


ratio=2; % default setting, =2 means positive : all = 1 : 2
[pairs] = GenConstraint(Gallery.y,ratio);
batch_size=size(pairs,1);

beta=1;
r=0.1; % the regularization of matrix M
WLR=0.000001;
MLR=0.000001;
ALR=0.000001;
iteration=20;

initial_alphabeta=0.5;
lr_decay=0.99;

max_margin=30;
min_margin=10;
%%-------------------




C = Compute_SPD (Gallery.X);
C_t =Compute_SPD (Probe.X);

%C = Compute_Log_Cov (Gallery.X);
%C_t =Compute_Log_Cov (Probe.X);

%% calculate similarity
sim_mat = zeros(length(Gallery.y),length(Probe.y));
for i = 1 : length(Gallery.y)
    T1=C(:,:,i);
    for j = 1 : length(Probe.y)
        T2=C_t(:,:,j);
    	i
    	j
    	%dm=norm(logm(T1^(-0.5)*T2*T1^(-0.5)),'fro');
    	%dm=norm(T1-T2, 'fro' );
    	dm=log(det((T1+T2).*0.5))-0.5*log(det(T1*T2));
        sim_mat(i,j) = dm;
    end
end

%% calculate accuracy
sampleNum = length(Probe.y);
[sim ind] = sort(sim_mat,1,'ascend');
correctNum = length(find((Probe.y-Gallery.y(ind(1,:)))==0));
fRate1 = correctNum/sampleNum;
fprintf('fRate = %f \n', fRate1);