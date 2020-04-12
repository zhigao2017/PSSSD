clear;
clc;

cd manopt;
addpath(genpath(pwd));
cd  ..;

addpath spd_metric_forback;

load('split1_data.mat')
Gallery=struct();
Gallery.X=tr_covariance_features';
Gallery.y=tr_labels;


Probe=struct();
Probe.X=te_covariance_features';
Probe.y=te_labels;


clear tr_covariance_features;
clear tr_labels;
clear te_covariance_features;
clear te_labels;
clear te_subspace_features;
clear tr_subspace_features;


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
global momentum;


inputspd_size=400;
project_size=300;
subspace_size=10;
mask_step=10;
subspace_num=((project_size-subspace_size)/mask_step)+1;
ratio=2; % default setting, =2 means positive : negative = 1 : 2
batch_size=30;
beta=1;
r=0.01; % the regularization of matrix M
WLR=0.00001;
MLR=0.00001;
ALR=0.00001;
iteration=4000;
initial_alphabeta=1;
lr_decay=0.9995;

max_margin=100;
min_margin=5;

momentum=0.9;

%%-------------------



global W;
global M;
global M0;
global A;

W=orth(rand(inputspd_size,project_size));
M0=eye(subspace_num);
A=ones(2,subspace_num);
A=initial_alphabeta*A;
M=M0;

C = Compute_SPD (Gallery.X);
C_t =Compute_SPD (Probe.X);


for i = 1: iteration
	[pairs] = generatepairs(Gallery.y,batch_size,ratio);
	batch_x1=zeros(size(C,1),size(C,2),size(pairs,1));
	batch_x2=zeros(size(C,1),size(C,2),size(pairs,1));
	for j=1:size(pairs,1)
		batch_x1(:,:,j)=C(:,:,pairs(j,1));
		batch_x2(:,:,j)=C(:,:,pairs(j,2));
	end

	[batchloss,sim_d,dis_d]=batchloss_train(batch_x1,batch_x2,pairs(:,3));
    
    i
    batchloss

    WLR=WLR*lr_decay;
    MLR=MLR*lr_decay;
    ALR=ALR*lr_decay;

    if mod(i,200) == 0
    	i
    	batchloss
    	%% calculate similarity
		sim_mat = zeros(length(Gallery.y),length(Probe.y));
		for k = 1 : length(Gallery.y)
		    T1=C(:,:,k);
		    for j = 1 : length(Probe.y)
		        T2=C_t(:,:,j);
		        [X1] = spd_subspace_forward(T1, W);
				[X2] = spd_subspace_forward(T2, W); 
				[X1subspace,X2subspace,X2subspaceinv,X1X2subspaceinv,eigenvector,eigenvalue_m,eigenvalue] = subspace_eig_forward(X1,X2);
				[ab] = abdivergence_forward(eigenvalue,A);
				[dm,p,loss] = divergence_loss_forward(ab,M,Probe.y);
		        sim_mat(k,j) = dm;
		    end
		end
		%% calculate accuracy
		sampleNum = length(Probe.y);
		[sim ind] = sort(sim_mat,1,'ascend');
		correctNum = length(find((Probe.y-Gallery.y(ind(1,:)))==0));
		fRate1 = correctNum/sampleNum;
		fprintf('fRate = %f \n', fRate1);

		fid=fopen('results.txt','a+');
		fprintf(fid,'i = %f \n',i);
		fprintf(fid,'fRate = %f \n', fRate1);
		fclose(fid);

    end
end


