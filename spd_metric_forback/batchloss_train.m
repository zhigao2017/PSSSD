function [Batch_Loss,sim_d,dis_d] = batchloss_train(batch_x1,batch_x2,batch_y)

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
	
    global W;
    global M;
    global M0;
    global A;

	Batch_Loss=0;
	DW=zeros(size(W,1),size(W,2));
	DA=zeros(size(A,1),size(A,2));
	DM=zeros(size(M,1),size(M,2));

	sim=sum(batch_y==1);
	dis=sum(batch_y==0);

	EPSILON=1e-6;

	sim_d=0;
	dis_d=0;

	for i =1:batch_size
		y=batch_y(i);
		%%forward
		[X1] = spd_subspace_forward(batch_x1(:,:,i), W);
		[X2] = spd_subspace_forward(batch_x2(:,:,i), W);
		[X1subspace,X2subspace,X2subspaceinv,X1X2subspaceinv,eigenvector,eigenvalue_m,eigenvalue] = subspace_eig_forward(X1,X2);
		[ab,lambda1,lambda2] = abdivergence_forward(eigenvalue,A);
		[dm,loss] = divergence_contrastiveloss_forward(ab,M,y);

		%y
		%dm
		if y==1
			sim_d=sim_d+dm;
		else
			dis_d=dis_d+dm;
		end

		%%backward
		[dldm,dldab] = divergence_contrastiveloss_backward(ab,M,dm,y,loss);
		[dlde,dldA] = abdivergence_backward(eigenvalue,A,ab,lambda1,lambda2,dldab);
		[dldx1,dldx2] = subspace_eig_backward(X1,X2,X1subspace,X2subspace,X2subspaceinv,X1X2subspaceinv,eigenvector,eigenvalue_m,eigenvalue,dlde);
		[~,dldw1] = spd_subspace_backward(batch_x1(:,:,i), W, dldx1);
		[~,dldw2] = spd_subspace_backward(batch_x2(:,:,i), W, dldx2);

		if y==1
			DW=DW+(dldw1+dldw2)/sim;
			DM=DM+dldm/sim;
			DA=DA+dldA/sim;
			Batch_Loss=Batch_Loss+loss/sim;
		else
			DW=DW+(dldw1+dldw2)/dis;
			DM=DM+dldm/dis;
			DA=DA+dldA/dis;
			Batch_Loss=Batch_Loss+loss/dis;
		end

	end

	Batch_Loss=	Batch_Loss+r*(trace(M*inv(M0))-log(det(M*inv(M0)))-subspace_num);
    

    DM=DM+r*(inv(M0)-inv(M));

    
	problemW.M = stiefelfactory(size(W,1), size(W,2));
	WRgrad = (problemW.M.egrad2rgrad(W, DW));
	W=problemW.M.retr(W, -WLR*WRgrad);

	problemM.M = sympositivedefinitefactory(size(M,1));
	MRgrad = (problemM.M.egrad2rgrad(M, DM));
	M=problemM.M.retr(M, -MLR*MRgrad);
	

	A=A-ALR*DA;

	alphazero=find(A(1,:)==0);
	for j =1:size(alphazero,2)
		A(1,alphazero(j))=EPSILON;
	end

	betazero=find(A(2,:)==0);
	for j =1:size(betazero,2)
		A(2,betazero(j))=EPSILON;
	end

	addzero=find(A(1,:)+A(2,:)==0);
	for j =1:size(addzero,2)
		r=unidrnd(2);
		A(r,alphazero(j))=A(r,alphazero(j))+EPSILON;
	end

	sim_d=sim_d/sim;
	dis_d=dis_d/dis;

end

