function [ab,lambda1,lambda2] = abdivergence_forward(eigenvalue,A)

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
	global iteration;
	lambda1=eigenvalue;
	lambda2=eigenvalue;
	
	alpha_m=repmat(A(1,:),subspace_size,1);
	beta_m=repmat(A(2,:),subspace_size,1);

	lambda1=(lambda1.^beta_m).*alpha_m;
	lambda2=(lambda2.^(-1*alpha_m)).*beta_m;

	lambda=sum(log(lambda1+lambda2),1);
	alphabeta=1./(A(1,:).*A(2,:));
	alphaaddbeta=subspace_size*(log(sum(A,1)));
	lambda=lambda-alphaaddbeta;
	ab=lambda.*alphabeta;
	
end
