function [dlde,dldA] = abdivergence_backward(eigenvalue,A,ab,lambda1,lambda2,dldy)

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

	alpha_m=repmat(A(1,:),subspace_size,1);
	beta_m=repmat(A(2,:),subspace_size,1);

	alphabeta=repmat(A(1,:).*A(2,:),subspace_size,1);
	dlde=repmat(dldy,subspace_size,1).*(1./alphabeta).*(1./(lambda1+lambda2)).*(alphabeta.*((eigenvalue.^(beta_m-1))-(eigenvalue.^(((-1)*alpha_m)-1))));

	%%calcaulate the gradient to alpha and beta

	dlda1=(lambda1-alpha_m.*lambda2.*(log(eigenvalue)))./(lambda1+lambda2);
	dlda2=repmat(A(1,:)./(A(1,:)+A(2,:)),subspace_size,1);
	dlda3=log((lambda1+lambda2)./(repmat((A(1,:)+A(2,:)),subspace_size,1)));
	dlda=dldy.*(1./(A(1,:).*A(1,:).*A(2,:))).*(sum(dlda1-dlda2-dlda3,1));

	dldb1=(lambda2+beta_m.*lambda1.*(log(eigenvalue)))./(lambda1+lambda2);
	dldb2=repmat(A(2,:)./(A(1,:)+A(2,:)),subspace_size,1);
	dldb3=dlda3;
	dldb=dldy.*(1./(A(1,:).*A(2,:).*A(2,:))).*(sum(dldb1-dldb2-dldb3,1));

	dldA=zeros(2,subspace_num);
	
	dldA(1,:)=dlda;
	dldA(2,:)=dldb;

end
