function [X1subspace,X2subspace,X2subspaceinv,X1X2subspaceinv,eigenvector,eigenvalue_m,eigenvalue] = subspace_eig_forward(X1,X2)

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
	
	n=size(X1,1);
	X1subspace=zeros(subspace_size,subspace_size,subspace_num);
	X2subspace=zeros(subspace_size,subspace_size,subspace_num);
	X2subspaceinv=zeros(subspace_size,subspace_size,subspace_num);
	X1X2subspaceinv=zeros(subspace_size,subspace_size,subspace_num);	
	eigenvector=zeros(subspace_size,subspace_size,subspace_num);
	eigenvalue_m=zeros(subspace_size,subspace_size,subspace_num);
	eigenvalue=zeros(subspace_size,subspace_num);

	coor=1;
	for i=1:subspace_num
		X1subspace(:,:,i)=X1(coor:coor+subspace_size-1,coor:coor+subspace_size-1);
		X2subspace(:,:,i)=X2(coor:coor+subspace_size-1,coor:coor+subspace_size-1);
		X2subspaceinv(:,:,i)=inv(X2(coor:coor+subspace_size-1,coor:coor+subspace_size-1));
		X1X2subspaceinv(:,:,i)=X1(coor:coor+subspace_size-1,coor:coor+subspace_size-1)*X2subspaceinv(:,:,i);
		[eigenvector(:,:,i),eigenvalue_m(:,:,i)]=eig(X1X2subspaceinv(:,:,i));
		eigenvalue(:,i)=diag(eigenvalue_m(:,:,i));
		coor=coor+mask_step;
	end

end

