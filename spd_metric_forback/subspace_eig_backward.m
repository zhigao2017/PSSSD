function [dldx1,dldx2] = subspace_eig_backward(X1,X2,X1subspace,X2subspace,X2subspaceinv,X1X2subspaceinv,eigenvector,eigenvalue_m,eigenvalue,dldy)

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
	dldvalue_m=zeros(subspace_size,subspace_size,subspace_num);
	dldX1X2inv=zeros(subspace_size,subspace_size,subspace_num);
	dldx1=zeros(n,n);
	dldx2=zeros(n,n);

	coor=1;
	for i=1:subspace_num
		dldvalue_m(:,:,i)=diag(dldy(:,i));
		dldX1X2inv(:,:,i)=(eigenvector(:,:,i)*dldvalue_m(:,:,i)*inv(eigenvector(:,:,i)))';

		dldx1(coor:coor+subspace_size-1,coor:coor+subspace_size-1) = dldX1X2inv(:,:,i)*X2subspaceinv(:,:,i)';
		dldx2(coor:coor+subspace_size-1,coor:coor+subspace_size-1) = (X2subspaceinv(:,:,i)*dldX1X2inv(:,:,i)'*X1subspace(:,:,i)*(-1)*X2subspaceinv(:,:,i))';
		coor=coor+mask_step;
	end

end 