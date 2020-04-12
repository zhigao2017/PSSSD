function [dm,p,loss] = divergence_loss_forward(ab,M,y)

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

	dm=ab*M*ab';
	p=exp(beta*dm);
	if y==1
		loss=log(1+p);
	else
		loss=log(1+(1/p));
	end

end

