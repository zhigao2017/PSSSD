function [dldm,dldab] = divergence_contrastiveloss_backward(ab,M,dm,y,dldy)

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
	global max_margin;
	global min_margin;

	dlddm=2*y*max(dm-min_margin,0)+2*(y-1)*max(max_margin-dm,0);
	dldm=ab'*dlddm*ab;
	dldab=dlddm*ab*(M'+M);
	
end