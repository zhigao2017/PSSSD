function [dm,loss] = divergence_contrastiveloss_forward(ab,M,y)

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


	dm=ab*M*ab';
	loss=y*((max(dm-min_margin,0))^2) + (1-y)*((max(max_margin-dm,0))^2);


end