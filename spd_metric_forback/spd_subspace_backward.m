function [dldx,dldw] = spd_subspace_backward(X, W, dldy)
	

    dldx=W*dldy*W';
    dldw=X'*W*dldy+X*W*dldy';
end