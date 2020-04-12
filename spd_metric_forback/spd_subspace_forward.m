function [Y] = spd_subspace_forward(X, W)

    Y=W'*X*W;
end
