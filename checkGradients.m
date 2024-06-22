function [cost numgrad] = checkGradients(J, W)

numgrad = zeros(size(W));
perturb = zeros(size(W));
cost = J(W);

e = 1e-4;
for p = 1:length(W)
    % Set perturbation vector
    perturb(p) = e;
    loss1 = J(W - perturb);
    loss2 = J(W + perturb);
    % Compute Numerical Gradient
    numgrad(p) = (loss2 - loss1) / (2*e);
    perturb(p) = 0;
end

end