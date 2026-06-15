% pihat = mnrval_octave_fallback(B, X)
%
% Octave compatibility: evaluates 2-class (binomial) logistic regression.
% Same interface as mnrval(B, X) for the case used in viterbiDecodePCG_Springer:
% B is (p+1)x1 from mnrfit (intercept + coefficients), X is n x p.
% Returns pihat n x 2: pihat(:,1) = P(class 1), pihat(:,2) = P(class 2).

function pihat = mnrval_octave_fallback(B, X)

B = B(:);
n = size(X, 1);
p = size(X, 2);
% Use only intercept + first p coefficients if B has more (e.g. model trained with 4 features, we have 3 under Octave)
n_use = min(p + 1, length(B));
B_use = B(1:n_use);
n_col = n_use - 1;  % number of feature columns to use
% Linear predictor: intercept + X * coefficients (B(1) = intercept, B(2:end) = coeffs)
z = [ones(n, 1), X(:, 1:min(p, n_col))] * B_use;
p2 = 1 ./ (1 + exp(-z));
pihat = [1 - p2, p2];

end
