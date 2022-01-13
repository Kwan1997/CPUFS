function [v] = l21norm(X)
    v = sum(vecnorm(X, 2, 2));
end