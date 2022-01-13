function [adjMat] = ConstructAffinity(X, k)
    % Construct affinity graph by Gaussian kernel function.
    % AdjList = knnsearch(X, X, 'k', k + 1);
    % adjMat = zeros(size(AdjList, 1));

    % for ind = 1:size(AdjList, 1)
    %     adjMat(ind, nonzeros(AdjList(ind, 2:end))) = 1;
    %     adjMat(nonzeros(AdjList(ind, 2:end)), ind) = 1;
    % end
    % disp(optSigma(X));
    % adjMat = constructW(X, struct('NeighborMode', 'KNN', 'k', k, 'WeightMode', 'HeatKernel', 't', optSigma(X)));
    adjMat = constructW(X, struct('NeighborMode', 'KNN', 'k', k, 'WeightMode', 'HeatKernel', 't', 1));
end
