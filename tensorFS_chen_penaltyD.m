function [A, B, C, F, U, V] = tensorFS_chen_penaltyD(X, c, mu, alfa, beda, theda, verbose)
    n1 = size(X, 1); % first feature dim
    n2 = size(X, 2); % second feature dim
    n = size(X, 3); % number of samples
    A = rand(n1, c);
    B = rand(n2, c);
    C = randn(n, c);
    F = max(C, 0);
    U = rand(c, n1);
    last_U = U;
    V = rand(c, n2);
    last_V = V;
    L = speye(n) - NormAdjac(ConstructAffinity(reshape(double(X), size(X, 1) * size(X, 2), size(X, 3))', 5));
    xi = eps;
    iter = 0;
    maxIter = 500;
    eta = 1e5;
    % tol = 1e-6;

    while (iter < maxIter)
        %% Update A

        CkhaB = khatrirao(C, B);
        A = A .* ((double(tenmat(X, 1)) * CkhaB) ./ (A * (CkhaB' * CkhaB) + xi));

        %% Update B

        CkhaA = khatrirao(C, A);
        B = B .* ((double(tenmat(X, 2)) * CkhaA) ./ (B * (CkhaA' * CkhaA) + xi));

        %% Update C

        S = khatrirao(B, A)';
        R = double(tenmat(X, 3));
        Q = 2 .* S * R' - mu .* F' * L' + 2 .* eta .* F';
        [UC, ~, VC] = svd(Q);
        C = VC * eye(n, c) * UC';

        %% Update F

        H = double(ttm(X, {U, V}, [1 2]));
        H = H(bsxfun(@plus, (1:c + 1:c * c)', (0:n - 1) * c * c))';
        Y = (alfa ./ (alfa + eta)) .* H + (eta ./ (alfa + eta)) .* C - ((0.5 .* mu) ./ (alfa + eta)) .* L' * C;
        F = max(Y, xi);

        %% Update U and V via gradient descent method
        cnt = 0;

        while (cnt <= 2)
            VF = V.^2;
            W = khatrirao(U', V');
            Q = 1 ./ (reshape(sqrt(sum(W .* W, 2)), [n2, n1]) + xi);
            H = double(ttm(X, {U, V}, [1 2]));
            E = H(bsxfun(@plus, (1:c + 1:c * c)', (0:n - 1) * c * c)) - F';
            dU1 = zeros(c, n1);

            for k = 1:n
                dU1 = dU1 + 2 .* diag(E(:, k)) * V * double(X(:, :, k))';
            end

            dU2 = (VF * Q) .* U;
            dU = alfa .* dU1 + beda .* dU2;
            % theda_U = linesearch_Unn(U, V, -dU, X, F, n, c, alfa, beda, 1);
            % U = max(U - theda_U .* dU, 0);
            U = max(U - theda .* dU, 0);

            UF = U.^2;
            W = khatrirao(U', V');
            Q = 1 ./ (reshape(sqrt(sum(W .* W, 2)), [n2, n1]) + xi);
            H = double(ttm(X, {U, V}, [1 2]));
            E = H(bsxfun(@plus, (1:c + 1:c * c)', (0:n - 1) * c * c)) - F';
            dV1 = zeros(c, n2);

            for k = 1:n
                dV1 = dV1 + 2 .* diag(E(:, k)) * U * double(X(:, :, k));
            end

            dV2 = (UF * Q') .* V;
            dV = alfa .* dV1 + beda .* dV2;
            % theda_V = linesearch_Vnn(U, V, -dV, X, F, n, c, alfa, beda, 1);
            % V = max(V - theda_V .* dV, 0);
            V = max(V - theda .* dV, 0);

            cnt = cnt + 1;
        end
        
        if any(any(isnan(U))) || any(any(isinf(U))) || any(any(isnan(V))) || any(any(isinf(V)))
            fprintf('learning rate is too large...\n');
            [A, B, C, F, U, V] = tensorFS_chen_penaltyD(X, c, mu, alfa, beda, theda / 2, verbose);
            return;
        end

        if verbose
            fprintf('This is %d th iteration.\n', iter + 1);
        end

        last_U = U;
        last_V = V;

        iter = iter + 1;
    end

    H = double(ttm(X, {U, V}, [1 2]));
    H = H(bsxfun(@plus, (1:c + 1:c * c)', (0:n - 1) * c * c))';
    obj = norm(double(tenmat(X, 1)) - A * khatrirao(C, B)', 'fro')^2 + mu .* trace(C' * L * F) + alfa .* norm(H - F, 'fro')^2 + beda .* l21norm(khatrirao(U', V')) + eta .* norm(C - F, 'fro')^2;
    fprintf('Final objective function is %f.\n', obj);
    CFnorm_perc = 2 .* norm(C - F, 'fro') ./ (norm(C, 'fro') + norm(F, 'fro'));
    fprintf('Final norm percentage is %f%%.\n', 100 .* CFnorm_perc);
    fprintf('Final number of iterations is %f.\n', iter);

end
