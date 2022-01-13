function lrate = getLrate(X, c, alfa, beda)
    candidates = 2.^ - (0:60);
    iter = 1;
    lrate = eps;
    while iter <= 60
        templr = candidates(iter);
        n1 = size(X, 1); % first feature dim
        n2 = size(X, 2); % second feature dim
        n = size(X, 3); % number of samples
        C = randn(n, c);
        F = max(C, 0);
        U = rand(c, n1);
        V = rand(c, n2);
        xi = 1e-16;
        H = double(ttm(X, {U, V}, [1 2]));
        H = H(bsxfun(@plus, [1:c + 1:c * c]', [0:n - 1] * c * c))';
        obj2 = alfa .* norm(H - F, 'fro')^2 + beda .* l21norm(khatrirao(U', V'));
        VF = V.^2;
        W = khatrirao(U', V');
        Q = 1 ./ (reshape(sqrt(sum(W .* W, 2)), [n2, n1]) + xi);
        H = double(ttm(X, {U, V}, [1 2]));
        E = H(bsxfun(@plus, [1:c + 1:c * c]', [0:n - 1] * c * c)) - F';
        dU1 = zeros(c, n1);

        for k = 1:n
            dU1 = dU1 + 2 .* diag(E(:, k)) * V * double(X(:, :, k))';
        end

        dU2 = (VF * Q) .* U;
        dU = alfa .* dU1 + beda .* dU2;
        U = U - templr .* dU;

        UF = U.^2;
        W = khatrirao(U', V');
        Q = 1 ./ (reshape(sqrt(sum(W .* W, 2)), [n2, n1]) + xi);
        H = double(ttm(X, {U, V}, [1 2]));
        E = H(bsxfun(@plus, [1:c + 1:c * c]', [0:n - 1] * c * c)) - F';
        dV1 = zeros(c, n2);

        for k = 1:n
            dV1 = dV1 + 2 .* diag(E(:, k)) * U * double(X(:, :, k));
        end

        dV2 = (UF * Q') .* V;
        dV = alfa .* dV1 + beda .* dV2;
        V = V - templr .* dV;

        H = double(ttm(X, {U, V}, [1 2]));
        H = H(bsxfun(@plus, [1:c + 1:c * c]', [0:n - 1] * c * c))';
        newobj2 = alfa .* norm(H - F, 'fro')^2 + beda .* l21norm(khatrirao(U', V'));

        if newobj2 >= obj2 || isnan(newobj2)
            iter = iter + 1;
            continue;
        else
            lrate = templr;
            break;
        end
    end

end
