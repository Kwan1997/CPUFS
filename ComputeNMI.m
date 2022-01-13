function [nmi_max, nmi_sqrt] = ComputeNMI(L1, L2)
    %   mutual information

    %===========
    L1 = L1(:);
    L2 = L2(:);

    if size(L1) ~= size(L2)
        error('size(L1) must == size(L2)');
    end

    L1 = L1 - min(L1) + 1; %   min (L1) <- 1;
    L2 = L2 - min(L2) + 1; %   min (L2) <- 1;
    %===========    make bipartition graph  ============
    nClass = max(max(L1), max(L2));
    G = zeros(nClass);

    for i = 1:nClass

        for j = 1:nClass
            G(i, j) = length(find(L1 == i & L2 == j)) + eps;
        end

    end

    sumG = sum(G(:));
    %===========    calculate MIhat
    P1 = sum(G, 2); P1 = P1 / sumG;
    P2 = sum(G, 1); P2 = P2 / sumG;
    H1 = sum(-P1 .* log2(P1));
    H2 = sum(-P2 .* log2(P2));
    P12 = G / sumG;
    PPP = P12 ./ repmat(P2, nClass, 1) ./ repmat(P1, 1, nClass);
    PPP(abs(PPP) < 1e-12) = 1;
    MI = sum(P12(:) .* log2(PPP(:)));
    nmi_max = real(MI) / max(real(H1), real(H2));

    nmi_sqrt = real(MI) / (real(sqrt(H1)) * real(sqrt(H2)) + eps);
end
