function [AN] = NormAdjac(A)
    AN = A - diag(diag(A));  % remove self-loops
    rowsum = 0.000000001 + AN * ones(size(AN, 2), 1);  % get row sum (degrees) and avoid NaN
    vec = sqrt(1./(rowsum));
    n = length(vec);
    zpfD = spdiags(vec(:),0,n,n);  
    AN = zpfD * AN * zpfD;
end