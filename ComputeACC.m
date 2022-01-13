function [acc] = ComputeACC(label_pred, label_true)
    %% Inputting format is aligned to ComputeNMI.
    label_true = label_true(:);
    label_pred = label_pred(:);

    if size(label_true) ~= size(label_pred)
        error('size(L1) must == size(L2)');
    end

    label_pred = bestMap(label_true, label_pred);
    acc = length(find(label_true == label_pred)) / length(label_true);
end
