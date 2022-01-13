clear all;
%% define model name
model = 'CPUFS_guan_penaltyD';
%% load data.
dataset = 'ORL';
fprintf('Dataset is %s.\n', dataset);
X = load(strcat(dataset, '_data.mat'));
X = tensor(double(X.X));
target = load(strcat(dataset, '_label.mat'));
target = target.T + 1;

disp(size(X));

%% specifying parameters.
c = length(unique(target)); % number of clusters
fprintf('Number of clusters is %f.\n', c);
n = size(X, 3);
kMeansTimes = 20;

if strcmp(dataset, 'USPS') || strcmp(dataset, 'BA') || strcmp(dataset, 'USPSnew')
    FeaNumCandi = 10:10:100;
else
    FeaNumCandi = 50:50:300;
end

disp(FeaNumCandi);

para1 = [1e-2 1e-1 1 1e1 1e2];
para2 = [1e-2 1e-1 1 1e1 1e2];
para3 = [1e-2 1e-1 1 1e1 1e2];
% para4 = [1e-2 1e-1 1 1e1 1e2];
% [para1, para2, para3, para4] = ndgrid(para1, para2, para3, para4);
[para1, para2, para3] = ndgrid(para1, para2, para3);
feaSubsets = cell(numel(para1), 1);
nmiCell = cell(numel(para1), numel(FeaNumCandi), kMeansTimes);
accCell = cell(numel(para1), numel(FeaNumCandi), kMeansTimes);
%% run model
parfor i1 = 1:numel(para1)
    rng(i1);
    fprintf('This is %d-th search. (total %d.)\n', i1, numel(para1));
    mu = para1(i1);
    alfa = para2(i1);
    beda = para3(i1);
    % eta = para4(i1);
    real_theda = getLrate(X, c, alfa, beda);
    [A, B, C, F, U, V] = tensorFS_guan_penaltyD(X, c, mu, alfa, beda, real_theda, false);
    %% get feature subset
    W = khatrirao(U', V');
    sqW = (W.^2);
    sumW = sum(sqW, 2);
    [~, id] = sort(sumW, 'descend');
    feaSubsets{i1, 1} = id;
end

fprintf('Begin evaluation.\n');
%% evaluation
for i2 = 1:length(FeaNumCandi)
    FeaNum = FeaNumCandi(i2);

    parfor i1 = 1:numel(para1)
        rng(i1);
        id = feaSubsets{i1, 1};
        vecX = reshape(permute(double(X), [2 1 3]), size(X, 1) * size(X, 2), size(X, 3));
        Xsub = (vecX(id(1:FeaNum), :))';

        for i3 = 1:kMeansTimes
            [label, ~, ~, ~] = litekmeans(Xsub, c, 'Replicates', 1);
            [~, nmi_sqrt] = ComputeNMI(label, target);
            acc = ComputeACC(label, target);
            nmiCell{i1, i2, i3} = nmi_sqrt;
            accCell{i1, i2, i3} = acc;
        end

    end

end

% save(strcat('./CPUFS_1220/', model, '_', dataset, '_NMI.mat'), 'nmiCell');
% save(strcat('./CPUFS_1220/', model, '_', dataset, '_ACC.mat'), 'accCell');
% save(strcat('./CPUFS_1220/', model, '_', dataset, '_feaSubsets.mat'), 'feaSubsets');
% fprintf('dataset is %s, model is %s, best result is: nmi = %f, acc = %f.\n', dataset, model, max(cell2mat(nmiCell), [], 'all'), max(cell2mat(accCell), [], 'all'));
fprintf('dataset is %s.\n', dataset);
nmiCell = mean(cell2mat(nmiCell), 3);
nmiCell = max(nmiCell, [], 1);
disp(nmiCell);