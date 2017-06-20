clear;

%% 訓練データとテストデータの読み込み
load('./usps_learn.mat');  % D：次元数 × 学習データ数の行列
load('./learn_label.mat');  % learn_label :学習データのクラスラベル x 1の行列 
load('./usps_test.mat');    % E：次元数 × テストデータ数の行列
load('./test_label.mat');    % test_label：テストデータのクラスラベル x 1の行列 

%% Dim：次元数，learn_num：学習データ数, test_num：テストデータ数, 
[Dim,learn_num]=size(D);
[~,test_num]=size(E);
K = 10; % クラス数

%% 各パターンのノルムを1に正規化
for i = 1:learn_num 
    D(:,i)=D(:,i)./norm(D(:,i));   % 学習データ
end
for i = 1:test_num 
    E(:,i)=E(:,i) ./norm(E(:,i));  % テストデータ
end

%% 各クラスの平均ベクトルを求める, クラス内
for j = 0: K - 1 
    X = D(:,learn_label==j);      % ラベルがjの学習データをXに格納
    Mk(:,j+1) = mean(X,2);          % jクラスの平均ベクトルを計算し，長さを正規化 (4.42)
    Mk(:,j+1)= (1/499) .* Mk(:,j+1);  % (256, 10)
    % Mk(:,j+1)= (1/499) .* Mk(:,j+1)/norm(Mk(:,j+1));  % (256, 10)
end

Sw = zeros(Dim, Dim);

for c = 1 : K
    X = D(:,learn_label==c - 1);  % (256, 500)
    Sk = zeros(Dim, Dim);
    for i = 1 : 500
       Xi = X(:, i);
       XM = Xi - Mk(:, c);
       Sk = Sk + XM * XM';  % (256, 256)
    end
    % XM = X - repmat(Mk(:, c), 1, 500);  % (256, 500);
    Sw = Sw + Sk;
end

% クラス間
Sb = zeros(Dim, Dim);  
M = mean(Mk, 2);  % (4.44)
for c = 1:K
    Sb = Sb + 500 .* (Mk(:,c) - M) * (Mk(:,c) - M)';  % (4.46)
end

[V, DD] = eig(Sb, Sw + 0.00001 .* eye(Dim, Dim));
[p, q] = sort(diag(DD), 'descend');
V3 = V(:, q(1:200));  % (Dim, 3)


hold on;
for i = 0 : K - 1
    X = E(:,test_label== i);
    y = X' * V3;  % (500, Dim) x (Dim, 3)
    scatter3(y(:,1), y(:,2), y(:,3), 'filled');
end

%% ユークリッド距離による識別
S = zeros(10, 1);
accuracy = 0;
disp(['ユークリッド距離']);
fprintf('\t');
tic;
for i = 1:test_num
    for j = 0 : K - 1
        X = E(:, i);
        y = X' * V3;
        projected_M = Mk(:, j + 1)' * V3; % (10, Dim) * (Dim, 10)
        S(j + 1) = norm(y - projected_M);
    end
    [value, index] = min(S);
    if index - 1 == test_label(i)
       accuracy = accuracy + 1;
    end
    
end
toc;
%% 識別率の計算
accuracy=(accuracy/test_num)*100; % 全テストデータが完全に識別できた場合には，混同行列の対角成分のみが非ゼロとなる
fprintf('\taccuracy=%3.2f\n',accuracy);


%% 正規化相関による識別
S=zeros(10,1);
CONF=zeros(10,10); % 識別結果を格納する行列 (Confusion matrix (混同行列) と呼ばれる)
disp(['正規化相関']);
fprintf('\t');
tic % 処理時間を計るためにセット
for i = 1:test_num
    for j = 0:9 
        % 第i番目のテストデータ E(:,i)とクラスjの平均ベクトルとの正規化相関の自乗 Sを計算
        projected_X = E(:, i)' * V3;
        projected_M = Mk(:, j + 1)' * V3; % (10, Dim) * (Dim, 10)
        S(j + 1) = (projected_M * projected_X')^2;
        % S(j+1)=(M(:,j+1)'*E(:,i))^2;
    end
    [value, index]=max(S);
    CONF(test_label(i)+1,index)=CONF(test_label(i)+1,index)+1; % 混同行列の結果が対応する要素に積算されていく
end
toc % 掛かった処理時間を表示

%% 識別率の計算
accuracy=(sum(diag(CONF))./test_num).*100; % 全テストデータが完全に識別できた場合には，混同行列の対角成分のみが非ゼロとなる
fprintf('\taccuracy=%3.2f\n',accuracy);

%% 部分空間法

%% forming subspaces, svdだよね
n_bias = 15;
U = zeros(Dim, n_bias, K);
for c = 1 : 10
  % X=trai(:,find(==c-1));
  X = D(:,learn_label==c - 1);
  [u, S, v] = svd(X);  % 特異値分解をする
  [p, q] = sort(diag(S), 'descend');
  U(:, :, c) = u(:, q(1:n_bias));  % 主成分ベクトルを取る
end

% weighted subspace classifier
S=zeros(10, 1);
test = zeros(size(E));
CONF=zeros(10);

disp(['部分空間法']);
fprintf('\t');
tic

for ii = 1 : test_num
  for c = 1 : 10
      projected_X = V3' * E(:, ii);  % テストデータ重判別空間に射影
      projected_U = V3' * U(:, :, c);  % 主成分ベクトルを重判別空間に射影
      S(c)= norm(projected_X' * projected_U);
  end
  [value index]=max(S);
  CONF(index,test_label(ii)+1)=CONF(index,test_label(ii)+1)+1;
end
toc

%% 識別率の計算
accuracy=(sum(diag(CONF))./test_num).*100; % 全テストデータが完全に識別できた場合には，混同行列の対角成分のみが非ゼロとなる
fprintf('\taccuracy=%3.2f\n',accuracy);