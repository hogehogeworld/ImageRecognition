%% prob_corr 正規化相関法による手書き数字の認識


%% 訓練データとテストデータの読み込み
load('./usps_learn.mat');  % D：次元数 × 学習データ数の行列
load('./learn_label.mat');  % learn_label :学習データのクラスラベル x 1の行列 
load('./usps_test.mat');    % E：次元数 × テストデータ数の行列
load('./test_label.mat');    % test_label：テストデータのクラスラベル x 1の行列 

%% Dim：次元数，learn_num：学習データ数, test_num：テストデータ数, 
[Dim,learn_num]=size(D);
[~,test_num]=size(E);

%% 各パターンのノルムを1に正規化
for i = 1:learn_num 
    D(:,i)=D(:,i)./norm(D(:,i)); 
end
for i = 1:test_num 
    E(:,i)=E(:,i)./norm(E(:,i)); 
end

U = zeros(256, 256, 10);

%% forming subspaces, svdだよね
for c = 1 : 10
  % X=trai(:,find(==c-1));
  X = D(:,learn_label==c - 1);
  [u, v] = svd(X);
  U(:, 1:size(u, 2), c) = u;
end


%% 各クラスの平均ベクトルを求める
W=zeros(Dim,100,10);  % 各クラスの直交基底 次元数×基底数×クラス数
Cx=zeros(256, 256, 10); % 分散共分散行列
for j = 0:9
    X=D(:,learn_label==j);      % ラベルがjの学習データをXに格納
    M(:,j+1)=mean(X,2);              % jクラスの平均ベクトルを計算し，長さを正規化
    M(:,j+1)=M(:,j+1)/norm(M(:,j+1));
    Xc = X - repmat(M(:, j + 1), 1, 500);
    Cx(:, :, j + 1) = Xc * Xc' .* (1/499); % 1/(500 - 1)
    % fprintf('class %d ... OK\n',j);
end

%{
%% KNNによる識別

K = 70;
disp(['KNN']);
fprintf('\t');
S = D * D';
[bv, ev] = eigs(S); 
[p, q] = sort(diag(ev), 'descend');
bv = bv(:, q(1:3));  % (256, 3);
projected_D = bv' * D;  % 学習データを3次元に落とす
projected_E = bv' * E;  % テストデータを3次元に落とす
norm_error = [];

accuracy = 0;
tic
for i = 1 : test_num   % テストデータ
   norm_error = [];
   for j = 1 : learn_num  % 学習データ
      norm_error = [norm_error norm(projected_D(:, j) - projected_E(:, i))];
   end
   [k_error, k_index] = sort(norm_error);
   k_index = k_index(1:K);  % 学習データのインデックス 5つ.
   k_label = learn_label(k_index);
   max_index = mode(k_label);
   if max_index == test_label(i)
      accuracy = accuracy + 1;
   end
end
toc

%% 識別率の計算
accuracy=(accuracy/test_num)*100; % 全テストデータが完全に識別できた場合には，混同行列の対角成分のみが非ゼロとなる
fprintf('\taccuracy=%3.2f\n',accuracy);

%}

%% ユークリッド距離による識別
S = zeros(10, 1);
accuracy = 0;
disp(['ユークリッド距離']);
fprintf('\t');
tic;
for i = 1:test_num
    for j = 1:10
        S(j) = norm(E(:, i) - M(:, j));
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
        S(j+1)=(M(:,j+1)'*E(:,i))^2;  
    end
    [value, index]=max(S);
    CONF(test_label(i)+1,index)=CONF(test_label(i)+1,index)+1; % 混同行列の結果が対応する要素に積算されていく
end
toc % 掛かった処理時間を表示

%% 識別率の計算
accuracy=(sum(diag(CONF))./test_num).*100; % 全テストデータが完全に識別できた場合には，混同行列の対角成分のみが非ゼロとなる
fprintf('\taccuracy=%3.2f\n',accuracy);



%% 特異値分解

% D: learn, E: test

disp(['主成分分析']);
fprintf('\t');
tic;
k = 30;
accuracy = 0;
Uk = zeros(10, 256, k);
for i = 1:10
   X=D(:,learn_label==i - 1);
   [U, ~, V] = svd(X);
   Uk(i, :, :) = U(:, 1:k);
end

errors = [];
test_size = size(E, 2);
for i = 1:test_size
   error = 10000;
   for j = 1:10
       Uk1 = reshape(Uk(j, :, :), 256, k);
       norm_error = norm((eye(256) - Uk1 * Uk1') * E(:, i));
       if norm_error < error
          error = norm_error;
          image_id = j - 1;
       end
   end
   if image_id == test_label(i)
      accuracy = accuracy + 1; 
   end
end
toc;

%% 識別率の計算
accuracy=(accuracy/test_num)*100; % 全テストデータが完全に識別できた場合には，混同行列の対角成分のみが非ゼロとなる
fprintf('\taccuracy=%3.2f\n',accuracy);

%}

%% 部分空間法


n_bias = 9;
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

for ii = 1 : test_size
    
  test(:,ii)=E(:,ii)./norm(E(:,ii));
  for c = 1 : 10
      % Uk1 = reshape(Uk(c, :, :), 256, k);
      S(c)= norm(U(:, :, c)'*E(:,ii));
  end
  [value index]=max(S);
  CONF(index,test_label(ii)+1)=CONF(index,test_label(ii)+1)+1;
end
toc

%% 識別率の計算
accuracy=(sum(diag(CONF))./test_num).*100; % 全テストデータが完全に識別できた場合には，混同行列の対角成分のみが非ゼロとなる
fprintf('\taccuracy=%3.2f\n',accuracy);


    
%% レオマハラノビス距離
S = zeros(10, 1);
accuracy = 0;
disp(['レオマハラノビス距離']);
fprintf('\t');
tic;
Cu = zeros(256, 256, 10);
for i = 1:10
    Cu(:, :, i) = pinv(Cx(:, :, i));
end

for i = 1:test_num
    for j = 1:10
        XM = E(:, i) - M(:, j);
        S(j) = sqrt(XM' * Cu(:, :, j) * XM);
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
