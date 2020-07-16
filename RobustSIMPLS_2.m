% Robust SIMPLS
% Written by Maison Clouatre
% 5/21/20
clc; clear;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% User Inputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rank = 40;                  % Rank of approximation in SIMPLS (can go to 40)
lambda0 = 1;                % RPCA weighting parameter
                    
WhichToPlot = 1;            % 0 = line plot, else = bar graph
WhichDayToPlot = 0;         % If you're only predicting one day, this is just 0.
PlotName = 'Forecasted Traffic Volume';
N = 14;                     % N-1 = Number of days you want to train SIMPLS over
n = 39;                     % Location in the data set predict over. 0 = end of data set.
time=[10:0.25:23.75];       % Generate time to plot against. 
                            % 10am-12:45pm
                            
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A = xlsread('Path\to\file\SpreadsheetName.xlsx');      % Training data
NN = xlsread('Path\to\file\SpreadsheetName.xlsx')';    % NN predictions trained in Tensorflow

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Perform Robust SIMPLS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load training data for SIMPLS/
% Use RPCA to filter sparse corruptions
[L,~] = RPCA(A((end-n)-(N+1):(end-n)-1,:),lambda0);

% Take predictors/ responses out of L
Z = L(:,1:40);
Y = L(:,41:end);
[rowsZ, colsZ] = size(Z);
[rowsY, colsY] = size(Y);

% Standardize predictors and responses
Ztilde = zeros(rowsZ, colsZ);
Ytilde = zeros(rowsY, colsY);
for i=1:colsZ
   Ztilde(:,i) = (Z(:,i)-mean(Z(:,i)))/std(Z(:,i)); 
end
for i=1:colsY
   Ytilde(:,i) = (Y(:,i)-mean(Y(:,i)))/std(Y(:,i)); 
end

% Compute SVD of Data
[U0, E0, V0] = svd(Ztilde'*Ytilde,'econ');
U = U0(:, 1:rank);          % Rank r truncation
E = E0(1:rank, 1:rank);
V = V0(:, 1:rank);

T = zeros(rowsZ, rank);     % Scores matrix
P = zeros(colsZ,1);         % Weight matrices
C = zeros(colsY,1);

% Build the SIMPLS model
for i=1:rank
    r=U(:,i);
    T(:,i) = Ztilde*r/norm(Ztilde*r);
    P(:,i) = Ztilde'*T(:,i);
    C(:,i) = Ytilde'*T(:,i);
    
    Ztilde = Ztilde - T(:,i)*T(:,i)'*Ztilde;
    Ytilde = Ytilde - T(:,i)*T(:,i)'*Ytilde;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Use the trained SIMPLS model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Collect Data
Zs = [Z;A(end-n,1:40)];             % Append new predictors to old

[rowsZs, colsZs] = size(Zs);
Ztildes=zeros(rowsZs, colsZs);
YsPredicted=zeros(rowsZs,colsY);

% Standardize predictors
for i=1:colsZs
   Ztildes(:,i) = (Zs(:,i)-mean(Zs(:,i)))/std(Zs(:,i)); 
end

% Make predictions
YsTildePredicted = Ztildes*pinv(P')*C';

% Reverse the standardization process using prior knowledge of responses
for i=1:colsY
    YsPredicted(:,i) = YsTildePredicted(:,i)*std(Y(:,i))+mean(Y(:,i));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Perform SIMPLS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Take predictors/ responses out of L
Z = A((end-n)-(N+1):(end-n)-1,1:40);
Y = A((end-n)-(N+1):(end-n)-1,41:end);
[rowsZ, colsZ] = size(Z);
[rowsY, colsY] = size(Y);

% Standardize predictors and responses
Ztilde = zeros(rowsZ, colsZ);
Ytilde = zeros(rowsY, colsY);
for i=1:colsZ
   Ztilde(:,i) = (Z(:,i)-mean(Z(:,i)))/std(Z(:,i)); 
end
for i=1:colsY
   Ytilde(:,i) = (Y(:,i)-mean(Y(:,i)))/std(Y(:,i)); 
end
[U0, E0, V0] = svd(Ztilde'*Ytilde,'econ');
U = U0(:, 1:rank);          % Rank r truncation
E = E0(1:rank, 1:rank);
V = V0(:, 1:rank);
T = zeros(rowsZ, rank);     % Scores matrix
P = zeros(colsZ,1);         % Weight matrices
C = zeros(colsY,1);
for i=1:rank
    r=U(:,i);
    T(:,i) = Ztilde*r/norm(Ztilde*r);
    P(:,i) = Ztilde'*T(:,i);
    C(:,i) = Ytilde'*T(:,i);
    
    Ztilde = Ztilde - T(:,i)*T(:,i)'*Ztilde;
    Ytilde = Ytilde - T(:,i)*T(:,i)'*Ytilde;
end
% Collect Data
Zs = [Z;A(end-n,1:40)];             % Append new predictors to old
[rowsZs, colsZs] = size(Zs);
Ztildes=zeros(rowsZs, colsZs);
SIMPLSYsPredicted=zeros(rowsZs,colsY);
% Standardize predictors
for i=1:colsZs
   Ztildes(:,i) = (Zs(:,i)-mean(Zs(:,i)))/std(Zs(:,i)); 
end
% Make predictions
YsTildePredicted = Ztildes*pinv(P')*C';

% Reverse the standardization process using prior knowledge of responses
for i=1:colsY
    SIMPLSYsPredicted(:,i) = YsTildePredicted(:,i)*std(Y(:,i))+mean(Y(:,i));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot Robust SIMPLS, SIMPLS, Neural Network, and Average
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

YActual = A((end-n)-(N+1):(end-n),41:end);      % Keep this to plot later

% Calculate average day
[~,cols] = size(A);
Atilde = zeros(1,cols);
for i=1:cols
    Atilde(i) = mean(A((end-n)-(N+1):(end-n)-1,i));
end

if WhichToPlot == 0             % line plot
    hold on
    box on
    plot(time, YActual(end-WhichDayToPlot,:), 'Linewidth', 2);
    plot(time, Atilde(1,41:end), 'Linewidth', 2);
    plot(time, NN, 'Linewidth', 2);
    plot(time, YsPredicted(end-WhichDayToPlot,:), 'Linewidth', 2);
    legend('Measured', 'Average', 'Neural Network', 'Robust SIMPLS');
    xlabel('Time (hr)');
    ylabel('Cars / 15 min');
    title(PlotName);
    grid on
    hold off
else                            % bar graph
    % Calculate mean absolute error
    NN_MAE = 0;
    Avg_MAE = 0;
    RobustSIMPLS_MAE = 0;
    SIMPLS_MAE = 0;
    n = length(time);
    Atilde = Atilde(1,41:end);
    for i=1:n
       NN_MAE = NN_MAE + abs(NN(i)-YActual(end-WhichDayToPlot,i)); 
       Avg_MAE = Avg_MAE + abs(Atilde(i)-YActual(end-WhichDayToPlot,i));
       RobustSIMPLS_MAE = RobustSIMPLS_MAE + abs(YsPredicted(end-WhichDayToPlot,i)-YActual(end-WhichDayToPlot,i));
       SIMPLS_MAE = SIMPLS_MAE + abs(SIMPLSYsPredicted(end-WhichDayToPlot,i)-YActual(end-WhichDayToPlot,i));
    end
    NN_MAE = NN_MAE/n;
    Avg_MAE = Avg_MAE/n;
    RobustSIMPLS_MAE = RobustSIMPLS_MAE/n;
    SIMPLS_MAE = SIMPLS_MAE/n;

    % build the graph
    X = categorical({'Robust SIMPLS','SIMPLS','Neural Network','Average'});
    X = reordercats(X,{'Robust SIMPLS','SIMPLS','Neural Network','Average'});
    Y = [RobustSIMPLS_MAE; SIMPLS_MAE; NN_MAE; Avg_MAE];
    b = bar(X,Y);
    b.FaceColor = 'flat';
    b.CData(1,:) = [0.4940 0.1840 0.5560];
    b.CData(3,:) = [0.9290 0.6940 0.1250];
    b.CData(4,:) = [0.8500 0.3250 0.0980];
    ylabel('Cars / 15 min')
    title('Mean Absolute Error')
    for i=1:length(Y)
       text(i,Y(i),num2str(Y(i)),'vert','bottom','horiz','center'); 
    end
end





%%%%% End of Program %%%%%





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% All user-defined functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% These functions were modified from those reported in "Data-driven science
% & engineering" by Steven L. Brunton & J. Nathan Kutz
% Cambridge University Press, 2019
% doi: 10.1017/9781108380690
function out = shrink(X, tau)
    out = sign(X).*max(abs(X)-tau,0);
end

function out = SVT(X, tau)
    [U, S, V] = svd(X, 'econ');
    out = U*shrink(S,tau)*V';
end

function [L,S] = RPCA(X,lambda0)
    [n1, n2] = size(X);
    mu = n1*n2/(4*sum(abs(X(:))));
    lambda = lambda0/sqrt(max(n1,n2));
    thresh = 1e-7*norm(X,'fro');
    
    L = zeros(size(X));
    S = zeros(size(X));
    Y = zeros(size(X));
    count = 0;
    while ((norm(X-L-S,'fro')>thresh) && (count<1000))
       L = SVT(X-S + (1/mu)*Y, 1/mu);
       S = shrink(X-L + (1/mu)*Y, lambda/mu);
       Y = Y + mu*(X-L-S);
       count = count +1;
    end
end