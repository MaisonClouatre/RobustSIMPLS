% Robust SIMPLS
% Written by Maison Clouatre
% 5/21/20
clc; clear;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% User Inputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rank = 40;                  % Rank of approximation in SIMPLS (can go to 40
                            % when predictors between 12am and 10am are in 15
                            % minute intervals).
lambda0 = 1;                % RPCA weighting parameter
                    
WhichDayToPlot = 0;         % If you're only predicting one day, this is just 0.
PlotName = 'Northbound Through @ Int. 1662';
N = 8;                      % N-1 = Number of days you want to train SIMPLS over
n = 3;                      % Location in the data set predict over. 0 = end of data set.
time=[10:0.25:23.75];       % Generate time to plot against. 
                            % 10am-12:45pm

                            
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A = xlsread('Path\to\file\SpreadsheetName.xlsx');

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
% Plot RSIMPLS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Denoise data to plot for reference
[Denoised,~] = RPCA(A((end-n)-(N+1):(end-n),:),lambda0);

YActual = A((end-n)-(N+1):(end-n),41:end);      % Keep this to plot later

% Calculate average day
[~,cols] = size(A);
Atilde = zeros(1,cols);
for i=1:cols
    Atilde(i) = mean(A((end-n)-(N+1):(end-n)-1,i));
end

hold on
box on
plot(time, YActual(end-WhichDayToPlot,:), ':', 'Linewidth', 2);
plot(time, Denoised(end,41:end),'Linewidth',2);
plot(time, YsPredicted(end-WhichDayToPlot,:), 'Linewidth', 2);
legend('Measured', 'Denoised via RPCA', 'Robust SIMPLS');
xlabel('Time (hr)');
ylabel('Cars / 15 min');
title(PlotName);
grid on
hold off

% Calculate mean absolute error
Avg_MAE = 0;
SIMPLS_MAE = 0;
n = length(time);
Atilde = Atilde(1,41:end);
for i=1:n
   Avg_MAE = Avg_MAE + abs(Atilde(i)-YActual(end-WhichDayToPlot,i));
   SIMPLS_MAE = SIMPLS_MAE + abs(YsPredicted(end-WhichDayToPlot,i)-YActual(end-WhichDayToPlot,i));
end
Avg_MAE = Avg_MAE/n
SIMPLS_MAE = SIMPLS_MAE/n



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
