% Robust SIMPLS
% Written by Maison Clouatre
% 5/21/20
clc; clear;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% User Inputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rank = 40;                  % Rank of approximation in SIMPLS
lambda0 = 1;                % RPCA weighting parameter
                    
WhichToPlot = 0;            % 0 = line plot, else = bar graph
WhichDayToPlot = 0;         % (ignore this parameter). If you're only predicting
                            % one day, this is just 0.
PlotName = 'Example 2';
N = 20;                     % N-1 = Number of days you want to train SIMPLS over
n = 2;                      % Location in the data set you want to train/ predict
                            % over. 0 = end of data set.

                            
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A = xlsread('C:\Users\maiso\Desktop\HoleyData\1663_Saturdays.xlsx');    % Training data
NN = xlsread('C:\Users\maiso\Desktop\NN_Predictions_2.xlsx')';          % NN predictions trained in Tensorflow

% Un-comment the movement that you would like to study
% WARNING: These movements are for 1665. Will need to be changed for other
% intersections to accommodate different turning movements.
% If your time is in AM/PM, you will need to change these to say "3:end"!!!
% A = A(1:end,1:96); % Eastbound Left
A = A(1:end,97:192); % Eastbound through
% A = A(1:end,193:288); % Eastbound through right
% A = A(1:end,289:384); % Eastbound total
% A = A(1:end,385:480); % Westbound left
% A = A(1:end,481:576); % Westbound through
% A = A(1:end,577:672); % Westbound total
% A = A(1:end,673:768); % Northbound left
% A = A(1:end,769:864); % Northbound through
% A = A(1:end,865:960); % Northbound total
% A = A(1:end,961:1056); % Southbound left
% A = A(1:end,1057:1152); % Southbound through
% A = A(1:end,1153:1248); % Southbound total
% A = A(1:end,1249:1344);% Total


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Perform Robust SIMPLS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Use RPCA to filter sparse corruptions
[L,~] = RPCA(A,lambda0);

% Load predictors and responses
Z = L((end-n)-(N+1):(end-n)-1,1:40);
Y = L((end-n)-(N+1):(end-n)-1,41:end);
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
Zs = L((end-n)-(N+1):(end-n),1:40);
Ys = L((end-n)-(N+1):(end-n),41:end);
YActual = A((end-n)-(N+1):(end-n),41:end);

[rowsZs, colsZs] = size(Zs);
[rowsYs, colsYs] = size(Ys);
Ztildes=zeros(rowsZs, colsZs);
YsPredicted=zeros(rowsYs,colsYs);

% Standardize predictors
for i=1:colsZs
   Ztildes(:,i) = (Zs(:,i)-mean(Zs(:,i)))/std(Zs(:,i)); 
end

% Make predictions
YsTildePredicted = Ztildes*pinv(P')*C';

% Reverse the standardization process using prior knowledge of responses
for i=1:colsYs
    YsPredicted(:,i) = YsTildePredicted(:,i)*std(Y(:,i))+mean(Y(:,i));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot SIMPLS, Neural Network, and Average
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate time to plot against
time=[10:0.25:23.75];           % 10am-12:45pm

% Calculate average day
[~,cols] = size(A);
Atilde = zeros(1,cols);
for i=1:cols
    Atilde(i) = mean(A(:,i));
end

if WhichToPlot == 0             % line plot
    hold on
    box on
    plot(time, YActual(end-WhichDayToPlot,:), 'Linewidth', 2);
    plot(time, Atilde(1,41:end), 'Linewidth', 2);
    plot(time, NN, 'Linewidth', 2);
    plot(time, YsPredicted(end-WhichDayToPlot,:), 'Linewidth', 2);
    legend('Actual', 'Average', 'Neural Network', 'Robust SIMPLS');
    xlabel('Time (hr)');
    ylabel('Cars / 15 min');
    title(PlotName);
    grid on
    hold off
else                            % bar graph
    % Calculate mean absolute error
    NN_MAE = 0;
    Avg_MAE = 0;
    SIMPLS_MAE = 0;
    n = length(time);
    Atilde = Atilde(1,41:end);
    for i=1:n
       NN_MAE = NN_MAE + abs(NN(i)-YActual(end-WhichDayToPlot,i)); 
       Avg_MAE = Avg_MAE + abs(Atilde(i)-YActual(end-WhichDayToPlot,i));
       SIMPLS_MAE = SIMPLS_MAE + abs(YsPredicted(end-WhichDayToPlot,i)-YActual(end-WhichDayToPlot,i));
    end
    NN_MAE = NN_MAE/n;
    Avg_MAE = Avg_MAE/n;
    SIMPLS_MAE = SIMPLS_MAE/n;

    % build the graph
    X = categorical({'Robust SIMPLS','Neural Network','Average'});
    X = reordercats(X,{'Robust SIMPLS','Neural Network','Average'});
    Y = [SIMPLS_MAE; NN_MAE; Avg_MAE];
    b = bar(X,Y);
    b.FaceColor = 'flat';
    b.CData(1,:) = [0.4940 0.1840 0.5560];
    b.CData(2,:) = [0.9290 0.6940 0.1250];
    b.CData(3,:) = [0.8500 0.3250 0.0980];
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
