% full = csvread("k_1d_3full_test.txt");
% rem = csvread("k_1d_3rem_test.txt");

% full = csvread("k_1d_3full_train.txt");
% rem = csvread("k_1d_3rem_train.txt");

% full = csvread("k_10d_3full_test.txt");
% rem = csvread("k_10d_3rem_test.txt");

full = csvread("k_10d_3full_train.txt");
rem = csvread("k_10d_3rem_train.txt");

x = 1:length(full);

plot(x,rem,'-',x,full,'-.');
xlabel('x (iterations)');
ylabel('y (NDCG)');
title('Plot of DBGD vs REM-DBGD k = 10 d = 3');
legend('REM-DBGD','DBGD');
