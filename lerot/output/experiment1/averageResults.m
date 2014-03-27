function y = averageResults(x,k,iters)
y = zeros(iters,1);
for i = (1:iters)
    tmp = zeros(k,1);
    for j = (1:k)
        tmp(j) = i+(k*iters)-iters;
    end
    y(i) = mean(x(tmp));    
end
end