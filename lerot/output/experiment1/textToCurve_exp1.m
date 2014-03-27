files = dir('27-03-2014/*.txt');
colour_array = ['r','b','g','y','c'];
total_iterations = 5000

%%%%%%%%%%%%%%%%%%%%%%%% k = 1 %%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% test instances %%%%%%
figure(1);
hold on;
colour_iter = 1;
for file = files(1:2:9)';
    text = fileread(file.name);
    eval(['x = [' text '];']);
    plot(x,colour_array(colour_iter));
    colour_iter = colour_iter + 1;
end
xlabel('iterations'), ylabel('NDCG'), title('test instances, k = 1');
legend('d=2','d=3','d=4','d=5','d=6','Location','SouthEast');
hold off;

%%%%%% train instances %%%%%%
figure(2);
hold on;
colour_iter = 1;
for file = files(2:2:10)';
    text = fileread(file.name);
    eval(['x = [' text '];']);
    plot(x,colour_array(colour_iter));
    colour_iter = colour_iter + 1;
end
xlabel('iterations'), ylabel('NDCG'), title('training instances, k = 1');
legend('d=2','d=3','d=4','d=5','d=6','Location','SouthEast');
hold off;

%%%%%%%%%%%%%%%%%%%%%%%% k = 2 %%%%%%%%%%%%%%%%%%%%%%%%%%
k = 2;
iters = total_iterations / k;
%%%%%% test instances %%%%%%
figure(3);
hold on;
colour_iter = 1;
for file = files(11:2:19)';
    text = fileread(file.name);
    eval(['x = [' text '];']);
    x = averageResults(x,k,iters);
    plot(x,colour_array(colour_iter));
    colour_iter = colour_iter + 1;
end
xlabel('iterations'), ylabel('NDCG'), title('test instances, k = 2');
legend('d=2','d=3','d=4','d=5','d=6','Location','SouthEast');
hold off;

%%%%%% train instances %%%%%%
figure(4);
hold on;
colour_iter = 1;
for file = files(12:2:20)';
    text = fileread(file.name);
    eval(['x = [' text '];']);
    x = averageResults(x,k,iters);
    plot(x,colour_array(colour_iter));
    colour_iter = colour_iter + 1;
end
xlabel('iterations'), ylabel('NDCG'), title('training instances, k = 2');
legend('d=2','d=3','d=4','d=5','d=6','Location','SouthEast');
hold off;

%%%%%%%%%%%%%%%%%%%%%%%% k = 5 %%%%%%%%%%%%%%%%%%%%%%%%%%
k = 5;
iters = total_iterations / k;
%%%%%% test instances %%%%%%
figure(5);
hold on;
colour_iter = 1;
for file = files(21:2:29)';
    text = fileread(file.name);
    eval(['x = [' text '];']);
    x = averageResults(x,k,iters);
    plot(x,colour_array(colour_iter));
    colour_iter = colour_iter + 1;
end
xlabel('iterations'), ylabel('NDCG'), title('test instances, k = 5');
legend('d=2','d=3','d=4','d=5','d=6','Location','SouthEast');
hold off;

%%%%%% train instances %%%%%%
figure(6);
hold on;
colour_iter = 1;
for file = files(22:2:30)';
    text = fileread(file.name);
    eval(['x = [' text '];']);
    x = averageResults(x,k,iters);
    plot(x,colour_array(colour_iter));
    colour_iter = colour_iter + 1;
end
xlabel('iterations'), ylabel('NDCG'), title('training instances, k = 5');
legend('d=2','d=3','d=4','d=5','d=6','Location','SouthEast');
hold off;

%%%%%%%%%%%%%%%%%%%%%%%% k = 10 %%%%%%%%%%%%%%%%%%%%%%%%%%
k = 10;
iters = total_iterations / k;
%%%%%% test instances %%%%%%
figure(7);
hold on;
colour_iter = 1;
for file = files(31:2:39)';
    text = fileread(file.name);
    eval(['x = [' text '];']);
    x = averageResults(x,k,iters);
    plot(x,colour_array(colour_iter));
    colour_iter = colour_iter + 1;
end
xlabel('iterations'), ylabel('NDCG'), title('test instances, k = 10');
legend('d=2','d=3','d=4','d=5','d=6','Location','SouthEast');
hold off;

%%%%%% train instances %%%%%%
figure(8);
hold on;
colour_iter = 1;
for file = files(32:2:40)';
    text = fileread(file.name);
    eval(['x = [' text '];']);
    x = averageResults(x,k,iters);
    plot(x,colour_array(colour_iter));
    colour_iter = colour_iter + 1;
end
xlabel('iterations'), ylabel('NDCG'), title('training instances, k = 10');
legend('d=2','d=3','d=4','d=5','d=6','Location','SouthEast');
hold off;