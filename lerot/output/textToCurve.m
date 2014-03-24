%A = dmlread('sampleExperiment.txt');
%file = fopen('sampleExperiment.txt');
%tline = fgetl(file);
%A= tline(3:end);
%C = strsplit(A,',');
%A=fscanf(file,'%f %f',[7 inf]);
%A=A';
%plot(A);
%s = '[1.5,2.3,4.0,9.1]'

%eval(['x = {' s '}']);
text = fileread('experiment.txt');
eval(['x = [' text ']']);
plot(x);
