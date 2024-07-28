load('hz_donut.mat')

save_path = '/Users/hv099/Documents/lcneuro/propofol/Neuroblox.jl/'
ses_title = 'cmc_test_output';

dt = 0.01;
%Hz = [0.0078 0.0116 0.0154 0.0192 0.0229 0.0267 0.0305 0.0343 0.0381...
%    0.0418 0.0456 0.0494 0.0532 0.0570 0.0607 0.0645 0.0683 0.0721...
%    0.0759 0.0796 0.0834 0.0872 0.0910 0.0948 0.0985 0.1023 0.1061...
%    0.1099 0.1137 0.1174 0.1212 0.1250];
hE = 8;
ihC = 128;


N = 4;
x = zeros(N, 6);
data = readmatrix('/Users/hv099/Documents/lcneuro/propofol/Neuroblox.jl/cmc_test_sol_output.csv');
data = [data(:,3) data(:,5) data(:,7) data(:,9)]

pE.A = zeros(N);
for i = 1:N
    for j = 1:N
        if i ~= j
            pE.A(i, j) = 0.0078125;
        end
    end
end

pC = zeros((N^2+(3*N)+6));
for i = 1:N^2
    for j = 1:N^2
        if i == j
            pC(i, j) = 0.015625;
        end
    end
end

for i = N^2+N+1:N^2+(2*N)+2
    for j = N^2+N+1:N^2+(2*N)+2
        if i == j
            pC(i, j) = 0.00390625;
        end
    end
end
for i = N^2+(2*N)+3:N^2+(3*N)+6
    for j = N^2+(2*N)+3:N^2+(3*N)+6
        if i == j
            pC(i, j) = 0.015625;
        end
    end
end

save([save_path, ses_title, '.mat'],'pC','pE','hE','ihC','x','dt','Hz','data');
