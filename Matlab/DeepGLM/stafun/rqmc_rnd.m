function f = rqmc_rnd(S,d)
% generate a matrix of RQMC of size S times d
max_sobol = 1111;
r = floor(d/max_sobol);
s = d-r*max_sobol;
if r>=1
    f = gen_Sobol(ceil(log2(S)),max_sobol)'; 
    for i = 2:r
        f = [f,gen_Sobol(ceil(log2(S)),max_sobol)']; 
    end
    f = [f,gen_Sobol(ceil(log2(S)),s)']; 
else
    f = gen_Sobol(ceil(log2(S)),d)'; 
end
    
end
