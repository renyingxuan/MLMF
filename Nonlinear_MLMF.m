% Yale Dataset -----------------------------
XX = cell(1,3);

X1=importdata('data/LSCC/LUNG_Gene_Expression.txt');
X1=importdata('data/LSCC/partial/LUNG_Gene_0.1.txt');
X2=importdata('data/LSCC/LUNG_Methy_Expression.txt');
X3=importdata('data/LSCC/LUNG_Mirna_Expression.txt');
XX{1,1} = X1.data;
XX{1,2} = X2.data;
XX{1,3} = X3.data;

fea = XX;

%%
savePath = 'MLMF_nonlinear/LUNG/0.1/';
lamda1 = 1;
lamda2 = 1;
layers = [100,50] ;


[Hc,H] = deepMF_nonlinear( fea, layers,...
   'lamda1',lamda1,'lamda2',lamda2, 'savePath', savePath);

return