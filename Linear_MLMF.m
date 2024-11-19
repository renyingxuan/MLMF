% Yale Dataset -----------------------------
XX = cell(1,3);

X1=importdata('data/MELANOMA/MELANOMA_Gene_Expression.txt');
X1=importdata('data/MELANOMA/partial/MELANOMA_Gene_0.5.txt');

X2=importdata('data/MELANOMA/MELANOMA_Methy_Expression.txt');
X3=importdata('data/MELANOMA/MELANOMA_Mirna_Expression.txt');
XX{1,1} = X1.data;
XX{1,2} = X2.data;
XX{1,3} = X3.data;

fea = XX;

%%
savePath = 'MLMF_linear/MELANOMA/0.5/'; 
lamda1 = 1;
lamda2 = 1;
layers = [200,100] ;


[Hc,H] = deepMF( fea, layers,...
   'lamda1',lamda1,'lamda2',lamda2, 'savePath', savePath);

return