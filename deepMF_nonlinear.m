function [ Z, H, dnorm, R ] = deepMF_nonlinear ( XX, layers, varargin )

for iv = 1:length(XX)
    X1 = XX{iv};
    %X1 = NormalizeFea(X1,0);
    ans=any(X1,1);
    ind_0 = find(ans == 0);
    
    Y{iv} = X1;
    % ------------- Construct indexing matrix of missing views----------- %
    W1 = eye(size(X1,2));
    W1(:,ind_0) = [];
    XX{iv}(:,ind_0) = [];
    G{iv} = W1;                            
    Ind_ms{iv} = ind_0;
end

% each view should be initialized also.

numOfView = numel(XX);
num_of_layers = numel(layers);
numOfSample = size(XX{1,2},2);


alpha = ones(numOfView,1).*(1/numOfView);

Z = cell(numOfView, num_of_layers);
H = cell(numOfView, num_of_layers);

% Process optional arguments
pnames = { ...
    'z0' 'h0' 'bUpdateH' 'bUpdateLastH' 'maxiter' 'g', 'g_inv', ...
    'g_inv_diff', 'TolFun', 'save', 'nonlinearity', 'gnd','lamda1','lamda2','savePath' ...
};

%dflts  = {0, 0, 1, 1, 500, @(x) sqrt(x), @(x) x.*x, @(x) 2 .* x, 1e-5};
dflts  = {0, 0, 1, 1, 300, @(x) x, @(x) x, @(x) x, 1e-5, 1, 'tanh', []};

[z0, h0, bUpdateH, bUpdateLastH, maxiter, g, g_inv, g_inv_diff, tolfun, doSave, nonlinearity, gnd,lamda1,lamda2,savePath] = ...
        internal.stats.parseArgs(pnames,dflts,varargin{:});
if strcmp(nonlinearity, 'tanh') == 1
    g = @(x) 3 .* atanh(x./1.7159 ) ./2;
    g_inv = @(x) 1.7159 * tanh( 2/3 * x);
    g_inv_diff = @(x)  1.7159 * 2 / 3 .* (sech((2 .* x) ./ 3) .^ 2);
elseif strcmp(nonlinearity, 'square') == 1
    g = @(x) x .^ 0.5;
    g_inv = @(x) x .* x;
    g_inv_diff = @(x) 2 * x;
elseif strcmp(nonlinearity, 'sigmoid') == 1
    sigmoid = @(x) (1./(1+exp(-x)));
    g_inv = sigmoid;
    g_inv_diff = @(x) sigmoid(x) .* (1 - sigmoid(x));
    g = @(x) log(x ./ (1 - x));
elseif strcmp(nonlinearity, 'softplus') == 1
    g_inv = @(x) log(1 + exp(x));
    g_inv_diff = @(x) exp(x) ./ (1 + exp(x));
    g = @(x) log(exp(x) - 1);
            
end



for v_ind = 1:numOfView
    X = XX{v_ind};
%     X = bsxfun(@rdivide,X,sqrt(sum(X.^2,1)));
    if ~iscell(z0) && ~iscell(h0)
        for i_layer = 1:length(layers)
            if i_layer == 1
               
                % R{i_layer} = NormaliseFactor(X);
                V = X;
            else 
                
                % R{i_layer} = NormaliseFactor(H{i_layer-1});
                V = g(H{v_ind,i_layer-1});
            end


            display(sprintf('Initialising Layer #%d...', i_layer));

            % For the later layers we use nonlinearities as we go from
            % g(H_{k-1}) to Z*H_k
            [Z{v_ind,i_layer}, H{v_ind,i_layer}, ~] = ...
                 seminmf(V, ...
                     layers(i_layer), ...
                     'maxiter', 100, ...
                     'bUpdateH', true, 'save', doSave, 'fast', 0); 

    %         Z{i_layer} = gpuArray(Z{i_layer});
    %         H{i_layer} = gpuArray(H{i_layer});
        end

    else
        Z=z0;
        H=h0;

        display('Skipping initialization, using provided init matrices...');
    end

    
end

GG=[0];
HG=[0];
for v_ind = 1:numOfView

    GG = G{v_ind}*G{v_ind}'+GG;
    HG = H{v_ind,num_of_layers}*G{v_ind}'+ HG;

end
Hc = HG * GG^(-1) ;
for iter = 1:30  
    for v_ind = 1:numOfView
        X = XX{v_ind};
        E = ones(layers(numel(layers)));
        
%         X = bsxfun(@rdivide,X,sqrt(sum(X.^2,1)));
        dnorm(v_ind) = norm(X - deep_recon(Z(v_ind,:), H(v_ind,:), g_inv), 'fro');
        dnorm(v_ind) = deep_cost(X, Z(v_ind,:), H(v_ind,:),1,lamda1,lamda2,E,Hc,G{v_ind},g_inv);

        for i = numel(layers):-1:1

            if i == 2    
                KSI = Z{v_ind,1}' * X;
                PSI = Z{v_ind,1}' * Z{v_ind,1};
            end

            if  bUpdateH && (i < numel(layers) || (i == numel(layers) && bUpdateLastH))  

                if i == 1
                    % A = Z{1}' * X;
                    % B = Z{1}' * ((Z{1} * H{1}));

                    % [H, ~] = gd_H(X, Z, H, R, B - A, i, g_inv, dnorm);       

                    H{v_ind,1} =  g_inv(Z{v_ind,2} * H{v_ind,2});
                    H{v_ind,i}(H{v_ind,i} <= 0) = eps;
                else 
                    c = g_inv_diff(Z{v_ind,2} * H{v_ind,2});

                    A = 2*KSI;
                    B = 2*PSI  * g_inv(Z{v_ind,2} * H{v_ind,2});
                    E = ones(layers(numel(layers)));
                    F = 2*lamda1*E*H{v_ind,i};
                    P = 2*lamda2*(H{v_ind,i}-Hc*G{v_ind});
                    

                    C = Z{v_ind,2}' * ((B-A) .* c)+F+P;
                    [H(v_ind,:), ~] = gd_H(X, Z(v_ind,:), H(v_ind,:), C, i, g_inv, dnorm(v_ind),lamda1,lamda2,E,Hc,G{v_ind});  
                  
                end
            end

            dnorm(v_ind) = deep_cost(X, Z(v_ind,:), H(v_ind,:),1,lamda1,lamda2,E,Hc,G{v_ind},g_inv);

            %dnorm(v_ind) = norm(X - deep_recon(Z(v_ind,:), H(v_ind,:), g_inv), 'fro');
            % fprintf(1, 'after H(%d)...#%d error: %f\n', i, iter, dnorm);

            if i == 1
               Z{v_ind,i} = X  * pinv(g_inv(Z{v_ind,2} * H{v_ind,2}));

            else
                c =  g_inv_diff(Z{v_ind,2} * H{v_ind,2}); 
                C =((Z{v_ind,1}' * (Z{v_ind,1} * g_inv(Z{v_ind,2} * H{v_ind,2}) - X) .* c)) * H{v_ind,2}';

                [Z(v_ind,:),~] = gd_Z(X, Z(v_ind,:), H(v_ind,:), C, i, g_inv, dnorm(v_ind),lamda1,lamda2,E,Hc,G{v_ind});       
            end
            

            dnorm(v_ind) = deep_cost(X, Z(v_ind,:), H(v_ind,:),1,lamda1,lamda2,E,Hc,G{v_ind},g_inv);
            %dnorm(v_ind) = norm(X - deep_recon(Z(v_ind,:), H(v_ind,:), g_inv), 'fro');
            
            if (~mod(iter,1))
              if ~exist([savePath,'/View',num2str(v_ind),'/Z1'],'dir')
                  mkdir([savePath,'/View',num2str(v_ind),'/Z1']);
              end

              fid=fopen([savePath,'/View',num2str(v_ind),'/Z1/',num2str(iter),'.txt'],'w');	
              [r,c]=size(Z{v_ind,1});		% �õ����������������
                 for i=1:r
                  for j=1:c
                  fprintf(fid,'%f\t',Z{v_ind,1}(i,j));
                  end
                  fprintf(fid,'\r\n');
                 end
                fclose(fid);
              if ~exist([savePath,'/View',num2str(v_ind),'/Z2'],'dir')
                  mkdir([savePath,'/View',num2str(v_ind),'/Z2']);
              end              
              fid=fopen([savePath,'/View',num2str(v_ind),'/Z2/',num2str(iter),'.txt'],'w');	
              [r,c]=size(Z{v_ind,2});		% �õ����������������
                 for i=1:r
                  for j=1:c
                  fprintf(fid,'%f\t',Z{v_ind,2}(i,j));
                  end
                  fprintf(fid,'\r\n');
                 end
                fclose(fid);
              if ~exist([savePath,'/View',num2str(v_ind),'/H1'],'dir')
                  mkdir([savePath,'/View',num2str(v_ind),'/H1']);
              end                
              fid=fopen([savePath,'/View',num2str(v_ind),'/H1/',num2str(iter),'.txt'],'w');	
              [r,c]=size(H{v_ind,1});		% �õ����������������
                 for i=1:r
                  for j=1:c
                  fprintf(fid,'%f\t',H{v_ind,1}(i,j));
                  end
                  fprintf(fid,'\r\n');
                 end
                fclose(fid);
              if ~exist([savePath,'/View',num2str(v_ind),'/H2'],'dir')
                  mkdir([savePath,'/View',num2str(v_ind),'/H2']);
              end
              fid=fopen([savePath,'/View',num2str(v_ind),'/H2/',num2str(iter),'.txt'],'w');	
              [r,c]=size(H{v_ind,2});		% �õ����������������
                 for i=1:r
                  for j=1:c
                  fprintf(fid,'%f\t',H{v_ind,2}(i,j));
                  end
                  fprintf(fid,'\r\n');
                 end
                fclose(fid);
            end


            % display(sprintf('after Z(%d)...#%d error: %f', i, iter, norm(X - deep_recon(Z, H, R, g_inv), 'fro')));

        end
    end
    GG=[0];
    HG=[0];
    for v_ind = 1:numOfView
        
        GG = G{v_ind}*G{v_ind}'+GG;
        HG = H{v_ind,num_of_layers}*G{v_ind}'+ HG;
       
    end
    Hc = HG * GG^(-1) ;
    
    %assert(i == numel(layers));
    for v_ind = 1:numOfView
        % get the error for each view
        X = XX{v_ind};
        
%         X = bsxfun(@rdivide,X,sqrt(sum(X.^2,1)));
        norm_(v_ind) = deep_cost(X, Z(v_ind,:), H(v_ind,:),1,lamda1,lamda2,E,Hc,G{v_ind},g_inv);
         %norm_(v_ind) = norm(X - deep_recon(Z(v_ind,:), H(v_ind,:), g_inv), 'fro');
    end
    

    
    display(sprintf('#%d error: %f', iter, sum(norm_)));
    maxDnorm = sum(norm_);
%     if verbose
%         fid=fopen([savePath,'error.txt'],'w');
%         fprintf(fid,'%f\n',maxDnorm);
%         
%         display(sprintf('#%d error: %f', iter, maxDnorm));
%         
%         derror(iter) = maxDnorm;
%     end
   

    
    dnorm0 = maxDnorm;
    derror(iter) = maxDnorm;
    
    if (~mod(iter,1)) 
      if ~exist([savePath,'/Hc'],'dir')
          mkdir([savePath,'/Hc']);
      end
      fid=fopen([savePath,'/Hc/',num2str(iter),'.txt'],'w');	
      [r,c]=size(Hc);		% �õ����������������
         for i=1:r
          for j=1:c
          fprintf(fid,'%f\t',Hc(i,j));
          end
          fprintf(fid,'\r\n');
         end
        fclose(fid);
    end
end


fid=fopen([savePath,'error.txt'],'w');
[r,c]=size(derror);		% �õ����������������
     for i=1:r
      for j=1:c
      fprintf(fid,'%f\n',derror(i,j));
      end
      fprintf(fid,'\r\n');
     end
fclose(fid);


end
function [Z, dnorm1] = gd_Z(X, Z, H, c, i, g_inv, dnorm,lamda1,lamda2,E,Hc,G)
    eta = 0.01;
    oldZ = Z{i};
    iter = 0;
    
    while(1)
        iter = iter + 1;
        eta = eta / 2;
        Z{i} = oldZ - eta .* c;
        
        dnorm1 = norm(X - deep_recon(Z, H, g_inv), 'fro');
        dnorm1 = deep_cost(X, Z, H,1,lamda1,lamda2,E,Hc,G,g_inv);

       
        if  eta < 0.00001
           Z{i} = oldZ; 
           dnorm1 = dnorm;
           break;
        end

%          fprintf(1, 'Z(%d) eta: %f iter: %d dnorm1: %f\n', i, eta, iter, dnorm1);
%          fprintf(1, '------Z(%d) eta: %f iter: %d dnorm: %f\n', i, eta, iter, dnorm);
        if dnorm1 <= dnorm
            fprintf(1, 'Z(%d) eta: %f iter: %d dnorm: %f\n', i, eta, iter, dnorm1);
            
            break;
        end                              
        %

    end
    
end








function [H, dnorm1] = gd_H(X, Z, H, c, i, g_inv, dnorm,lamda1,lamda2,E,Hc,G)
    eta = 0.01;
    oldH = H{i};
    iter = 0;
    
    if i == 1
        dnorm = norm(X - Z{1} * H{1}, 'fro');
    end
        
    while(1)
        iter = iter + 1;
        eta = eta / 2;
        H{i} = oldH - eta .* c;
        H{i}(H{i} <= 0) = eps;
        
        if i == 1
           dnorm1 =  norm(X -  Z{1} * H{1}, 'fro');
        else
           dnorm1 = norm(X - deep_recon(Z, H, g_inv), 'fro');
           dnorm1 = deep_cost(X, Z, H,1,lamda1,lamda2,E,Hc,G,g_inv);
        end
%         fprintf(1, 'H(%d) eta: %f iter: %d dnorm1: %f\n', i, eta, iter, dnorm1);
%         fprintf(1, '-------H(%d) eta: %f iter: %d dnorm: %f\n', i, eta, iter, dnorm);
%         fprintf(1, 'H{%d}: eta: %f dnorm: %f\n', i, eta, dnorm1);

        if  eta < 0.00001
           H{i} = oldH; 
           dnorm1 = dnorm;
           break;
        end
        
        %

        if dnorm1 <= dnorm
            fprintf(1, 'H(%d) eta: %f iter: %d dnorm: %f\n', i, eta, iter, dnorm1);
           
            break;
        end                              
    end
end









