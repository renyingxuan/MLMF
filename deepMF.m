function [ Hc ,H ] = deepMF( XX, layers, varargin )
% contain all the stuffs,
% including graph and beta between lost and graph term

% Process optional arguments
pnames = { ...
    'z0' 'h0' 'bUpdateH' 'bUpdateLastH' 'maxiter' 'TolFun', ...
    'verbose', 'bUpdateZ', 'cache', 'gnd','lamda1','lamda2', 'savePath'...
    };


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

dflts  = {0, 0, 1, 1, 100, 1e-5, 1, 1, 0, 0};

[z0, h0, bUpdateH, bUpdateLastH, maxiter, tolfun, verbose, bUpdateZ, cache, gnd, lamda1,lamda2,savePath] = ...
    internal.stats.parseArgs(pnames,dflts,varargin{:});





for v_ind = 1:numOfView
    X = XX{v_ind};
    X = bsxfun(@rdivide,X,sqrt(sum(X.^2,1)));
    
    if  ~iscell(h0)
        for i_layer = 1:length(layers)
            if i_layer == 1
                % For the first layer we go linear from X to Z*H, so we use id
                V = X;
            else
                V = H{v_ind,i_layer-1};
            end
            
            if verbose
                display(sprintf('Initialising Layer #%d with k=%d with size(V)=%s...', i_layer, layers(i_layer), mat2str(size(V))));
            end
            if ~iscell(z0)
                % For the later layers we use nonlinearities as we go from
                % g(H_{k-1}) to Z*H_k
                [Z{v_ind,i_layer}, H{v_ind,i_layer}, ~] = ...
                    seminmf(V, ...
                    layers(i_layer), ...
                    'maxiter', maxiter, ...
                    'bUpdateH', true, 'bUpdateZ', bUpdateZ, 'verbose', verbose, 'save', cache, 'fast', 0);
            else
                display('Using existing Z');
                [Z{v_ind,i_layer}, H{v_ind,i_layer}, ~] = ...
                    seminmf(V, ...
                    layers(i_layer), ...
                    'maxiter', 1, ...
                    'bUpdateH', true, 'bUpdateZ', 0, 'z0', z0{i_layer}, 'verbose', verbose, 'save', cache, 'fast', 1);
            end
        end
        
    else
        Z=z0;
        H=h0;
        
        if verbose
            display('Skipping initialization, using provided init matrices...');
        end
    end
    
    
%     rand('seed',220);
%     Hc = rand(layers(num_of_layers),numOfSample);
    %dnorm0(v_ind) = cost_function(X, Z(v_ind,:), H(v_ind,:));
    %dnorm(v_ind) = dnorm0(v_ind) + 1;
    
    if verbose
        %display(sprintf('#%d error: %f', 0, sum(dnorm0)));
    end
end

GG=[0];
HG=[0];
for v_ind = 1:numOfView

    GG = G{v_ind}*G{v_ind}'+GG;
    HG = H{v_ind,num_of_layers}*G{v_ind}'+ HG;

end
Hc = HG * GG^(-1) ;
%% Error Propagation

if verbose
    display('Finetuning...');
end
H_err = cell(numOfView, num_of_layers);
derror = [];
for iter = 1:20
    Hm_a = 0; Hm_b = 0;
    for v_ind = 1:numOfView
        X = XX{v_ind};
        
        X = bsxfun(@rdivide,X,sqrt(sum(X.^2,1)));
        
        H_err{v_ind,numel(layers)} = H{v_ind,numel(layers)};
        for i_layer = numel(layers)-1:-1:1
            H_err{v_ind,i_layer} = Z{v_ind,i_layer+1} * H_err{v_ind,i_layer+1};
        end
        
        for i = 1:numel(layers)
            if bUpdateZ
                try
                    if i == 1
                        Z{v_ind,i} = X  * pinv(H_err{v_ind,1});
                    else
                        Z{v_ind,i} = pinv(D') * X * pinv(H_err{v_ind,i});
                    end
                catch
                    display(sprintf('Convergance error %f. min Z{i}: %f. max %f', norm(Z{v_ind,i}, 'fro'), min(min(Z{v_ind,i})), max(max(Z{v_ind,i}))));
                end
            end
            
            if i == 1
                D = Z{v_ind,1}';
            else
                D =  Z{v_ind,i}' * D;
            end
        
            if bUpdateH && (i < numel(layers))
                % original one
                A = D * X;

                Ap = (abs(A)+A)./2;
                An = (abs(A)-A)./2;

                % original noe
                B = D * D';
                Bp = (abs(B)+B)./2;
                Bn = (abs(B)-B)./2;


                % update graph part


                H{v_ind,i} = H{v_ind,i} .* sqrt((Ap + Bn* H{v_ind,i} ) ./ max(An + Bp* H{v_ind,i}, 1e-10));
            end
                % set H{v_ind,n_of_layer} = Hm
                % update the last consensus layer
                % update Hm  
            if (i == numel(layers)) && bUpdateLastH
                B = D * XX{v_ind} + lamda2 * Hc * G{v_ind};
                E = ones(layers(numel(layers)));
                I = eye(layers(numel(layers)));
                C = D * D' + lamda1 * E + lamda2 * I;
                Ba = (abs(B)+B)./2;
                Bb = (abs(B)-B)./2;
                Ca = (abs(C)+C)./2;
                Cb = (abs(C)-C)./2;
                A=H{v_ind,i};
                Hm_a = Ba + Cb*A;
                Hm_b = Bb + Ca*A;
                H{v_ind,i} = H{v_ind,i} .* sqrt(Hm_a ./ Hm_b);
            end
      
      if (~mod(iter,1))
          
          if ~exist([savePath,'/View',num2str(v_ind),'/Z1'],'dir')
              mkdir([savePath,'/View',num2str(v_ind),'/Z1']);
          end


          fid=fopen([savePath,'/View',num2str(v_ind),'/Z1/',num2str(iter),'.txt'],'w');	
          [r,c]=size(Z{v_ind,1});		% 得到矩阵的行数和列数
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
          [r,c]=size(Z{v_ind,2});		% 得到矩阵的行数和列数
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
          [r,c]=size(H{v_ind,1});		% 得到矩阵的行数和列数
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
          [r,c]=size(H{v_ind,2});		% 得到矩阵的行数和列数
             for i=1:r
              for j=1:c
              fprintf(fid,'%f\t',H{v_ind,2}(i,j));
              end
              fprintf(fid,'\r\n');
             end
            fclose(fid);
            %{
            fid=fopen([savePath,'/V',num2str(v_ind),'/H3/',num2str(iter),'.txt'],'w');
            [r,c]=size(H{v_ind,3});		% 得到矩阵的行数和列数
             for i=1:r
              for j=1:c
              fprintf(fid,'%f\t',H{v_ind,3}(i,j));
              end
              fprintf(fid,'\r\n');
             end
            fclose(fid);  
            %}
      end
       
          %assert(i == numel(layers));
        end       
        
    end
    %update Hc
    GG=[0];
    HG=[0];
    for v_ind = 1:numOfView
        
        GG = G{v_ind}*G{v_ind}'+GG;
        HG = H{v_ind,num_of_layers}*G{v_ind}'+ HG;
       
    end
    Hc = HG * GG^(-1) ;
        
        
        
        
    for v_ind = 1:numOfView
        % get the error for each view
        X = XX{v_ind};
        
        X = bsxfun(@rdivide,X,sqrt(sum(X.^2,1)));
        dnorm(v_ind) = cost_function(X, Z(v_ind,:), H(v_ind,:),1,lamda1,lamda2,E,Hc,G{v_ind});
    end




    
    
    
    
    % finish update Z H and other variables in each view
    % disp result
    
    maxDnorm = sum(dnorm);
    if verbose
        fid=fopen([savePath,'error.txt'],'w');
        fprintf(fid,'%f\n',maxDnorm);
        
        display(sprintf('#%d error: %f', iter, maxDnorm));
        
        derror(iter) = maxDnorm;
    end
   

    
    dnorm0 = maxDnorm;
 
  if (~mod(iter,1)) 
      if ~exist([savePath,'/Hc'],'dir')
          mkdir([savePath,'/Hc']);
      end
      fid=fopen([savePath,'/Hc/',num2str(iter),'.txt'],'w');	
      [r,c]=size(Hc);		% 得到矩阵的行数和列数
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
[r,c]=size(derror);		% 得到矩阵的行数和列数
     for i=1:r
      for j=1:c
      fprintf(fid,'%f\n',derror(i,j));
      end
      fprintf(fid,'\r\n');
     end
fclose(fid);

end


