close all
clear
addpath(('./tensor_toolbox'))
addpath(('./tensor-factorization/bin'))
syms sig(x)
sig(x) = x^3 + x^2;
diff_sigma = diff(sig);

for d = 50:50:300
    % Hyperparameters
    B = 2; % Batch size
    m = 10000; % Network width
    % % d = size(XTrain,1) * size(XTrain,2);
    %d = 100; % feature dim

    % X: d X B
    X = eye(d, B);% XTrain(:,:,:,1:B)

    % Y: 1 X B

    % y = double(YTrain(1:B,:))';
    y = 2*randi([0 1], 1,B) - 1;

    % Sampled W
    mu = zeros(d, 1);
    Sigma = eye(d);
    % rng('default')  % For reproducibility

    W = mvnrnd(mu,Sigma,m); % with size m X d

    % 
    g = calculate_gradient(X, y, W, sig, diff_sigma); % m X 1
    % estimate the span of X with matrix M

    M = zeros(d,d);
    a = sum(g);
    for i = 1:d
        for j = 1:d
            M(i,j) = sum(g.*W(:,i).*W(:,j));
            if i==j
                M(i,i) = M(i,i) - a;
            end
        end
    end
    [V,D] = eigs(M,B); 
    WV = W*V; % m x B <= m x 50

    %T = zeros(d,d,d);
    T = zeros(B,B,B);
    for i = 1:B
        for j=i:B
            for k=j:B
                T(i,j,k) = sum(g.*WV(:,i).*WV(:,j).*WV(:,k));
                T(i,k,j) = T(i,j,k);
                T(j,i,k) = T(i,j,k);
                T(j,k,i) = T(i,j,k);
                T(k,i,j) = T(i,j,k);
                T(k,j,i) = T(i,j,k);
            end
        end
    end

    for i = 1:B
        for j = 1:B
            a = sum(g.*WV(:,i));
            T(i,j,j) = T(i,j,j) - a;
            T(j,i,j) = T(j,i,j) - a;
            T(j,j,i) = T(j,j,i) - a;
        end
    end
    % 
    % for i = 1:size(g,1)
    %     % g_bar = g(i,:);
    %     w_bar = W(i,:);
    %     % desired outerproduct
    %     T = T + outerProduct(g(i) * w_bar, w_bar ,w_bar) - SpecialOuterProduct(g(i) * w_bar);
    % %     g_bar(xx) .* w_bar(yy) .* w_bar(zz) - SpecialOuterProduct(g_bar); % squeeze(OuterProduct(g_bar, eye(d,d)));
    %     fprintf('node: %d done! \n', i)
    % 
    % %     calc_vector_tensor_prod(g_bar, d);
    % end

    T = T/m;


    % save('T_tensor.mat','T','-v7.3');

    fprintf('Reconsctruction starts! \n')

    [rec_X, ~, misc] = no_tenfact(T, 100, B);

    new_recX = V * rec_X; % transform

    % save('X_rec.mat','rec_X','-v7.3');
    for i = 1:B
        if min(new_recX(:,i))<-0.5
            new_recX(:,i)=-new_recX(:,i);
        end
        new_recX(:,i) = new_recX(:,i)/norm(new_recX(:,i));
    end
    if new_recX(1,1)<new_recX(1,2)
        b=new_recX(:,2);
        new_recX(:,2) = new_recX(:,1);
        new_recX(:,1)=b;
    end 
    l = norm(new_recX - X);
    fprintf('d=%d, rec_loss=%.3f\n',d,l*l/2)
end

function [V1 Lambda misc] = no_tenfact(T, L, k)

    % NO_TENFACT  Computes the CP decomposition of a tensor
    %   [V, Lambda, flops, V0] = tendecomp(T, L, k) computes
    %   the CP decomposition of a rank-k tensor via simultaneous
    %   matrix diagonalization.
    %
    %   INPUTS:
    %       T:      Third-order tensor
    %       k:      Rank of T
    %       L:      Number of random projections of T
    %
    %   OUTPUTS:
    %       V1:     (d x k) matrix of tensor components of T
    %       Lambda: (k x 1) vector of component weights
    %       misc:   Structure with extra (diagnostic) output:
    %         misc.flops:   Estimate of the number of flops performed
    %         misc.V0:      matrix components obtained only form random projection
    %
    %   The algorithm first projects T onto L random vectors to obtain·
    %   L projeted matrices, which are decomposed used the QRJ1D algorithm·
    %   for joint non-orthogonal matrix diagonalization.
    %   The resulting components are then used as plug-in estimates for the·
    %   true components in a second projection step along V0. The joint decomposit
    %   of this second set of matrices produces the final result V1.
    %
    %   Our implementation requires the MATLAB Tensor Toolbox v. 2.5 or greater.
    %   The input tensor object must be constructed using the Tensor Toolbox.
    %
    %   For more information on the method see the following papers:
    %
    %   V. Kuleshov, A. Chaganty, P. Liang, Tensor Factorization via Matrix
    %   Factorization, AISTATS 2015.
    %
    %   V. Kuleshov, A. Chaganty, P. Liang, Simultaneous diagonalization:
    %   the asymmetric, low-rank, and noisy settings. ArXiv Technical Report.
    
    p = size(T,1);
    sweeps = [0 0];
    
    % STAGE 1: Random projections
    
    M = zeros(p, p*L);
    W = zeros(p,L);
    
    for l=1:L
        W(:,l) = randn(p,1);
        W(:,l) = W(:,l) ./ norm(W(:,l));
        M(:,(l-1)*p+1:l*p) = double(ttm(tensor(T),{eye(p), eye(p), W(:,l)'}));
    endS
    
    [D, U, S] = qrj1d(M);
    
    % calculate the true eigenvalues across all matrices
    Ui = inv(U);
    
    Ui_norms = sqrt(sum(Ui.^2,1));
    Ui_normalized = bsxfun(@times, 1./Ui_norms, Ui);
    
    dot_products = Ui_normalized'*W;
    Lambdas = zeros(p,L);
    for l=1:L
        Lambdas(:,l) = (diag(D(:,(l-1)*p+1:l*p)) ./ dot_products(:,l)) .* (Ui_norms.^2)';
    end
    
    % calculate the best eigenvalues and eigenvectors
    [~, idx0] = sort(mean(abs(Lambdas),2),1,'descend');
    Lambda0 = mean(Lambdas(idx0(1:k),:),2);
    V = Ui_normalized(:, idx0(1:k));
    
    % store number of sweeps
    sweeps(1) = S.iterations;
    sweeps(2) = S.iterations;
    
    % STAGE 2: Plugin projections
    
    W=Ui_normalized;
    M = zeros(p, p*size(W,2));
    
    for l=1:size(W,2)
        w = W(:,l);
        w = w ./ norm(w);
    
        M(:,(l-1)*p+1:l*p) = double(ttm(tensor(T),{eye(p), eye(p), w'}));
    endx
    
    [D, U, S] = qrj1d(M);
    Ui = inv(U);
    Ui_norm=bsxfun(@times,1./sqrt(sum(Ui.^2)),Ui);
    V1 = Ui_norm;
    sweeps(2) = sweeps(2) + S.iterations;
    
    Lambda = zeros(p,1);
    for l=1:p
        Z = inv(V1);
        X = Z * M(:,(l-1)*p+1:l*p) * Z';
        Lambda = Lambda + abs(diag(X));
    end
    [~, idx] = sort(abs(Lambda), 'descend');
    V1 = Ui_norm(:, idx(1:k));
    
    misc = struct;
    misc.V0 = V;
    misc.sweeps = sweeps;
    
    end











    function [Y,B,varargout]=qrj1d(X,varargin)
        %QR based Jacbi-like JD; This function minimizes the cost 
        %J_{1}(B)=\sum{i=1}^{N} \|BC_{i}B^{T}-diag(BC_{i}B^{T})\|_{F}^{2}
        %where \{C_{i}\}_{i=1}^{N} is a set of N, n\times n symmetric matrices 
        %and B the joint diagonalizer sought. A related measure that is used
        %to measure the error is J_{2}=\sum{i=1}^{N} \|C_{i}-B^{-1}diag(BC_{i}B^{T})B^{-T}\|_{F}^{2}
        %
        %
        %Standard usage: [Y,B]=QRJ1D(X), 
        %Here X is a large matrix of size n\times nN which contains the 
        %matrices to be jointly diagonalized such that X=[C1,C2,...,CN], 
        %Y contains the jointly diagonalized version of the input 
        %matrices, and B is the found diagonalizer.
        %
        %
        %More controlled usage:[X,B,S,BB]=QRJ1D(X,'mode',ERR or ITER,RBALANCE): 
        %
        %'mode'='B' or 'E' or 'N':  In the 'B' mode the stopping criteria at each 
        %                           step is max(max(abs(LU-I))) which measures 
        %                           how much the diagonalizer B has changed
        %                           after a sweep. In the 'E' mode 
        %                           the stopping criterion is the difference between 
        %                           the values of the cost function J2 in two consequtive 
        %                           updates.In the 'N' mode the stopping criterion is 
        %                           the number of sweeps over L and U phases. 
        %
        %ERR: In the 'B' mode it specifies the stopping value for the change in B max(max(abs(LU-I))).
        %The default value for ERR in this mode and other modes including standard usage 
        %is ERR=10^-5. In implementation of the algorithm in order to account 
        %for dpendence of accuracy on the dimension n ERR is multiplied 
        %by n the size of matrices for JD. In the 'E' mode it ERR specifies the stopping value
        %for the relative change of J_{2} in two consequetive sweeps. 
        %In the 'B' or 'E' mode or the standard mode
        %if the change in B or relative change in J2 does not reach ERR after the default number of 
        %iterations (=200) then the program aborts and itreturns the current computed variables.
        %
        %ITER: Number of iterations in the 'N' mode  
        %
        %%RBALANCE: if given it is the period for row balancing after each sweep.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Outputs:
        %Y= the diagonalized set of matrices
        %B=the found joint diagonalizer
        %S=a structure containing some information about the run program:
        %          S.iterations: number of iterations
        %          S.LUerror: the LU error after each sweep
        %          S.J2error: the J2 error after each sweep
        %          S.J2RelativeError:the relative J2 error after each sweep
        %BB=a three dimensional array containing the joint diagonalizer after each sweep
        %Note: S and BB are not required outputs in the function call
        %
        %This algorithm is based on a paper presented in ICA2006 conference and published in Springer LNCS
        %Bijan Afsari, ''Simple LU and QR based Non-Orthogonal Matrix Joint Diagonalization''
        %%Coded by Bijan Afsari. Please forward any questions and problem to bijan@glue.umd.edu
        %v.1.1
        
        %Acknowledgements: Some data structures and implementation ideas in this code are inspired from the code for JADE
        %written by J.F. Cardoso and from the code FFDIAG written by Andreas Ziehe and Pavel Laskov
        %Disclaimer: This code is to be used only for non-commercial research purposes and the author does not
        %accept any reponsibility about its performance or fauilure
        [n,m]=size(X);N=m/n;
        
        BB=[];
        %defaulat values
        ERR=1*10^-4;RBALANCE=3;ITER=200;
        %%%
        MODE='B';
        if nargin==0, display('you must enter the data'); B=eye(n); return; end;
        if nargin==1, Err=ERR;Rbalance=RBALANCE;end;
        if nargin> 1, MODE=upper(varargin{1});
           switch MODE
           case {'B'} 
              ERR=varargin{2}; mflag='D'; if ERR >= 1, disp('Error value should be much smaller than unity');B=[];S=[]; return; end;
           case ('E')
              ERR=varargin{2};mflag='E'; if ERR >=1, disp('Error value should be much smaller than unity'); B=[];S=[];return;end;
           case ('N');mflag='N'; ITER=varargin{2}; ERR=0; if ITER <= 1, disp('Number of itternations should be higher than one');B=[];S=[];return;end;
           end
        end;
        if nargin==4, RBALANCE=varargin{3}; if ceil(RBALANCE)~=RBALANCE | RBALANCE<1, disp('RBALANCE should be a positive integer');B=[];S=[];return;end;end;
        JJ=[];EERR=[]; EERRJ2=[];  
        X1=X;
        B=eye(n,n);Binv=eye(n);
        J=0;
        
        for t=1:N
           J=J+norm(X1(:,(t-1)*n+1:t*n)-diag(diag(X(:,(t-1)*n+1:t*n))),'fro')^2;
        end
        JJ=[JJ,J];
        
        %err=10^-3;
        %the following part implements a sweep 
        %%%%%%%%%%%%%%%%%%%%%%%%%%
        err=ERR*n+1;
        if MODE=='B', ERR=ERR*n;end,
           k=0;
           while err>ERR & k<ITER
              k=k+1;
              
              L=eye(n);%Linv=eye(n);
              U=eye(n);%Uinv=eye(n);
              Dinv=eye(n);
        
              
        for i=2:n,
           for j=1:i-1,
              G=[-X(i,[i:n:m])+X(j,[j:n:m]);-2*X(i,[j:n:m])];
              [U1,D1,V1]=svd(G*G');
              v=U1(:,1);
              tetha=1/2*atan(v(2)/v(1));
              c=cos(tetha);
              s=sin(tetha);
              h1=c*X(:,[j:n:m])-s*X(:,[i:n:m]);
              h2=c*X(:,[i:n:m])+s*X(:,[j:n:m]);
              X(:,[j:n:m])=h1;
              X(:,[i:n:m])=h2;
              h1=c*X(j,:)-s*X(i,:);
              h2=s*X(j,:)+c*X(i,:);
              X(j,:)=h1;
              X(i,:)=h2;
              %h1=c*U(:,j)+s*U(:,i);
              %h2=-s*U(:,j)+c*U(:,i);    
              h1=c*U(j,:)-s*U(i,:);
              h2=s*U(j,:)+c*U(i,:);
              U(j,:)=h1;
              U(i,:)=h2;
        
              % % sort the rows
              % diagsums = zeros(n,1);
              % for l=1:L
              %     diagsums = diagsums + abs(diag(X(:,(l-1)*n+1:l*n)));
              % end
              
              % if diagsums(q,q) > diagsums(p,p)
              %   I = eye(m);
              %   P = eye(m);
              %   P(p,:) = I(q,:);
              %   P(q,:) = I(p,:);
        
              %   for l=1:L
              %       A(:,(l-1)*m+1:l*m)=P*A(:,(l-1)*m+1:l*m)*P';
              %   end
              %   V=V*P';
              % end
        
              % I = eye(n);
              % [~, idx] = sort(abs(diagsums), 'descend');
              %
              % P = I(idx,:);
              % for l=1:L
              %     A(:,(l-1)*n+1:l*n)=P*X(:,(l-1)*n+1:l*n)*P';
              % end
              % U=U*P';
            
            end;%end for i
         end;%end for j
              for i=1:n
                 %for j=i+1:n
                 
                 rindex=[];
                 Xj=[];
                 for j=i+1:n
                    cindex=1:m;
                    cindex(j:n:m)=[];
                    a=-(X(i,cindex)*X(j,cindex)')/(X(i,cindex)*X(i,cindex)');
                    %coorelation quefficient
                    %a=-(X(i,cindex)*X(j,cindex)')/(norm(X(i,cindex))*norm(X(j,cindex)));
                    %a=tanh(a);
                    if abs(a)>1, a=sign(a)*1; end;
                    X(j,:)=a*X(i,:)+X(j,:);
                    I=i:n:m;
                    J=j:n:m;
                    X(:,J)=a*X(:,I)+X(:,J);
                    L(j,:)=L(j,:)+a*L(i,:);
                    %Linv(j,:)=Linv(j,:)-a*Linv(i,:);
                 end%end loop over j
              end
              
              
              B=L*U*B;%Binv=Binv*Uinv*Linv;
              %err=norm(L*U-eye(n,n),'fro');
              err=max(max(abs(L*U-eye(n))));EERR=[EERR,err];
              if rem(k,RBALANCE)==0
                 d=sum(abs(X')); 
                 D=diag(1./d*N); Dinv=diag(d*N);
                 J=0;
                 for t=1:N
                    X(:,(t-1)*n+1:t*n)=D*X(:,(t-1)*n+1:t*n)*D;
                 end;
                 B=D*B; %Binv=Binv*Dinv;
              end
              J=0;
              BB(:,:,k)=B;
              Binv=inv(B);
              for t=1:N
                 J=J+norm(X1(:,(t-1)*n+1:t*n)-Binv*diag(diag(X(:,(t-1)*n+1:t*n)))*Binv','fro')^2;
              end
              JJ=[JJ,J];
              if MODE=='E', err=abs(JJ(end-1)-JJ(end))/JJ(end-1);EERRJ2=[EERRJ2,err];end
           end
           Y=X;
           S=struct('iterations',k,'LUerror',EERR,'J2error',JJ,'J2RelativeError',EERRJ2);varargout{1}=S;varargout{2}=BB;