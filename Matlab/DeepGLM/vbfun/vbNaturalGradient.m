function prod = vbNaturalGradient(b,c,grad,isotropic)
%VBNATURALGRADIENT compute the product inverse_fisher times grad for two 
% cases: isotropic factor decompostion or rank-1 decomposition
% INPUT: 
%   grad:           the traditional gradient
%   b,c:            parameters in the factor decomposition
%   isotropic:      true if isotropic structure is used, rand-1 otherwise
% 
% OUTPUT: natural gradient
%
%   Copyright 2018 Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au) and Nghia
%   Nguyen (nghia.nguyen@sydney.edu.au).
%   
%   http://www.xxx.com
%
%   Version: 1.0
%   LAST UPDATE: April, 2018

if isotropic
    d = length(b);
    bb = b'*b;
    alpha = 1/(c^2+bb);
    omega = (2/c^2)*(d-1+c^4*alpha^2);
    kappa = (1+c^2/bb-.5*(1+c^2/bb)^2)*2*c*bb*alpha^2+2*c^3*alpha/bb;
    c2 = omega-2*c*alpha^2*kappa*bb;

    grad1 = grad(1:d);
    grad2 = grad(d+1:2*d);
    grad3 = grad(end);

    b_grad2 = b'*grad2;
    const1 = (1+c^2/bb-.5*(1+c^2/bb)^2);
    const2 = c^2*(1+c^2/bb);
    Ainv_times_grad2 = (const1*b_grad2)*b+const2*grad2;

    prod = [(b'*grad1)*b+c^2*grad1;Ainv_times_grad2+(kappa^2/c2*b_grad2)*b-(kappa/c2*grad3)*b;-kappa/c2*b_grad2+grad3/c2];
else
    % Close-form method
%     d = length(b);
%     grad1 = grad(1:d);
%     grad2 = grad(d+1:2*d);
%     grad3 = grad(2*d+1:end);
% 
%     c2 = c.^2;
%     b2 = b.^2;
% 
%     prod1 = (b'*grad1)*b+(grad1.*c2);
% 
%     alpha = 1/(1+sum(b2./c2));
%     Cminus = diag(1./c2);
%     Cminus_b = b./c2;
%     Sigma_inv = Cminus-alpha*(Cminus_b*Cminus_b');
% 
%     A11_inv = (1/(1-alpha))*((1-1/(sum(b2)+1-alpha))*(b*b')+diag(c2));
% 
%     C = diag(c);
%     A12 = 2*(C*Sigma_inv*b*ones(1,d)).*Sigma_inv;
%     A21 = A12';
%     A22 = 2*C*(Sigma_inv.*Sigma_inv)*C;
%     D = A22-A21*A11_inv*A12;
%     prod2 = A11_inv*grad2+(A11_inv*A12)*(D\A21)*(A11_inv*grad2)-(A11_inv*A12)*(D\grad3);
%     prod3 = -(D\A21)*(A11_inv*grad2)+D\grad3;
%     prod = [prod1;prod2;prod3];
    
%     % Approximation method
    d = length(b);
    grad1 = grad(1:d);
    grad2 = grad(d+1:2*d);
    grad3 = grad(2*d+1:end);

    c2 = c.^2;
    b2 = b.^2;

    prod1 = (b'*grad1)*b+(grad1.*c2);

    const = sum(b2./c2);
    const1 = 1/2+1/2/const;
    prod2 = (b'*grad2)*b+(grad2.*c2);
    prod2 = const1*prod2;
    alpha = 1/(1+const);
    x = alpha*b2./(c.^3);
    y = 1./c2 - 2*alpha*(b./c2).^2;
    aux = x./y;    
    prod3 = grad3./y-(1/(1+sum(x.^2./y)))*(aux'*grad3)*aux;
    prod3 = prod3/2;
    prod = [prod1;prod2;prod3];
end
end

