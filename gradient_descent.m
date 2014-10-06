function [cur_theta_new,E_temp_prev] = gradient_descent(Near_policies, Near_costs, cov_inv, initial)

%  Input:   Near_policies, a Matrix where each column is a experienced
%  parameter vector. Dimension=Number of variables to learn X Number of
%  experiences obtained (or depends on how many you pick for simplification).
%           
%           Near_costs, a row vector where each element corresponds to cost
%           of the Near_policies. Dimension=Number of experiences obtained
%           
%           cov, noise covariance. Dimension=number of variables^2
%
%           cov_inv= inverse of cov (it is calculated prior to shorten
%           calculation time)
%
%           initial= initial point of gradient descent
%
%           cost2policy_cov_factor= currently not used
%
%  Output: cur_theta_new, minimum vector found by gradient descent,
%           Dimension= same as policy vector
%          E_temp_prev= expected cost at cur_theta_new
%
%          Note that multiple initialization is needed in more complex cost
%          functions
%


    if length(Near_costs)<11
        mean_cost=(max(Near_costs)+min(Near_costs))/2;
        %mean_cost=max(Near_costs);
    else
        sorted=sort(Near_costs);
        mean_cost=mean(sorted(1:5));
    end
    
    Pprior=1;
    Prev_policy=initial;
    counter=0;
    
    while(1)
       
        % Calculate the gradient. It is an implementation of quotient rule
        % of derivative which is (adotb-bdota)/b2
        
        initial_repmat=repmat(Prev_policy,length(Near_costs),1);
    	diffOftheta=Near_policies-initial_repmat;
        MD2=dot((-0.25*diffOftheta*cov_inv)',diffOftheta');
        residual_term=diffOftheta*cov_inv.*0.5;
        expMD2=exp(MD2);
        expMD2Residual=expMD2*(residual_term);
        adotb=expMD2.*Near_costs'*(residual_term).*(sum(expMD2)+Pprior);
        bdota=expMD2Residual.*(sum(expMD2.*Near_costs')+Pprior*mean_cost);
        b2=(sum(expMD2)+Pprior)^2;
        Jaco=(adotb-bdota)./b2;
        
        % Evaluate the policy
        if(counter==0)
            alpha=inv(2*cov_inv)/(mean_cost-min(Near_costs));
        else
            
        end
        Prev_Jaco_norm=norm(Jaco);
   
        New_Policy=Prev_policy-Jaco*alpha;

        % Compare and update
        if (Jaco*alpha)*cov_inv*(Jaco*alpha)'<0.0001||(New_Policy-initial)*cov_inv*(New_Policy-initial)'>2||counter>200
           break
        end
        Prev_policy=New_Policy;
         counter=counter+1;
    end
    %counter
    cur_theta_new=Prev_policy;
    initial_repmat=repmat(Prev_policy,length(Near_costs),1);
    diffOftheta=Near_policies-initial_repmat;
    expMD2=exp(dot((-0.5*diffOftheta*cov_inv)',diffOftheta'));
    E_temp_prev=(dot(expMD2,Near_costs')+mean_cost*Pprior)/(sum(expMD2)+Pprior);