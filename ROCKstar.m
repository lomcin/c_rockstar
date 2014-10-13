%
% Copyright (c) 2014, ADRL/ETHZ. Jemin Hwangbo
% All rights reserved.
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
%     * Redistributions of source code must retain the above copyright
%       notice, this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above copyright
%       notice, this list of conditions and the following disclaimer in the
%       documentation and/or other materials provided with the distribution.
%     * Neither the name of the Autonomous Systems Lab, ETH Zurich nor the
%       names of its contributors may be used to endorse or promote products
%       derived from this software without specific prior written permission.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
% ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
% WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
% DISCLAIMED. IN NO EVENT SHALL Christian Gehring, Hannes Sommer, Paul Furgale,
% Remo Diethelm BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
% OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
% GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
% HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
% OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%

%%  Rock* Algorithm/Task Parameters
unit_exploration_noise=0.05; %Starting Noise
how_often_evaluate=5; %Not an algorithm parameter. Just to give feedback to the user
lambda=0.5;
n_parameter=50;
task = task_pointmassmotion(n_parameter);
theta=task.init;
n_total_rollouts=2000;
lambdaMD=10;
initial_exp=2;
expansion_factor_sigma=1.3^(1/log(n_parameter+2.5))-1;
imp_factor=1.3;

% set up the task (initial and goal state, cost function for a rollout)
covar_init=eye(n_parameter,n_parameter);
cost2policy_cov_factor=chi2inv(0.95,n_parameter)*-0.5/log(lambda);
covar = covar_init;

% CMA parameters
[B,D]=eig(covar);

% setting global step size. This is to improve numerical stability.
% It is the best practice to use linear scaling for all parameters since
% it will remove numerical instability in calculating the determinant and the eigen vectors/values.
sigma=unit_exploration_noise;

C=covar;
determinant=det(C);
cc = 3/(n_parameter+6)/log(n_parameter+6);
ccov = 6/(n_parameter+7)/log(n_parameter+7);
pc = zeros(n_parameter,1);
chiN=n_parameter^0.5*(1-1/(4*n_parameter)+1/(21*n_parameter^2));

%% initialization
policy_history=zeros(n_total_rollouts,n_parameter);
sampling_history=zeros(n_total_rollouts,n_parameter);
cost_history=zeros(n_total_rollouts,1);
theta_history=zeros(n_total_rollouts,n_parameter);
theta_history(1,:)=theta;
learning_history = [];
counter_eval=1;
best_policy=theta;
best_cost=inf;

data.evaluation_cost=zeros(ceil(n_total_rollouts/how_often_evaluate),2);
data.evaluation_policy=zeros(ceil(n_total_rollouts/how_often_evaluate),n_parameter);
range=chi2inv(0.95,length(theta))*lambdaMD;

disp('----------------------------------')
disp('               Rock*')
disp('----------------------------------')
disp('rollout      noiseless policy cost    global step size')


%% Rock* algorithm

for iteration_n=1:n_total_rollouts

if mod(iteration_n-1,how_often_evaluate)==0
   % This is not a part of algorithm but gives learning feedback to the
   % user.
   cost_eval = task.perform_rollout(task,theta);
   data.evaluation_cost(counter_eval,1)=iteration_n-1;
   data.evaluation_cost(counter_eval,2)=cost_eval;
   data.evaluation_cost(counter_eval,:);
   data.evaluation_policy(counter_eval,:)=theta;
   counter_eval=counter_eval+1;
   disp(sprintf('%3.0f             %.4f                  %.5f',iteration_n-1,cost_eval,sigma));
end

  %------------------------------------------------------------------
  % search

  theta_eps_cur = mvnrnd(theta,covar);
  policy_history(iteration_n,:)=theta_eps_cur;
  costs_rollouts_cur = task.perform_rollout(task,theta_eps_cur);
  cost_history(iteration_n)=costs_rollouts_cur(1);

  if costs_rollouts_cur(1)<best_cost
      prev_best_cost=best_cost;
      best_cost=costs_rollouts_cur(1);
      prev_best_policy=best_policy;
      best_policy=theta_eps_cur;
  end
  
  if iteration_n>initial_exp-1

     covar_inv=inv(covar);
     
     Near_policies=zeros(n_total_rollouts,length(theta_eps_cur));
     Near_policy_costs=zeros(n_total_rollouts,1);
     
     counter=1;
     temp_coef=1;

    while(1)
      for smaple_n=1:iteration_n
        if temp_coef*range > (policy_history(smaple_n,:)-theta)*(covar_inv)*(policy_history(smaple_n,:)-theta)'
          Near_policies(counter,:)=policy_history(smaple_n,:);
          Near_policy_costs(counter,:)=cost_history(smaple_n);
          counter=counter+1;
        end
      end
      if counter>min(iteration_n,2)
        break;
      end
      temp_coef=temp_coef*2;
      counter=1;
    end
    
    Near_policies(counter:end,:)=[];
    Near_policy_costs(counter:end) = [];
    [cur_theta_new,E_cost_theta]=gradient_descent(Near_policies, Near_policy_costs,covar_inv./cost2policy_cov_factor, theta);
   
   for i=1:min(length(Near_policy_costs),5)
        [temp, IDX]=sort(Near_policy_costs);
        [cur_theta_new2,E_cost_theta2]=gradient_descent(Near_policies, Near_policy_costs, covar_inv./cost2policy_cov_factor, Near_policies(IDX(i),:));
        if E_cost_theta2<E_cost_theta
            cur_theta_new=cur_theta_new2;
        end
    end
    
    if costs_rollouts_cur<mean(Near_policy_costs)
        [cur_theta_new2,E_cost_theta2]=gradient_descent(Near_policies, Near_policy_costs, covar_inv./cost2policy_cov_factor, theta_eps_cur);
        if E_cost_theta2<E_cost_theta
            cur_theta_new=cur_theta_new2;
        end
    end
    
    % Initial from the best position
    [dump, min_idx]=sort(Near_policy_costs);
    
    if Near_policies(min_idx(1),:) ~=theta_eps_cur
      [cur_theta_new2,E_cost_theta2]=gradient_descent(Near_policies, Near_policy_costs, covar_inv./cost2policy_cov_factor, Near_policies(min_idx(1),:));

        if E_cost_theta2<E_cost_theta
        cur_theta_new=cur_theta_new2;
        end
    
    end
    
    if min(Near_policy_costs)~=best_cost
          [cur_theta_new2,E_cost_theta2]=gradient_descent(policy_history(1:iteration_n,:), cost_history(1:iteration_n), covar_inv./cost2policy_cov_factor, best_policy);

        if E_cost_theta2<E_cost_theta
        cur_theta_new=cur_theta_new2;
        end
    
    end
    
    theta_history(iteration_n,:)=cur_theta_new;
        %% Covariance Matrix Adaptation

     if cost_history(iteration_n-1)>cost_history(iteration_n)
         sigma=sigma*(1+expansion_factor_sigma);
     else
         sigma=sigma/(1+expansion_factor_sigma)^(imp_factor);
     end
     
     if sqrt((cur_theta_new-theta)*covar_inv*(cur_theta_new-theta)')<chiN*1.5

         cost2policy_cov_factor/3;
         pc=(1-cc)*pc+cc * (cur_theta_new-theta)' / sigma;
         C=(1-ccov)*C+pc*pc'*ccov;
         C=C.*(determinant/det(C)).^(1/n_parameter);
     end
     
     C = triu(C) + triu(C,1)';
     covar=C.*sigma^2;
     covar_inv=inv(covar);
     theta=cur_theta_new;
  end
  
  if sigma<1e-8
      break;
  end
  
end
save data;


%% visualization

figure('position',[10 10 1300 700],'name','Learning History');
subplot(2,3,1)
plot(data.evaluation_cost(1:counter_eval-1,1),data.evaluation_cost(1:counter_eval-1,2))
title('Learning curve');
[y, yd, ydd,time]=task.perform_rollout_evaluation(task,theta);
xlabel('number of roll-outs')
ylabel('cost')

subplot(2,3,2)
plot(time,y)
xlabel('time [s]')
ylabel('position [m]')
title('Trajectory');

subplot(2,3,3)
plot(time,yd)
xlabel('time [s]')
ylabel('velocity [m/s]')
title('Velocity');

subplot(2,3,4)
plot(time,ydd.*task.mass)
xlabel('time [s]')
ylabel('force [N]')
title('Force');
hold on
animation_handle=subplot(2,3,5:6);
title('Animation');
axis([-0.3 1.3 0 0.5])

for animation_time=1:length(time)
    pause(0.06)
    cla(animation_handle)
    box=rectangle('Position',[y(animation_time),0,0.1,0.1]);
    xlabel('position [m]')
end

