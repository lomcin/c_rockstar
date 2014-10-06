function [ task ] = task_pointmassmotion(n_parameter)

task.name = 'pointmassmotion';
task.perform_rollout = @perform_rollout_pointmassmotion;
task.perform_rollout_evaluation = @perform_rollout_pointmassmotion_evaluation;

% Initial and goal state
task.y0 = 0;
task.yd0= 0;
task.g  = 1;
task.gv  = 0;
task.mass= 1;

% parameters
task.time = 1;
task.dt = 1/200;
task.viapoint = [0.5 0.5]';
task.init = ones(1,n_parameter)*0;
task.Noftimestep=ceil(task.time/task.dt);



  function cost = perform_rollout_pointmassmotion(task,theta)
    %theta scaling
%     theta(1)=theta(1)*0.001;
%     theta(3)=theta(3)*0.001;
%     theta(5)=theta(5)*0.001;
    
    trajectory.y=zeros(task.Noftimestep,1);
    trajectory.yd=zeros(task.Noftimestep,1);
    trajectory.ydd=zeros(task.Noftimestep,1);
    trajectory.y(1)=task.y0;
    trajectory.yd(1)=task.yd0;
    trajectory.ydd(1)=parameterization(0,theta,task.time)/task.mass;

    for time_step=2:task.Noftimestep
	  trajectory.y(time_step)=trajectory.y(time_step-1)+trajectory.yd(time_step-1)*task.dt+0.5*trajectory.ydd(time_step-1)*task.dt*task.dt;
      trajectory.yd(time_step)=trajectory.yd(time_step-1)+trajectory.ydd(time_step-1)*task.dt;
      trajectory.ydd(time_step)=parameterization((time_step-1)*task.dt,theta,task.time)/task.mass;
    end

    % Cost due to acceleration and final position and velocity
    torque = trajectory.ydd.*task.mass;
    cost = sum(0.5*torque.^2)*task.dt*0.03+(trajectory.y(task.Noftimestep)-task.g)^2+(trajectory.yd(task.Noftimestep)-task.gv)^2;
    %cost=theta(1)^2+theta(2)^2+theta(3)^2+theta(4)^2+theta(5);
  end



  function [y, yd, ydd, time] = perform_rollout_pointmassmotion_evaluation(task,theta)

%     theta(1)=theta(1)*0.001;
%     theta(3)=theta(3)*0.001;
%     theta(5)=theta(5)*0.001;
    
    trajectory.y=zeros(task.Noftimestep,1);
    trajectory.yd=zeros(task.Noftimestep,1);
    trajectory.ydd=zeros(task.Noftimestep,1);
    trajectory.time=zeros(task.Noftimestep,1);
    trajectory.y(1)=task.y0;
    trajectory.yd(1)=task.yd0;
    trajectory.ydd(1)=parameterization(0,theta,task.time)/task.mass;

    for time_step=2:task.Noftimestep
	  trajectory.y(time_step)=trajectory.y(time_step-1)+trajectory.yd(time_step-1)*task.dt;
      trajectory.yd(time_step)=trajectory.yd(time_step-1)+trajectory.ydd(time_step-1)*task.dt;
      trajectory.ydd(time_step)=parameterization((time_step-1)*task.dt,theta,task.time)/task.mass;
      trajectory.time(time_step)=time_step*task.dt;
    end
    
    y=trajectory.y;
    yd=trajectory.yd;
    ydd=trajectory.ydd.*task.mass;
    time=trajectory.time;

  end



end



function force=parameterization(t,theta,period)
	force=0;
	for i=1:length(theta)
		force=force+theta(i)*exp(-1*(t/period-(i-1)/(length(theta)-1))^2*length(theta)*length(theta));
	end
end
