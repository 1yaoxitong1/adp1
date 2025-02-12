%% 
clear; clc;
% 仿真参数
T = 60;         % 仿真时间
dt = 0.01;      % 步长
t = 0:dt:T;

% 小车参数
v_max = 1;    % 最大线速度
w_max = 2.0;  % 最大角速度

% 期望轨迹圆周
xd = @(t) 2*cos(0.5*t);
yd = @(t) 2*sin(0.5*t);
theta_d = @(t) 0.5*t + pi/2;
vd = @(t) 1;  % 期望线速度
wd = @(t) 0.5;  % 期望角速度

% ADP参数
Q = diag([30, 30, 30]);   % 状态权重矩阵
R = diag([5, 5]);         % 控制输入权重矩阵                 
alpha = 0.1;              % 学习率

% 神经网络参数
phi = @(e) [e(1)^2; e(2)^2; e(3)^2; e(1)*e(2); e(1)*e(3); e(2)*e(3)];
W_hat = zeros(6,1);       % 初始权值

% 误差阈值
error_threshold = 0.02;  % 误差阈值，可根据实验调参

%% 
% 初始状态
x = 2.1; y = 0; theta = pi/2;
% 误差状态
e = zeros(3,1);
% 历史记录
e_history = zeros(3, length(t));
u_history = zeros(2, length(t));
W_history = zeros(6, length(t));
x_history = zeros(1, length(t));    % 实际轨迹x
y_history = zeros(1, length(t));    % 实际轨迹y
x_d_history = zeros(1, length(t));  % 参考轨迹x
y_d_history = zeros(1, length(t));  % 参考轨迹y

%% 
for k = 1:length(t)
    % 当前时间
    current_t = t(k);
    d_v = 0.5*exp(-0.002*current_t)*0.02 * (sin(0.5*current_t) + 0.3*cos(2.8*current_t) );
    d_w = 0.5*exp(-0.5*current_t)*0.01 * (0.5*sin(0.2*current_t) + cos(0.525*current_t));

    % 期望轨迹并记录
    x_d = xd(current_t);
    y_d = yd(current_t);
    theta_des = theta_d(current_t);
    x_d_history(k) = x_d;
    y_d_history(k) = y_d;
    
    % 误差计算
    R_mat = [cos(theta), sin(theta), 0;
            -sin(theta), cos(theta), 0;
             0,          0,          1];
    e = R_mat * [x_d - x; y_d - y; theta_des - theta];
    
    % f(e,t) 和 g(e)
    f = [wd(current_t)*e(2);
        -wd(current_t)*e(1) + vd(current_t)*sin(e(3));
        0];  
    g = [-1, e(2);
         0,  -e(1);
         0,  -1];
    
    % 控制输入
    grad_phi = [2*e(1), 0, 0;
               0, 2*e(2), 0;
               0, 0, 2*e(3);
               e(2), e(1), 0;
               e(3), 0, e(1);
               0, e(3), e(2)];
    u_nominal = [vd(current_t)*cos(theta_des-theta); wd(current_t)];
    phi_e = phi(e);

    % 计算 ADP 控制量
    u_ADP = -0.5 * inv(R) * g' * (grad_phi' * W_hat) + [d_v; d_w];
    u_total = u_nominal + u_ADP ;  

    % 计算 sigma
    sigma = grad_phi * (f + g*u_ADP); % 6×1 向量
    
    % 计算控制代价和delta
    u_cost = u_ADP' * R * u_ADP;
    delta = e' * Q * e + u_cost - sigma' * W_hat;
    
    % 误差阈值限制权值更新
    if norm(e) < error_threshold
        W_dot = zeros(size(W_hat));  % 停止更新
    else
        W_dot = -alpha * sigma * delta / (sigma' * sigma + 1)^2;  % 正则化更新
    end

    % 更新权值
    W_hat = W_hat + W_dot * dt;
   
    % 应用控制输入
    v = u_total(1);
    w = u_total(2);
    
    % 输入限幅
    v = sign(v) * min(abs(v), v_max);
    w = sign(w) * min(abs(w), w_max);

    % 记录数据
    e_history(:,k) = e;
    u_history(:,k) = [v; w];
    W_history(:,k) = W_hat;
    x_history(k) = x;
    y_history(k) = y;
    
    % 更新小车状态
    x = x + v*cos(theta)*dt;
    y = y + v*sin(theta)*dt;
    theta = theta + w*dt;
end

%% 
figure;
plot(x_d_history, y_d_history, 'r--', 'LineWidth', 1.5);  % 参考轨迹
hold on;
plot(x_history, y_history, 'b-', 'LineWidth', 1.5);       % 实际轨迹
xlabel('X ');
ylabel('Y ');
title('圆形轨迹跟踪');
legend('参考', '实际', 'Location', 'best');
grid on;
axis equal;

figure;
subplot(3,1,1);
plot(t, e_history(1,:), 'LineWidth',1.5);
title(' e_x');
grid on;

subplot(3,1,2);
plot(t, e_history(2,:), 'LineWidth',1.5);
title(' e_y');
grid on;

subplot(3,1,3);
plot(t, e_history(3,:), 'LineWidth',1.5);
title(' e_{\theta}');
grid on;

figure;
subplot(2,1,1);
plot(t, u_history(1,:), 'LineWidth',1.5);
title('线速度 v');
grid on;

subplot(2,1,2);
plot(t, u_history(2,:), 'LineWidth',1.5);
title('角速度 \omega');
grid on;

figure;
plot(t, W_history', 'LineWidth',1.5);
title('权值收敛');
legend('W_1','W_2','W_3','W_4','W_5','W_6');
grid on;
