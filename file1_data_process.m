%% 数据预处理（训练集 验证集 测试集划分）
clc;clear;close all

%% 加载原始数据
load 0HP/48k_Drive_End_B007_0_122;    a2=X122_DE_time'; %1
load 0HP/48k_Drive_End_B014_0_189;    a3=X189_DE_time'; %2
load 0HP/48k_Drive_End_B021_0_226;    a4=X226_DE_time'; %3
load 0HP/48k_Drive_End_IR007_0_109;   a5=X109_DE_time'; %4
load 0HP/48k_Drive_End_IR014_0_174 ;  a6=X173_DE_time';%5
load 0HP/48k_Drive_End_IR021_0_213 ;  a7=X213_DE_time';%6
load 0HP/48k_Drive_End_OR007@6_0_135 ;a8=X135_DE_time';%7
% load 0HP/48k_Drive_End_OR014@6_0_201 ;a9=X201_DE_time';%8
% load 0HP/48k_Drive_End_OR021@6_0_238 ;a10=X238_DE_time';%9
load 0HP/normal_0_97                 ;a1=X097_DE_time';%10
% % 加载个人数据
% load 0.7mm/Healthy.mat;    a1=DE'; %1
% load 0.7mm/inner_fault.mat;    a2=DE'; %1
% load 0.7mm/outer_fault.mat;    a3=DE'; %1
% load real_data1/3;    a5=DE'; %1
% load 0HP/48k_Drive_End_B007_0_122;    a2=X122_DE_time'; %2
% load 0HP/48k_Drive_End_IR007_0_109;   a3=X109_DE_time'; %3
% load 0HP/48k_Drive_End_OR007@6_0_135; a4=X135_DE_time'; %4
% load real_data/normal-40960-0;    a2=DE'; %1
% load real_data/GD-1-40960-0;    a3=DE'; %1
% load real_data/GL-1-40960-0;    a4=DE'; %1
% load real_data/ND-1-40960-0;    a5=DE'; %1
% load real_data/NL-1-40960-0;    a6=DE'; %1
% load real_data/WD-1-40960-0;    a7=DE'; %1
% load real_data/WL-1-40960-0;    a8=DE'; %1

function denoised_signal = lowpass_denoise(signal, fs, cutoff)
    % 设计低通滤波器
    [b, a] = butter(4, cutoff / (fs / 2), 'low'); % 四阶低通滤波器
    denoised_signal = filtfilt(b, a, signal); % 双向滤波，避免相位失真
end

%% 
N=400;
L=864;% 每种状态取N个样本  每个样本长度为L
data=[];label=[];

cutoff_freq = 5000; % 设置截止频率
fs = 48000;         % 采样频率数

for i=1:8
    if i==1;ori_data=a1;end
    if i==2;ori_data=a2;end
    if i==3;ori_data=a3;end
    if i==4;ori_data=a4;end
    if i==5;ori_data=a5;end
    if i==6;ori_data=a6;end
    if i==7;ori_data=a7;end
    if i==8;ori_data=a8;end
    % if i==9;ori_data=a9;end
    % if i==10;ori_data=a10;end

     % 使用低通滤波去噪
    ori_data = lowpass_denoise(ori_data, fs, cutoff_freq); 
    
    for j=1:N
        start_point=randi(length(ori_data)-L);%随机取一个起点
        end_point=start_point+L-1;
        data=[data ;ori_data(start_point:end_point)];
        label=[label;i];
    end    
end
%% 标签转换 onehot编码
N=size(data,1);
output=zeros(N,3);
for i = 1:N
    output(i,label(i))=1;
end
%% 划分训练集 验证集与测试集 7:2:1比例
n=randperm(N);
m1=round(0.7*N);
m2=round(0.9*N);
train_X=data(n(1:m1),:);
train_Y=output(n(1:m1),:);

valid_X=data(n(m1+1:m2),:);
valid_Y=output(n(m1+1:m2),:);

test_X=data(n(m2+1:end),:);
test_Y=output(n(m2+1:end),:);

save data_process33 train_X train_Y valid_X valid_Y test_X test_Y

% % % % % % % % 预处理效果可视化
% 加载示例信号
fs = 48000;         % 采样频率
L = 864;            % 采样点数
cutoff_freq = 5000; % 截止频率

% 选取一段原始信号
ori_data = a1; % 示例信号
start_point = randi(length(ori_data) - L); % 随机起点
signal_raw = ori_data(start_point:(start_point + L - 1)); % 截取原始信号

% 使用低通滤波去噪
[b, a] = butter(4, cutoff_freq / (fs / 2), 'low'); % 设计低通滤波器
signal_filtered = filtfilt(b, a, signal_raw);      % 双向滤波

% 绘制滤波前后的信号
t = (0:L-1) / fs; % 时间轴
figure;
subplot(2, 1, 1);
plot(t, signal_raw, 'b');
title('Raw signal');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

subplot(2, 1, 2);
plot(t, signal_filtered, 'r');
title('Filtered signal');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;
% 保存可视化结果
% saveas(gcf, 'filter_visualization.png');
% % % % % % % % 预处理效果可视化


% 加载示例信号
fs = 48000;         % 采样频率
L = 864;            % 采样点数
cutoff_freq = 5000; % 截止频率
wavename = 'cmor3-3'; % 小波类型
totalscal = 256;     % 总尺度数

% 选取一段原始信号
ori_data = a3; % 示例信号
start_point = randi(length(ori_data) - L); % 随机起点
signal_raw = ori_data(start_point:(start_point + L - 1)); % 截取原始信号

% 使用低通滤波去噪
[b, a] = butter(4, cutoff_freq / (fs / 2), 'low'); % 设计低通滤波器
signal_filtered = filtfilt(b, a, signal_raw);      % 双向滤波

% 小波变换
Fc = centfrq(wavename); % 中心频率
c = 2 * Fc * totalscal;
scals = c ./ (1:totalscal);
f = scal2frq(scals, wavename, 1 / fs); % 将尺度转换为频率

% 小波变换求系数
coefs_raw = cwt(signal_raw, scals, wavename); % 原始信号
coefs_filtered = cwt(signal_filtered, scals, wavename); % 滤波后信号

% 可视化滤波前后的小波时频图
figure;
subplot(2, 1, 1);
imagesc((0:L-1)/fs, f, abs(coefs_raw) / max(max(abs(coefs_raw))));
axis xy;
title('WT of the original signal');
xlabel('Time (s)');
ylabel('Frequency (Hz)');
colorbar;

subplot(2, 1, 2);
imagesc((0:L-1)/fs, f, abs(coefs_filtered) / max(max(abs(coefs_filtered))));
axis xy;
title('Filtered WT');
xlabel('Time (s)');
ylabel('Frequency (Hz)');
colorbar;

% % 保存可视化结果
% saveas(gcf, 'wavelet_filter_visualization.png');
% 






