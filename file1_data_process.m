%% ����Ԥ����ѵ���� ��֤�� ���Լ����֣�
clc;clear;close all

%% ����ԭʼ����
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
% % ���ظ�������
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
    % ��Ƶ�ͨ�˲���
    [b, a] = butter(4, cutoff / (fs / 2), 'low'); % �Ľ׵�ͨ�˲���
    denoised_signal = filtfilt(b, a, signal); % ˫���˲���������λʧ��
end

%% 
N=400;
L=864;% ÿ��״̬ȡN������  ÿ����������ΪL
data=[];label=[];

cutoff_freq = 5000; % ���ý�ֹƵ��
fs = 48000;         % ����Ƶ����

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

     % ʹ�õ�ͨ�˲�ȥ��
    ori_data = lowpass_denoise(ori_data, fs, cutoff_freq); 
    
    for j=1:N
        start_point=randi(length(ori_data)-L);%���ȡһ�����
        end_point=start_point+L-1;
        data=[data ;ori_data(start_point:end_point)];
        label=[label;i];
    end    
end
%% ��ǩת�� onehot����
N=size(data,1);
output=zeros(N,3);
for i = 1:N
    output(i,label(i))=1;
end
%% ����ѵ���� ��֤������Լ� 7:2:1����
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

% % % % % % % % Ԥ����Ч�����ӻ�
% ����ʾ���ź�
fs = 48000;         % ����Ƶ��
L = 864;            % ��������
cutoff_freq = 5000; % ��ֹƵ��

% ѡȡһ��ԭʼ�ź�
ori_data = a1; % ʾ���ź�
start_point = randi(length(ori_data) - L); % ������
signal_raw = ori_data(start_point:(start_point + L - 1)); % ��ȡԭʼ�ź�

% ʹ�õ�ͨ�˲�ȥ��
[b, a] = butter(4, cutoff_freq / (fs / 2), 'low'); % ��Ƶ�ͨ�˲���
signal_filtered = filtfilt(b, a, signal_raw);      % ˫���˲�

% �����˲�ǰ����ź�
t = (0:L-1) / fs; % ʱ����
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
% ������ӻ����
% saveas(gcf, 'filter_visualization.png');
% % % % % % % % Ԥ����Ч�����ӻ�


% ����ʾ���ź�
fs = 48000;         % ����Ƶ��
L = 864;            % ��������
cutoff_freq = 5000; % ��ֹƵ��
wavename = 'cmor3-3'; % С������
totalscal = 256;     % �ܳ߶���

% ѡȡһ��ԭʼ�ź�
ori_data = a3; % ʾ���ź�
start_point = randi(length(ori_data) - L); % ������
signal_raw = ori_data(start_point:(start_point + L - 1)); % ��ȡԭʼ�ź�

% ʹ�õ�ͨ�˲�ȥ��
[b, a] = butter(4, cutoff_freq / (fs / 2), 'low'); % ��Ƶ�ͨ�˲���
signal_filtered = filtfilt(b, a, signal_raw);      % ˫���˲�

% С���任
Fc = centfrq(wavename); % ����Ƶ��
c = 2 * Fc * totalscal;
scals = c ./ (1:totalscal);
f = scal2frq(scals, wavename, 1 / fs); % ���߶�ת��ΪƵ��

% С���任��ϵ��
coefs_raw = cwt(signal_raw, scals, wavename); % ԭʼ�ź�
coefs_filtered = cwt(signal_filtered, scals, wavename); % �˲����ź�

% ���ӻ��˲�ǰ���С��ʱƵͼ
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

% % ������ӻ����
% saveas(gcf, 'wavelet_filter_visualization.png');
% 






