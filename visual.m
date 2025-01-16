clc; clear; close all;

%% 加载原始数据
load 0HP/48k_Drive_End_B007_0_122;    a2=X122_DE_time'; % 1
load 0HP/48k_Drive_End_B014_0_189;    a3=X189_DE_time'; % 2
load 0HP/48k_Drive_End_B021_0_226;    a4=X226_DE_time'; % 3
load 0HP/48k_Drive_End_IR007_0_109;   a5=X109_DE_time'; % 4
load 0HP/48k_Drive_End_IR014_0_174 ;  a6=X173_DE_time'; % 5
load 0HP/48k_Drive_End_IR021_0_213 ;  a7=X213_DE_time'; % 6
load 0HP/48k_Drive_End_OR007@6_0_135 ;a8=X135_DE_time'; % 7
load 0HP/48k_Drive_End_OR014@6_0_201 ;a9=X201_DE_time'; % 8
load 0HP/48k_Drive_End_OR021@6_0_238 ;a10=X238_DE_time'; % 9
load 0HP/normal_0_97                 ;a1=X097_DE_time'; % 10

signals = {a1, a2, a3, a4, a5, a6, a7, a8, a9, a10};
fs = 48000; % 采样频率
L = 864;   % 每段信号长度
cutoff_freq = 5000; % 截止频率
wavename = 'cmor3-3'; % 小波类型
totalscal = 256;     % 总尺度数

for i = 1:10
    %% 随机截取信号并低通滤波
    ori_data = signals{i};
    start_point = randi(length(ori_data) - L);
    signal_raw = ori_data(start_point:(start_point + L - 1));
    [b, a] = butter(4, cutoff_freq / (fs / 2), 'low');
    signal_filtered = filtfilt(b, a, signal_raw);

    %% 计算FFT
    f = (0:L/2-1)*(fs/L); % 频率轴
    fft_raw = abs(fft(signal_raw) / L);
    fft_filtered = abs(fft(signal_filtered) / L);
    fft_raw = fft_raw(1:L/2);
    fft_filtered = fft_filtered(1:L/2);

    %% 小波变换
    Fc = centfrq(wavename);
    c = 2 * Fc * totalscal;
    scals = c ./ (1:totalscal);
    freq = scal2frq(scals, wavename, 1 / fs);

    coefs_raw = cwt(signal_raw, scals, wavename);
    coefs_filtered = cwt(signal_filtered, scals, wavename);

    %% 可视化FFT和CWT
    figure('Units', 'normalized', 'Position', [0.1, 0.1, 0.8, 0.8]);

    % FFT 可视化
    subplot(1, 2, 1);
    plot(f, fft_raw, 'b', 'LineWidth', 1.2); hold on;
    plot(f, fft_filtered, 'r', 'LineWidth', 1.2);
    title(['Class ', num2str(i), ' FFT']);
    xlabel('Frequency (Hz)');
    ylabel('Amplitude');
    legend({'Raw', 'Filtered'});
    grid on;

    % CWT 可视化
    subplot(1, 2, 2);
    imagesc((0:L-1)/fs, freq, abs(coefs_filtered) / max(max(abs(coefs_filtered))));
    axis xy;
    title(['Class ', num2str(i), ' CWT']);
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    colorbar;

    % 保存图像
    saveas(gcf, ['Signal_Class_', num2str(i), '_FFT_CWT.png']);
    close;
end
