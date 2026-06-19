% [S, F, T, P] = spectrogram_octave_fallback(x, window_len, noverlap, f, fs)
%
% Octave compatibility: implements spectrogram when the signal package
% does not provide it. Same interface as MATLAB spectrogram for the usage
% in get_PSD_feature_Springer_HMM.m:
%   window_len, noverlap in samples; f = frequency vector (Hz); fs = sampling rate.
% Returns S (complex STFT), F (= f), T (time vector), P (power = |S|.^2).

function [S, F, T, P] = spectrogram_octave_fallback(x, window_len, noverlap, f, fs)

x = x(:);
window_len = round(window_len);
noverlap = round(noverlap);
hop = window_len - noverlap;
if hop <= 0
    error('spectrogram_octave_fallback: noverlap must be < window length');
end

F = f(:)';
nfft = max(2^nextpow2(window_len), 2 * length(F));

% Time frames
num_frames = max(0, floor((length(x) - window_len) / hop) + 1);
T = (0 : num_frames - 1) * hop / fs + (window_len / 2) / fs;

w = hamming(window_len);
S = zeros(length(F), num_frames);

% FFT bin frequencies: (0:nfft/2)*fs/nfft; broadcast vs F for 501x500 matrix
freqs_bins = (0 : nfft/2)' * (fs / nfft);
[~, bin_idx] = min(abs(freqs_bins - F(:)'), [], 1);
bin_idx = bin_idx(:);

for j = 1 : num_frames
    start_idx = (j - 1) * hop + 1;
    segment = x(start_idx : start_idx + window_len - 1) .* w;
    X = fft(segment, nfft);
    X = X(1 : nfft/2 + 1);
    S(:, j) = X(bin_idx(:));
end

P = abs(S) .^ 2;

end
