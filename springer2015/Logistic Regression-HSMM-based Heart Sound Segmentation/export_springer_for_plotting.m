% export_springer_for_plotting(assigned_states, audio_data, Fs, export_dir, base_name)
%
% Saves Springer segmentation results so they can be viewed in the
% bpm_analysis plotting interface. Writes:
%   - export_dir/base_name.wav         : the PCG audio (so the plotter can load it)
%   - export_dir/base_name_springer_export.mat : assigned_states, Fs, export_audio_wav
%
% The Python script plot_octave_springer_results.py reads the .mat and WAV
% and generates the same HTML report as the GUI (in processed_files).

function export_springer_for_plotting(assigned_states, audio_data, Fs, export_dir, base_name)

if nargin < 5 || isempty(base_name)
    base_name = 'octave_springer';
end
if nargin < 4 || isempty(export_dir)
    export_dir = pwd();
end

assigned_states = assigned_states(:);
audio_data = audio_data(:);

if ~exist(export_dir, 'dir')
    mkdir(export_dir);
end

wav_filename = [base_name '.wav'];
wav_path = fullfile(export_dir, wav_filename);
mat_filename = [base_name '_springer_export.mat'];
mat_path = fullfile(export_dir, mat_filename);

% Normalize audio to [-1, 1] for WAV
a = double(audio_data);
a = a / (max(abs(a)) + 1e-10);
audiowrite(wav_path, a, round(Fs));

% Save .mat for Python: assigned_states, Fs, and the WAV filename (relative to export_dir)
export_audio_wav = wav_filename;
save(mat_path, 'assigned_states', 'Fs', 'export_audio_wav', '-v7');

fprintf('Exported for plotting:\n  %s\n  %s\n', wav_path, mat_path);

end
