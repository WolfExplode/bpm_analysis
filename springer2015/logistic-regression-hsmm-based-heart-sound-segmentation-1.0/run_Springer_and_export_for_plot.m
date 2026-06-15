%% Run Springer segmentation and export results for viewing in bpm_analysis plotting interface
% Same as run_Example_Springer_Script but saves assigned_states + WAV to an export
% directory. Then run the Python script to generate the HTML report:
%   python -m bpm_analysis.plot_octave_springer_results path/to/export_dir/base_springer_export.mat
% (or from the repo root: python -m bpm_analysis.plot_octave_springer_results ...)
%
% Set MODEL_DIR to a folder containing Springer_B_matrix.mat, Springer_pi_vector.mat,
% and Springer_total_obs_distribution.mat to use a pre-trained model instead of training.

%%
close all;
clear all;

%% Optional: use pre-trained model (set to empty '' to train from example_data.mat)
MODEL_DIR = fullfile(fileparts(mfilename('fullpath')), 'cristhian_potes-204');
% MODEL_DIR = '';  % uncomment to train from data instead

if exist('OCTAVE_VERSION', 'builtin')
    pkg load statistics;
    pkg load signal;
end

springer_options = default_Springer_HSMM_options();
data = load_springer_data('example_data.mat');

test_index = 665;
test_recordings = data.audio_data(test_index);
test_annotations = data.annotations(test_index, :);

if ~isempty(MODEL_DIR) && exist(MODEL_DIR, 'dir')
    %% Load pre-trained model from MODEL_DIR
    B_path = fullfile(MODEL_DIR, 'Springer_B_matrix.mat');
    pi_path = fullfile(MODEL_DIR, 'Springer_pi_vector.mat');
    tot_path = fullfile(MODEL_DIR, 'Springer_total_obs_distribution.mat');
    if exist(B_path, 'file') && exist(pi_path, 'file') && exist(tot_path, 'file')
        L = load(B_path);
        fn = fieldnames(L);
        B_matrix = L.(fn{1});
        if iscell(B_matrix)
            B_matrix = B_matrix(1:4);
            if size(B_matrix, 1) > 1, B_matrix = B_matrix'; end
            B_matrix = B_matrix(1:min(4, numel(B_matrix)));
        else
            B_matrix = num2cell(B_matrix, 2);
            if size(B_matrix, 1) == 1, B_matrix = B_matrix'; end
            B_matrix = B_matrix(1:4)';
        end
        L = load(pi_path);
        fn = fieldnames(L);
        pi_vector = L.(fn{1});
        pi_vector = pi_vector(:)';
        if numel(pi_vector) > 4, pi_vector = pi_vector(1:4); end
        L = load(tot_path);
        fn = fieldnames(L);
        total_obs_distribution = L.(fn{1});
        if iscell(total_obs_distribution)
            total_obs_distribution = total_obs_distribution(1:2);
        else
            total_obs_distribution = {total_obs_distribution(1,:), total_obs_distribution(2,:)};
        end
        % Under Octave we use 3 features (no wavelet); truncate mean/cov to match
        if exist('OCTAVE_VERSION', 'builtin') && ~springer_options.include_wavelet_feature
            nfeat = 3;
            m = total_obs_distribution{1};
            c = total_obs_distribution{2};
            total_obs_distribution{1} = m(1:min(nfeat, numel(m)));
            total_obs_distribution{2} = c(1:min(nfeat, size(c,1)), 1:min(nfeat, size(c,2)));
        end
        fprintf('Loaded pre-trained model from %s\n', MODEL_DIR);
    else
        error('Model dir set but missing .mat files. Need Springer_B_matrix.mat, Springer_pi_vector.mat, Springer_total_obs_distribution.mat');
    end
else
    %% Train model from example_data.mat
    training_indices = [5, 48, 375, 402, 572];
    train_recordings = data.audio_data(training_indices);
    train_annotations = data.annotations(training_indices, :);
    [B_matrix, pi_vector, total_obs_distribution] = trainSpringerSegmentationAlgorithm(train_recordings, train_annotations, springer_options.audio_Fs, false);
end

%% Run on one test recording and export for plotting
numPCGs = length(test_recordings);
export_dir = fullfile(pwd(), '..', 'processed_files');  % bpm_analysis/processed_files
base_name = 'octave_springer';

% Use first test recording only for export (avoid overwriting)
test_audio = test_recordings{1};
[assigned_states] = runSpringerSegmentationAlgorithm(test_audio, springer_options.audio_Fs, B_matrix, pi_vector, total_obs_distribution, false);
export_springer_for_plotting(assigned_states, test_audio, springer_options.audio_Fs, export_dir, base_name);

fprintf('Done. To generate the HTML plot, from the bpm_analysis repo root run:\n');
fprintf('  python -m bpm_analysis.plot_octave_springer_results "%s"\n', fullfile(export_dir, [base_name '_springer_export.mat']));
