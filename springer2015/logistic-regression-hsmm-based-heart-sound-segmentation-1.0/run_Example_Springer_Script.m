%% Example Springer script
% A script to demonstrate the use of the Springer segmentation algorithm

%% Copyright (C) 2016  David Springer
% dave.springer@gmail.com
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

%%
close all;
clear all;

%% GNU Octave: load required packages (Statistics, Signal for mnrval/mvnpdf and wavelets)
if exist('OCTAVE_VERSION', 'builtin')
    pkg load statistics;
    pkg load signal;
end

%% Load the default options:
% These options control options such as the original sampling frequency of
% the data, the sampling frequency for the derived features and whether the
% mex code should be used for the Viterbi decoding (disabled automatically under Octave):
springer_options = default_Springer_HSMM_options();

%% Load the audio data and the annotations:
% example_data.mat contains: audio (1000 Hz), R-peak and end-T-wave annotations,
% and optionally binary_diagnosis (1 = pathology) and patient_number per recording.
data = load_springer_data('example_data.mat');

%% Split the data into train and test sets:
% Select 5 recordings for training and one for testing (indices into full dataset):
training_indices = [1, 47, 361, 402, 572];
train_recordings = data.audio_data(training_indices);
train_annotations = data.annotations(training_indices, :);

test_index = 664;
test_recordings = data.audio_data(test_index);
test_annotations = data.annotations(test_index, :);


%% Train the HMM:
[B_matrix, pi_vector, total_obs_distribution] = trainSpringerSegmentationAlgorithm(train_recordings,train_annotations,springer_options.audio_Fs, false);

%% Run the HMM on an unseen test recording:
% And display the resulting segmentation
numPCGs = length(test_recordings);

for PCGi = 1:numPCGs
    [assigned_states] = runSpringerSegmentationAlgorithm(test_recordings{PCGi}, springer_options.audio_Fs, B_matrix, pi_vector, total_obs_distribution, true);
end

