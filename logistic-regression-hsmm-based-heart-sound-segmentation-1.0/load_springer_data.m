% function data = load_springer_data(mat_path)
%
% Load example_data.mat with structure compatible with both MATLAB and Octave.
% The .mat file contains:
%   - example_audio_data: cell array of PCG signals (1000 Hz)
%   - example_annotations: Nx2 cell, col1 = R-peak positions, col2 = end-T-wave (samples)
%   - binary_diagnosis (optional): 0 = normal, 1 = pathology
%   - patient_number (optional): patient ID per recording
%
% Returns struct data with fields: audio_data, annotations, diagnosis (if present),
% patient_number (if present), Fs (default 1000).

function data = load_springer_data(mat_path)

if nargin < 1 || isempty(mat_path)
    mat_path = 'example_data.mat';
end

loaded = load(mat_path);

% Handle variable name: file may contain 'example_data' as struct or as name of struct
if isfield(loaded, 'example_data')
    ed = loaded.example_data;
else
    error('load_springer_data: no example_data in %s', mat_path);
end

% Octave/MATLAB: sometimes example_data is 1x1 struct array; unpack to single struct
if numel(ed) > 1
    ed = ed(1);
end
if isstruct(ed) && numel(ed) == 1
    % get fields from the struct
    fn = fieldnames(ed);
    for i = 1:length(fn)
        data.(fn{i}) = ed.(fn{i});
    end
else
    data = ed;
end

% Normalize field names for downstream use (expect audio_data and annotations)
if isfield(data, 'example_audio_data')
    data.audio_data = data.example_audio_data;
elseif ~isfield(data, 'audio_data')
    error('load_springer_data: no example_audio_data or audio_data in %s', mat_path);
end

if isfield(data, 'example_annotations')
    data.annotations = data.example_annotations;
elseif ~isfield(data, 'annotations')
    error('load_springer_data: no example_annotations or annotations in %s', mat_path);
end

% Optional fields (from paper: binary_diagnosis, patient_number)
if ~isfield(data, 'binary_diagnosis') && isfield(data, 'diagnosis')
    data.binary_diagnosis = data.diagnosis;
end
if ~isfield(data, 'Fs') && ~isfield(data, 'fs')
    data.Fs = 1000;
elseif isfield(data, 'fs')
    data.Fs = data.fs;
end

end
