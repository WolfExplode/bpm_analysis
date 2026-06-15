% function [audio_cell, annotations_cell, Fs] = load_pcg_from_wav(wav_paths_or_dir, annotations_path, target_fs)
%
% Load PCG recordings from .wav files for use with Springer segmentation.
% Compatible with MATLAB and GNU Octave (uses audioread).
%
% INPUTS:
%   wav_paths_or_dir - Either:
%                      - Cell array of .wav file paths, or
%                      - String: path to a directory (all .wav files inside are loaded)
%   annotations_path - (optional) Path to annotations file. If omitted, returns
%                      empty annotations (suitable for runSpringerSegmentationAlgorithm
%                      only; training requires annotations).
%                      Supported format: .mat with variables 'audio_filenames' (or
%                      'filenames'), 's1_positions', 's2_positions' (each cell or
%                      matrix matching order of files), or .csv with columns:
%                      filename, s1_samples, s2_samples (s1/s2 as space-separated lists).
%   target_fs        - (optional) Target sampling rate in Hz. Default 1000 (Springer default).
%
% OUTPUTS:
%   audio_cell       - Cell array of PCG signals (each 1xN or Nx1), resampled to target_fs.
%   annotations_cell - Nx2 cell: annotations_cell{i,1} = R-peak positions (samples at target_fs),
%                      annotations_cell{i,2} = end-T-wave positions (samples at target_fs).
%                      Empty if no annotations_path or file not found.
%   Fs               - target_fs (1000 by default).

function [audio_cell, annotations_cell, Fs] = load_pcg_from_wav(wav_paths_or_dir, annotations_path, target_fs)

if nargin < 3 || isempty(target_fs)
    target_fs = 1000;
end
Fs = target_fs;

%% Resolve list of .wav files
if ischar(wav_paths_or_dir) || isstring(wav_paths_or_dir)
    wav_dir = char(wav_paths_or_dir);
    if exist(wav_dir, 'dir') ~= 7
        error('load_pcg_from_wav: directory not found: %s', wav_dir);
    end
    list = dir(fullfile(wav_dir, '*.wav'));
    wav_paths = cell(1, length(list));
    for k = 1:length(list)
        wav_paths{k} = fullfile(wav_dir, list(k).name);
    end
elseif iscell(wav_paths_or_dir)
    wav_paths = wav_paths_or_dir;
else
    error('load_pcg_from_wav: first argument must be directory path or cell of .wav paths');
end

if isempty(wav_paths)
    audio_cell = {};
    annotations_cell = cell(0, 2);
    return;
end

%% Load each .wav and resample to target_fs
audio_cell = cell(1, length(wav_paths));
for k = 1:length(wav_paths)
    path_k = wav_paths{k};
    if exist(path_k, 'file') ~= 2
        warning('load_pcg_from_wav: file not found: %s', path_k);
        audio_cell{k} = [];
        continue;
    end
    try
        [y, fs_file] = audioread(path_k);
    catch
        warning('load_pcg_from_wav: audioread failed for: %s', path_k);
        audio_cell{k} = [];
        continue;
    end
    y = y(:);
    if size(y, 2) > 1
        y = mean(y, 2);
    end
    if fs_file ~= target_fs
        y = resample(y, target_fs, fs_file);
    end
    audio_cell{k} = y;
end

%% Annotations (optional)
annotations_cell = cell(length(wav_paths), 2);
for i = 1:length(wav_paths)
    annotations_cell{i, 1} = [];
    annotations_cell{i, 2} = [];
end

if nargin < 2 || isempty(annotations_path)
    return;
end
if exist(annotations_path, 'file') ~= 2
    warning('load_pcg_from_wav: annotations file not found: %s', annotations_path);
    return;
end

[~, ~, ext] = fileparts(annotations_path);
ext = lower(ext);

if strcmp(ext, '.mat')
    ann = load(annotations_path);
    if isfield(ann, 's1_positions') && isfield(ann, 's2_positions')
        s1 = ann.s1_positions;
        s2 = ann.s2_positions;
        if iscell(s1)
            for i = 1:min(length(audio_cell), length(s1))
                annotations_cell{i, 1} = s1{i}(:);
                if i <= length(s2)
                    annotations_cell{i, 2} = s2{i}(:);
                end
            end
        else
            for i = 1:min(length(audio_cell), size(s1, 1))
                annotations_cell{i, 1} = s1(i, :)(:);
                annotations_cell{i, 2} = s2(i, :)(:);
            end
        end
    end
elseif strcmp(ext, '.csv')
    % CSV: one row per file (same order as wav_paths). Columns: [filename,] s1_samples, s2_samples
    % s1/s2 can be space- or comma-separated sample indices at target_fs.
    fid = fopen(annotations_path, 'r');
    if fid < 0
        return;
    end
    row = 0;
    while ~feof(fid) && row < length(wav_paths)
        line = fgetl(fid);
        if ~ischar(line) || isempty(strtrim(line)), continue; end
        row = row + 1;
        parts = strsplit(line, ',');
        if length(parts) < 2
            continue;
        end
        % If 3+ columns: filename, s1, s2. If 2 columns: s1, s2 (row order = file order).
        if length(parts) >= 3
            s1_str = strtrim(parts{2});
            s2_str = strtrim(parts{3});
        else
            s1_str = strtrim(parts{1});
            s2_str = strtrim(parts{2});
        end
        s1_vec = str2num(s1_str);
        s2_vec = str2num(s2_str);
        if row <= length(wav_paths)
            annotations_cell{row, 1} = s1_vec(:);
            annotations_cell{row, 2} = s2_vec(:);
        end
    end
    fclose(fid);
end

end
