% function y = wkeep1(x, len, opt)
%
% Octave compatibility: MATLAB Wavelet Toolbox provides wkeep1; Octave signal
% has wkeep but not wkeep1. This keeps L elements from vector X (central part
% by default, matching MATLAB Wavelet Toolbox behavior).
%
% Usage: y = wkeep1(x, len)       % central len elements
%        y = wkeep1(x, len, 'c')  % central (default)
%        y = wkeep1(x, len, 'f')  % first len elements
%        y = wkeep1(x, len, 'l')  % last len elements

function y = wkeep1(x, len, opt)

if nargin < 3
    opt = 'c';
end

x = x(:)';
n = length(x);

if len >= n
    % Pad to length len (center the original)
    pad = len - n;
    left = floor(pad / 2);
    right = pad - left;
    y = [zeros(1, left), x, zeros(1, right)];
    return;
end

switch lower(opt(1))
    case 'f'
        y = x(1:len);
    case 'l'
        y = x(n - len + 1 : n);
    otherwise
        % 'c' or central
        start = floor((n - len) / 2) + 1;
        y = x(start : start + len - 1);
end

end
