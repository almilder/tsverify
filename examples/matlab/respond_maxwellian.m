function respond_maxwellian(bundle)
%RESPOND_MAXWELLIAN Answer a tsverify bundle with a small Maxwellian Thomson code written in Matlab.
%   RESPOND_MAXWELLIAN(BUNDLE) reads BUNDLE/runs.csv. For every zprime run, and every spectral_density run
%   whose electrons and ions are all Maxwellian, it reads the run's inputs and writes output.csv into the
%   run's folder. It skips every other run, which tsverify then reports as not supported.
%
%   Export the bundle with tsverify's canonical conventions, and score it afterwards:
%       tsverify export bundle
%       matlab -batch "respond_maxwellian('bundle')"
%       tsverify score bundle --name matlab-example --report report.md
runs = readtable(fullfile(bundle, 'runs.csv'), 'TextType', 'string', 'Delimiter', ',');
written = 0;
for r = 1:height(runs)
    folder = fullfile(bundle, runs.folder(r));
    switch runs.kind(r)
        case "zprime"
            zeta = read_column(folder, 'zeta.csv', 'zeta');
            zp = zprime(zeta);
            write_columns(fullfile(folder, 'output.csv'), {'re', 'im'}, [real(zp), imag(zp)]);
        case "spectral_density"
            plasma = readtable(fullfile(folder, 'plasma.csv'), 'TextType', 'string', 'Delimiter', ',');
            ions = readtable(fullfile(folder, 'ions.csv'), 'TextType', 'string', 'Delimiter', ',');
            if plasma.electron_edf ~= "maxwellian" || any(ions.edf ~= "maxwellian")
                continue
            end
            lam = read_column(folder, 'wavelengths.csv', 'wavelength_nm');
            S = maxwellian_spectral_density(plasma, ions, lam);
            write_columns(fullfile(folder, 'output.csv'), {'wavelength_nm', 'S'}, [lam, S]);
        otherwise
            continue
    end
    written = written + 1;
end
fprintf('matlab example: wrote %d outputs\n', written);
end

function x = read_column(folder, file, name)
t = readtable(fullfile(folder, file), 'Delimiter', ',');
x = t.(name);
end

function write_columns(path, names, values)
fid = fopen(path, 'w');
fprintf(fid, '%s\n', strjoin(names, ','));
fprintf(fid, [strjoin(repmat({'%.17g'}, 1, numel(names)), ','), '\n'], values.');
fclose(fid);
end
