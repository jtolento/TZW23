clc;clear all; close all;clc;
% File paths
inputFilePath = '/Users/jtolento/RRTMG_SW/run_examples_std_atm/ppr1/sps/snw_wvc/input_mls_brd';
outputFilePath = '/Users/jtolento/RRTMG_SW/run_examples_std_atm/ppr1/sps/snw_wvc/input_mls_brd10';
scalar = 1.0;
% Read the input file
fidIn = fopen(inputFilePath, 'r');
fidOut = fopen(outputFilePath, 'w');

if fidIn == -1
    error('Could not open the input file.');
end

if fidOut == -1
    error('Could not create the output file.');
end

% Read and save the first 10 lines
headerLines = cell(10, 1);
for i = 1:10
    headerLines{i} = fgetl(fidIn);
end

modifiedLines = {};
a = 0;
% Read, modify, and write the remaining lines
while ~feof(fidIn)
    a = a+1;
    disp(a)
    %disp(fidIn)
    %disp(fgetl(fidIn))
    line = fgetl(fidIn);
    
    
    %disp()
    if ischar(line)
        % Check if the line contains ' 3 '
        if contains(line,' 3 ') 
            % Write the original line (no modification needed)
            modifiedLines{end+1} = line;
        else
        % Split the line by spaces
        parts = strsplit(line);        
        %disp(parts)
        % Convert the first column value to a number
        x = str2double(parts{2});
        %disp(x)
        if ~isnan(x)
            %disp('true')
            % Multiply x by the scalar b
            y = x * scalar;
            
            % Format x and y in exponential notation
            xFormatted = sprintf('%0.7E', x);
            yFormatted = sprintf('%0.7E', y);
            %disp(xFormatted)
            %disp(yFormatted)
            
            % Replace x with y in the line
            modifiedLine = strrep(line, xFormatted, yFormatted);
            modifiedLines{end+1} = modifiedLine;
            %disp(modifiedLine)
            %disp(line)
            % Write the modified line to the output file
            
        else
            % Write the original line (no modification needed)
            modifiedLines{end+1} = line;
        end
       
    end
    end
end

% Close the files
fclose(fidIn);

% Write the header lines to the output file
for i = 1:10
    fprintf(fidOut, '%s\n', headerLines{i});
end

% Write the modified lines to the output file
for i = 1:numel(modifiedLines)
    fprintf(fidOut, '%s\n', modifiedLines{i});
end

str_end1 = '%%%%%';
str_end2 = '123456789-123456789-123456789-123456789-123456789-123456789-123456789-123456789-';

fprintf(fidOut,'%s\n%s\n',str_end1,str_end2);

fclose(fidOut);



disp('Modification completed.');
