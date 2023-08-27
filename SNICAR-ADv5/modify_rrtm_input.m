function modify_rrtm_input(file_path, binned_albedo,line)
    % Convert the binned albedo array to a string with 4 digits precision
    albedo_str = sprintf(' %.4f', binned_albedo);
    add = '           2  0';
    space = ' ';
    albedo_str = strcat(add,space,albedo_str);
    % Read the contents of the file
    file_content = fileread(file_path);
    
    % Split the file content into lines
    lines = splitlines(file_content);
    
    % Replace the fifth line with the binned albedo values
    lines{line} = albedo_str;
    
    % Join the lines back into a single string
    updated_content = strjoin(lines, '\n');
    
    % Write the updated content back to the file
    fid = fopen(file_path, 'w');
    fprintf(fid, '%s', updated_content);
    fclose(fid);
    
    %disp('Fifth line replaced with binned albedo values.');
end

