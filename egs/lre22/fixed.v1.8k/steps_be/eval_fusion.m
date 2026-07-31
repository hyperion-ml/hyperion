function eval_fusion(in_files, out_file, model_file)

    load(model_file, 'alpha', 'beta', 'labels');
    n_files = length(in_files);
    scores={};
    for i=1:n_files
        T_i = readtable(in_files{i}, 'FileType', 'delimitedtext', 'Delimiter','tab', 'ReadRowNames', true, 'VariableNamingRule', 'preserve');
        T_i = sortrows(T_i, 'RowNames');
        s_i = T_i.Variables';
        scores{i}=s_i;
    end
    scores = apply_nary_lin_fusion(scores, alpha, beta);
    T_i.Variables = scores';
    %T_i.Properties.VariableNames = T_i.Properties.VariableDescriptions;
    writetable(T_i, out_file, 'FileType', 'text', 'Delimiter','tab', 'WriteRowNames', true)

    