function train_fusion(train_list, in_files, model_file)

    train_list = readtable(train_list, 'FileType', 'delimitedtext', 'Delimiter', ' ', 'ReadVariableNames', false, 'ReadRowNames', true);
    train_list = sortrows(train_list, 'RowNames');
    [labels, ia, ic]=unique(train_list);
    n_files = length(in_files);
    scores={};
    for i=1:n_files
        T_i = readtable(in_files{i}, 'FileType', 'delimitedtext', 'Delimiter','tab', 'ReadRowNames', true, 'VariableNamingRule', 'preserve');
        T_i = sortrows(T_i, 'RowNames');
        s_i = T_i.Variables';
        scores{i}=s_i;
    end
    [alpha, beta] = train_nary_llr_fusion(scores, ic, 0, 1e-6, [], ones(1,1))
    save(model_file, 'alpha', 'beta', 'labels');
    