function take_sigmoid_par(feature_type)

    for i = 1 : 5
        for j = 1 : 5
            if j == 1
                C = num2str(0.01 * (10 ^ (j - 1)), '%.2f');
            else
                C = num2str(0.01 * (10 ^ (j - 1)), '%.1f');
            end
            fprintf('M = %d, C = %s\n', i, C);
            load(['/home/Hao/Work/Cmts/calibrate/ranking/ranking_', feature_type,'_Data6_4_20_C', C, 'M', num2str(i), 'I600.mat']);
            for z = 1 : i
                par(z,:) = esvm_learn_sigmoid(Data(:,z), Y(:,z));
            end
            save(['/home/Hao/Work/Cmts/calibrate/ranking/ranking_', feature_type,'_par6_4_20_C', C, 'M', num2str(i), 'I600.mat'], 'par');
            clear par
        end
    end
