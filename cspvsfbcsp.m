clc;
clear;
close all;

%% Classifier names
classifiers = categorical({'SVM', 'KNN', 'LDA'});
classifiers = reordercats(classifiers, {'SVM', 'KNN', 'LDA'});

%% CSP results
csp_accuracy = [70.59, 77.94, 73.53];

%% FBCSP results
fbcsp_accuracy = [86.76, 91.18, 83.82];

%% Combine data for grouped bar chart
accuracy_matrix = [csp_accuracy; fbcsp_accuracy]';

%% Plot
figure;
bar(classifiers, accuracy_matrix);

ylabel('Accuracy (%)');
title('CSP vs FBCSP Classification Performance');

legend({'CSP', 'FBCSP'}, 'Location', 'northwest');

grid on;

%% Optional: annotate values on bars
xtips = [];
ytips = [];
labels = [];

for i = 1:size(accuracy_matrix,1)
    for j = 1:size(accuracy_matrix,2)
        xtips = [xtips; i + (j-1)*0.15 - 0.075];
        ytips = [ytips; accuracy_matrix(i,j)];
        labels = [labels; string(sprintf('%.2f', accuracy_matrix(i,j)))];
    end
end

text(xtips, ytips, labels, ...
    'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'bottom', ...
    'FontSize', 10);

%% Save figure
saveas(gcf, 'csp_vs_fbcsp_comparison.png');