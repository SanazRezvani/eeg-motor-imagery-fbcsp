clc;
clear;
close all;

%% EEG Motor Imagery FBCSP Extension Pipeline
% Extension of the baseline CSP pipeline.
% This version applies a filter bank across mu and beta rhythms,
% computes CSP filters for each sub-band, extracts CSP variance features,
% concatenates all sub-band features, and trains classifiers.

%% Configuration
config.dataset_path = 'data_set_IVa_al.mat';
config.spatial_filter = 'CAR';  % options: 'CAR', 'Low Laplacian', 'High Laplacian'
config.filter_order = 3;
config.train_ratio = 0.70;
config.num_csp_pairs = 1;
config.trial_length_s = 3.5;
config.plot_figures = true;
config.visualise_csp = false;

% FBCSP filter bank: 4 Hz windows with 2 Hz overlap across 8–30 Hz
config.filter_bank = [
     8 12;
    10 14;
    12 16;
    14 18;
    16 20;
    18 22;
    20 24;
    22 26;
    24 28;
    26 30
];

%% Load dataset
load(config.dataset_path);

raw_eeg_signal = 0.1 * double(cnt);   % convert to microvolts
sampling_rate = nfo.fs;
channel_labels = nfo.clab;
channel_x_positions = nfo.xpos;
channel_y_positions = nfo.ypos;
electrode_positions = [channel_x_positions'; channel_y_positions'];

trial_start_samples = mrk.pos;
trial_labels = mrk.y;
trial_length_samples = round(config.trial_length_s * sampling_rate);

results_folder = 'results';

if ~exist(results_folder, 'dir')
    mkdir(results_folder);
end

%% Plot channel layout
if config.plot_figures
    figure;
    plot(channel_x_positions, channel_y_positions, 'ro', ...
        'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'k');
    grid on;
    text(channel_x_positions + 0.02, channel_y_positions, channel_labels, ...
        'FontSize', 15, 'Color', 'b');
    title('EEG Channel Locations');
    saveas(gcf, fullfile(results_folder, 'channel_layout.png'));
end

%% Class labels
% Dataset IVa competition labels:
%   1 = right hand
%   2 = foot
class_1_label = 1;
class_2_label = 2;
class_1_name = 'Right';
class_2_name = 'Foot';

%% FBCSP feature extraction
train_features_class_1_all_bands = [];
train_features_class_2_all_bands = [];
test_features_class_1_all_bands = [];
test_features_class_2_all_bands = [];

fprintf('\nRunning FBCSP pipeline...\n');
fprintf('Number of sub-bands: %d\n\n', size(config.filter_bank, 1));

for band_idx = 1:size(config.filter_bank, 1)

    low_cutoff_hz = config.filter_bank(band_idx, 1);
    high_cutoff_hz = config.filter_bank(band_idx, 2);

    fprintf('Processing sub-band %d: %d-%d Hz\n', ...
        band_idx, low_cutoff_hz, high_cutoff_hz);

    %% Band-pass filtering for current sub-band
    normalized_cutoff = [low_cutoff_hz high_cutoff_hz] / (sampling_rate / 2);
    [b, a] = butter(config.filter_order, normalized_cutoff, 'bandpass');
    bandpassed_signal = filtfilt(b, a, raw_eeg_signal);

    %% Optional plot for first sub-band only
    if config.plot_figures && band_idx == 1
        figure;
        subplot(2,1,1);
        plot(raw_eeg_signal(:, 1), 'r', 'LineWidth', 0.5);
        title('Raw EEG Signal - First Channel');

        subplot(2,1,2);
        plot(bandpassed_signal(:, 1), 'b', 'LineWidth', 0.5);
        title(sprintf('Band-Pass Filtered EEG Signal - First Channel (%d-%d Hz)', ...
            low_cutoff_hz, high_cutoff_hz));

        saveas(gcf, fullfile(results_folder, 'time_domain_signals_first_subband.png'));
    end

    %% Apply spatial filtering
    spatially_filtered_signal = apply_spatial_filter( ...
        bandpassed_signal, ...
        config.spatial_filter, ...
        electrode_positions);

    %% Extract trials by class for current sub-band
    class_1_trials = [];
    class_2_trials = [];
    class_1_count = 0;
    class_2_count = 0;

    for trial_idx = 1:length(trial_labels)

        current_label = trial_labels(trial_idx);

        if isnan(current_label)
            continue;
        end

        start_sample = trial_start_samples(trial_idx);
        end_sample = start_sample + trial_length_samples - 1;

        if end_sample > size(spatially_filtered_signal, 1)
            continue;
        end

        sample_indices = start_sample:end_sample;
        trial_signal = spatially_filtered_signal(sample_indices, :);

        if current_label == class_1_label
            class_1_count = class_1_count + 1;
            class_1_trials(:, :, class_1_count) = trial_signal;

        elseif current_label == class_2_label
            class_2_count = class_2_count + 1;
            class_2_trials(:, :, class_2_count) = trial_signal;
        end
    end

    %% Split each class into training and test sets
    num_class_1_trials = size(class_1_trials, 3);
    num_class_2_trials = size(class_2_trials, 3);

    num_train_class_1 = floor(config.train_ratio * num_class_1_trials);
    num_train_class_2 = floor(config.train_ratio * num_class_2_trials);

    train_class_1 = class_1_trials(:, :, 1:num_train_class_1);
    train_class_2 = class_2_trials(:, :, 1:num_train_class_2);

    test_class_1 = class_1_trials(:, :, num_train_class_1 + 1:end);
    test_class_2 = class_2_trials(:, :, num_train_class_2 + 1:end);

    %% Compute CSP filters for current sub-band
    csp_filters = compute_csp_filters( ...
        train_class_1, ...
        train_class_2, ...
        config.num_csp_pairs);

    %% Optional CSP visualisation for first sub-band only
    if config.plot_figures && config.visualise_csp && band_idx == 1
        visualise_csp_transformation_video( ...
            train_class_1, train_class_2, ...
            test_class_1, test_class_2, ...
            csp_filters);
    end

    %% Extract CSP variance features for current sub-band
    train_features_class_1 = extract_csp_variance_features(train_class_1, csp_filters);
    train_features_class_2 = extract_csp_variance_features(train_class_2, csp_filters);
    test_features_class_1 = extract_csp_variance_features(test_class_1, csp_filters);
    test_features_class_2 = extract_csp_variance_features(test_class_2, csp_filters);

    %% Concatenate features across sub-bands
    train_features_class_1_all_bands = [
        train_features_class_1_all_bands;
        train_features_class_1
    ];

    train_features_class_2_all_bands = [
        train_features_class_2_all_bands;
        train_features_class_2
    ];

    test_features_class_1_all_bands = [
        test_features_class_1_all_bands;
        test_features_class_1
    ];

    test_features_class_2_all_bands = [
        test_features_class_2_all_bands;
        test_features_class_2
    ];
end

%% Build final FBCSP feature matrices and labels
train_features = [
    train_features_class_1_all_bands, ...
    train_features_class_2_all_bands
];

train_labels = [
    ones(1, size(train_features_class_1_all_bands, 2)), ...
    2 * ones(1, size(train_features_class_2_all_bands, 2))
];

test_features = [
    test_features_class_1_all_bands, ...
    test_features_class_2_all_bands
];

test_labels = [
    ones(1, size(test_features_class_1_all_bands, 2)), ...
    2 * ones(1, size(test_features_class_2_all_bands, 2))
];

fprintf('\nFinal FBCSP feature matrix size:\n');
fprintf('Training features: %d features x %d trials\n', ...
    size(train_features, 1), size(train_features, 2));
fprintf('Testing features:  %d features x %d trials\n\n', ...
    size(test_features, 1), size(test_features, 2));

%% Train classifiers and evaluate performance
results = struct();

% Support Vector Machine
results.svm.model = fitcsvm(train_features', train_labels');
results.svm.predictions = predict(results.svm.model, test_features');
results.svm.accuracy = mean(results.svm.predictions' == test_labels) * 100;

figure;
confusionchart(test_labels, results.svm.predictions);
title(sprintf('FBCSP Confusion Matrix - SVM (%s vs %s)', class_1_name, class_2_name));
saveas(gcf, fullfile(results_folder, 'fbcsp_confusion_svm.png'));

fprintf('FBCSP Accuracy (SVM): %.2f%%\n', results.svm.accuracy);

% K-Nearest Neighbors
num_neighbors = 5;
results.knn.model = fitcknn(train_features', train_labels', ...
    'NumNeighbors', num_neighbors);
results.knn.predictions = predict(results.knn.model, test_features');
results.knn.accuracy = mean(results.knn.predictions' == test_labels) * 100;

figure;
confusionchart(test_labels, results.knn.predictions);
title(sprintf('FBCSP Confusion Matrix - KNN (k = %d)', num_neighbors));
saveas(gcf, fullfile(results_folder, 'fbcsp_confusion_knn.png'));

fprintf('FBCSP Accuracy (KNN): %.2f%%\n', results.knn.accuracy);

% Linear Discriminant Analysis
results.lda.model = fitcdiscr(train_features', train_labels', ...
    'DiscrimType', 'pseudoLinear');
results.lda.predictions = predict(results.lda.model, test_features');
results.lda.accuracy = mean(results.lda.predictions' == test_labels) * 100;

figure;
confusionchart(test_labels, results.lda.predictions);
title('FBCSP Confusion Matrix - LDA');
saveas(gcf, fullfile(results_folder, 'fbcsp_confusion_lda.png'));

fprintf('FBCSP Accuracy (LDA): %.2f%%\n', results.lda.accuracy);

%% Compare classifier accuracies
classifier_names = categorical({'SVM', 'KNN', 'LDA'});
classifier_names = reordercats(classifier_names, {'SVM', 'KNN', 'LDA'});

accuracy_values = [
    results.svm.accuracy, ...
    results.knn.accuracy, ...
    results.lda.accuracy
];

figure;
bar(classifier_names, accuracy_values);
ylabel('Accuracy (%)');
title('FBCSP Classifier Performance Comparison');
grid on;

saveas(gcf, fullfile(results_folder, 'fbcsp_accuracy_comparison.png'));

%% Save results
save(fullfile(results_folder, 'fbcsp_results.mat'), ...
    'results', 'config', 'train_features', 'test_features', ...
    'train_labels', 'test_labels');

fprintf('\nFBCSP pipeline complete. Results saved in the results folder.\n');

%% Local helper function
function features = extract_csp_variance_features(trials, csp_filters)
%EXTRACT_CSP_VARIANCE_FEATURES Extract variance-based features after CSP projection.
%
% Inputs:
%   trials       - EEG trials [samples x channels x trials]
%   csp_filters  - CSP projection matrix [channels x num_filters]
%
% Output:
%   features     - Feature matrix [num_filters x num_trials]

    num_trials = size(trials, 3);
    num_filters = size(csp_filters, 2);
    features = zeros(num_filters, num_trials);

    for trial_idx = 1:num_trials
        trial_data = trials(:, :, trial_idx)';   % [channels x samples]
        projected_trial = csp_filters' * trial_data;
        features(:, trial_idx) = var(projected_trial, 0, 2);
    end
end