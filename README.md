# EEG Motor Imagery Decoding with FBCSP

This repository is an extension of my previous EEG motor imagery classification project, which you can check out [here](https://github.com/SanazRezvani/eeg-motor-imagery-csp)     

In the original project, I implemented a baseline motor imagery decoding pipeline using Common Spatial Patterns (CSP). In this extension, I improve the feature extraction stage by adding a Filter Bank Common Spatial Pattern (FBCSP) approach across the mu and beta rhythms.

This is a complete MATLAB pipeline for **EEG-based motor imagery classification** using signal processing, Filter Bank Common Spatial Patterns (FBCSP), and machine learning.

This work is based on **BCI Competition III – Dataset IVa**. Read the [Dataset description](https://www.bbci.de/competition/iii/desc_IVa.html)

This project implements a full Brain-Computer Interface (BCI) pipeline to classify **motor imagery EEG signals** (Right Hand vs Foot).

The objective is to transform raw EEG signals into discriminative features and evaluate multiple classifiers.

## How to Run

1- Download the dataset from [BCI Competition III – Dataset IVa](https://www.bbci.de/competition/iii/)

2- Open MATLAB

3- Start by loading one of the subjects. ` data_set_IVa_al.mat ` is chosen here.

4- Run: ` run_pipeline.m `

## Project Motivation

Motor imagery EEG signals contain important discriminative information mainly in the mu (8–13 Hz) and beta (13–30 Hz) frequency ranges. However, the most informative frequency sub-bands can vary between subjects.

To address this, this project decomposes the EEG signal into multiple overlapping frequency sub-bands and applies CSP separately to each sub-band. This allows the model to extract more subject-specific spatial features compared with a single broad-band CSP approach.

## Key Features

- Extension of a baseline CSP-based motor imagery decoding pipeline
- Filter bank decomposition across mu and beta rhythms
- Overlapping sub-bands within 8–30 Hz
- CSP feature extraction from each sub-band
- Subject-specific EEG feature representation
- Classification of motor imagery tasks using machine learning

## Pipeline Overview

1. Load EEG motor imagery data
2. Preprocess EEG signals
3. Apply filter bank decomposition across 8–30 Hz
4. Extract CSP features from each sub-band
5. Concatenate sub-band CSP features
6. Train and evaluate a classifier

## Why FBCSP?

Standard CSP extracts spatial filters from a selected frequency band. FBCSP extends this by applying CSP across multiple frequency sub-bands, allowing the model to capture discriminative patterns that may appear in different parts of the mu and beta rhythms.

This is especially useful in motor imagery BCI because EEG patterns are highly subject-specific.

## Relation to Real-Time BCI

This project is still an offline motor imagery decoding pipeline. However, it provides a stronger feature extraction foundation for a future real-time EEG decoding simulation using sliding windows and latency-aware inference.

## Next Extension

The next planned extension is:

➡️ Real-time EEG decoding simulation using sliding windows and streaming predictions.
