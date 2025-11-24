#!/usr/bin/env python3
"""
ASVspoof-like Feature Extractor (CatBoost-ready)
Automatically scans 'original' and 'spoofed' subfolders.
Features:
- 39 MFCCs (static + delta + delta²)
- Group Delay (mean/std)
- Energy (RMS)
- Pitch + Voicing
- Spectral Contrast & Rolloff
- Multiprocessing + checkpointing
"""

import os
import json
import traceback
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import librosa

# --------------------- FEATURE EXTRACTION ---------------------
def extract_features(file_path, sr=16000, pitch_method="pyin"):
    try:
        y, sr = librosa.load(file_path, sr=sr)

        # MFCC + Δ + ΔΔ
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        mfcc_feat = np.vstack([mfcc, delta, delta2])

        # Group Delay
        D = librosa.stft(y)
        phase = np.angle(D)
        gd = np.gradient(phase, axis=1)
        gd_mean = np.mean(gd, axis=1)
        gd_std = np.std(gd, axis=1)

        # Energy
        rms = librosa.feature.rms(y=y)

        # Pitch / Voicing
        if pitch_method == "pyin":
            f0, voiced_flag, voiced_prob = librosa.pyin(y, fmin=50, fmax=500)
        else:
            f0 = librosa.yin(y, fmin=50, fmax=500)
            voiced_flag = f0 > 0
            voiced_prob = voiced_flag.astype(float)
        f0 = np.nan_to_num(f0)

        # Spectral Contrast & Rolloff
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

        # Stats pooling
        def pool(x):
            return np.mean(x, axis=1), np.std(x, axis=1)

        mfcc_stats = np.concatenate(pool(mfcc_feat))
        contrast_stats = np.concatenate(pool(contrast))
        rolloff_stats = np.concatenate(pool(rolloff))
        rms_stats = pool(rms)[0]

        f0_mean = np.mean(f0)
        f0_std = np.std(f0)
        voiced_mean = np.mean(voiced_prob)

        # Final vector
        features = np.concatenate([
            mfcc_stats, gd_mean, gd_std,
            contrast_stats, rolloff_stats,
            rms_stats, [f0_mean, f0_std, voiced_mean]
        ])

        return features, None
    except Exception as e:
        return None, f"Error processing {file_path}: {str(e)}\n{traceback.format_exc()}"

# --------------------- MAIN ---------------------
def main():
    # Update this path to your dataset folder
    data_dir = "C:/Users/tusha/OneDrive/Desktop/final attempt/slicedDataset"
    checkpoint_dir = "C:/Users/tusha/OneDrive/Desktop/checkpoints"
    output_path = "C:/Users/tusha/OneDrive/Desktop/features_final.npz"

    os.makedirs(checkpoint_dir, exist_ok=True)

    save_every = 100
    pitch_method = "pyin"
    workers = max(1, os.cpu_count() - 1)

    # Automatically scan original/spoofed folders
    file_names, labels = [], []
    for label, folder in enumerate(["original", "spoofed"]):
        folder_path = os.path.join(data_dir, folder)
        for fname in os.listdir(folder_path):
            if fname.endswith(".wav"):
                file_names.append(os.path.join(folder, fname))  # relative path
                labels.append(1 if folder == "original" else 0)

    features, utt_ids, errors = [], [], []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(extract_features, os.path.join(data_dir, fname), 16000, pitch_method): (fname, label)
            for fname, label in zip(file_names, labels)
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            fname, label = futures[future]
            feat, err = future.result()
            if feat is not None:
                features.append(feat)
                utt_ids.append(fname)
            if err:
                errors.append(err)

            # Checkpoint
            if len(features) % save_every == 0:
                np.savez(os.path.join(checkpoint_dir, 'checkpoint.npz'),
                         features=np.array(features), labels=np.array(labels[:len(features)]))
                with open(os.path.join(checkpoint_dir, 'meta.json'), 'w') as f:
                    json.dump({'processed': utt_ids, 'errors': errors}, f, indent=2)

    # Save final
    np.savez(output_path,
             features=np.array(features),
             labels=np.array(labels[:len(features)]),
             utt_ids=utt_ids)

    with open(os.path.join(checkpoint_dir, 'meta.json'), 'w') as f:
        json.dump({'processed': utt_ids, 'errors': errors}, f, indent=2)

if __name__ == '__main__':
    main()
