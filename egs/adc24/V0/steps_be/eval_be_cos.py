import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine
import json
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def load_xvectors(xvector_scp):
    logging.info(f"Loading x-vectors from {xvector_scp}")
    xvectors = {}
    with open(xvector_scp, 'r') as f:
        for line in f:
            utt_id, path = line.strip().split()
            xvectors[utt_id] = np.load(path)
    logging.info(f"Loaded {len(xvectors)} x-vectors")
    return xvectors

def load_preproc(preproc_file):
    logging.info(f"Loading preprocessing data from {preproc_file}")
    with h5py.File(preproc_file, 'r') as f:
        pca_mean = f['pca_mean'][:]
        pca_components = f['pca_components'][:]
    logging.info("Loaded PCA mean and components")
    return pca_mean, pca_components

def apply_preproc(xvectors, pca_mean, pca_components):
    logging.info("Applying PCA preprocessing to x-vectors")
    xvectors_centered = {utt_id: xvec - pca_mean for utt_id, xvec in xvectors.items()}
    xvectors_pca = {utt_id: np.dot(pca_components, xvec) for utt_id, xvec in xvectors_centered.items()}
    logging.info("Applied PCA preprocessing")
    return xvectors_pca

def calculate_cosine_similarity(xvectors_train, xvectors_test):
    logging.info("Calculating cosine similarity")
    scores = {}
    for test_id, test_xvec in xvectors_test.items():
        for train_id, train_xvec in xvectors_train.items():
            score = 1 - cosine(test_xvec, train_xvec)
            scores[(test_id, train_id)] = score
    logging.info("Calculated cosine similarity for all pairs")
    return scores

def save_scores(scores, output_file):
    logging.info(f"Saving scores to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(scores, f)
    logging.info("Scores saved")

def main(preproc_file, xvector_train_scp, xvector_test_scp, output_file):
    setup_logging()
    logging.info("Starting evaluation")
    xvectors_train = load_xvectors(xvector_train_scp)
    xvectors_test = load_xvectors(xvector_test_scp)
    pca_mean, pca_components = load_preproc(preproc_file)
    xvectors_train_pca = apply_preproc(xvectors_train, pca_mean, pca_components)
    xvectors_test_pca = apply_preproc(xvectors_test, pca_mean, pca_components)
    scores = calculate_cosine_similarity(xvectors_train_pca, xvectors_test_pca)
    save_scores(scores, output_file)
    logging.info("Evaluation completed")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate model using cosine similarity.")
    parser.add_argument('--preproc-file', type=str, required=True, help="Path to the preprocessing file (preproc.h5).")
    parser.add_argument('xvector_train_scp', type=str, help="Path to the xvector.scp file for training data.")
    parser.add_argument('xvector_test_scp', type=str, help="Path to the xvector.scp file for test data.")
    parser.add_argument('output_file', type=str, help="Path to the output file for saving scores.")
    args = parser.parse_args()

    main(args.preproc_file, args.xvector_train_scp, args.xvector_test_scp, args.output_file)
