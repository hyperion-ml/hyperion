import argparse
import numpy as np
import kaldiio
import h5py
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import normalize
import logging

def load_xvectors(xvector_scp, part_idx, num_parts):
    xvectors = []
    utt_ids = []
    for i, (utt_id, xvector) in enumerate(kaldiio.load_scp_sequential(xvector_scp)):
        if i % num_parts == (part_idx - 1):
            xvectors.append(xvector)
            utt_ids.append(utt_id)
    return np.array(xvectors), utt_ids

def load_preproc_model(preproc_model_path):
    with h5py.File(preproc_model_path, 'r') as f:
        pca_mean = np.array(f['pca_mean'])
        pca_components = np.array(f['pca_components'])
    return pca_mean, pca_components

def transform_xvectors(xvectors, pca_mean, pca_components):
    xvectors = (xvectors - pca_mean).dot(pca_components.T)
    return xvectors

def load_true_labels(utt2lang_path):
    utt2lang = {}
    with open(utt2lang_path, 'r') as f:
        for line in f:
            utt_id, lang = line.strip().split()
            utt2lang[utt_id] = lang
    return utt2lang

def cosine_similarity(a, b):
    a = normalize(a)
    b = normalize(b)
    return np.dot(a, b.T)

def evaluate_metrics(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    cm = confusion_matrix(true_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')
    
    return accuracy, cm, precision, recall, f1

def main(args):
    logging.info("Loading x-vectors from %s", args.xvector_scp)
    xvectors, utt_ids = load_xvectors(args.xvector_scp, args.part_idx, args.num_parts)

    logging.info("Loading preprocessing model from %s", args.preproc_model_path)
    pca_mean, pca_components = load_preproc_model(args.preproc_model_path)

    logging.info("Transforming x-vectors using the preprocessing model")
    xvectors_transformed = transform_xvectors(xvectors, pca_mean, pca_components)

    logging.info("Loading true labels from %s", args.utt2lang_path)
    utt2lang = load_true_labels(args.utt2lang_path)
    true_labels = [utt2lang[utt_id] for utt_id in utt_ids]

    logging.info("Scoring x-vectors using cosine similarity")
    scores = cosine_similarity(xvectors_transformed, xvectors_transformed)

    logging.info("Predicting dialects based on highest cosine similarity score")
    predicted_labels = [true_labels[np.argmax(score)] for score in scores]

    logging.info("Evaluating metrics")
    accuracy, cm, precision, recall, f1 = evaluate_metrics(true_labels, predicted_labels)

    logging.info("Accuracy: %.4f", accuracy)
    logging.info("Confusion Matrix:\n%s", cm)
    logging.info("Precision: %.4f", precision)
    logging.info("Recall: %.4f", recall)
    logging.info("F1 Score: %.4f", f1)

    results_path = f"{args.results_dir}/metrics.{args.part_idx}.csv"
    logging.info("Saving metrics to %s", results_path)
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Value': [accuracy, precision, recall, f1]
    })
    metrics_df.to_csv(results_path, index=False)

    cm_path = f"{args.results_dir}/confusion_matrix.{args.part_idx}.csv"
    logging.info("Saving confusion matrix to %s", cm_path)
    cm_df = pd.DataFrame(cm, columns=[f'Pred_{i}' for i in range(cm.shape[1])], index=[f'True_{i}' for i in range(cm.shape[0])])
    cm_df.to_csv(cm_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--xvector_scp', type=str, required=True, help='Path to xvector scp file')
    parser.add_argument('--preproc_model_path', type=str, required=True, help='Path to preprocessing model h5 file')
    parser.add_argument('--utt2lang_path', type=str, required=True, help='Path to utt2lang file')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--part-idx', type=int, required=True, help='Part index for parallel processing')
    parser.add_argument('--num-parts', type=int, required=True, help='Total number of parts for parallel processing')

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    main(args)
