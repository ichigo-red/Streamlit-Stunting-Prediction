# 3. Membuat Random Forest Manual dengan History
from collections import Counter
import numpy as np

class HistoryRandomForest:
    def __init__(self, max_depth, max_features, n_trees_initial):
        self.max_depth = max_depth
        self.max_features = max_features
        self.n_trees_initial = n_trees_initial
        self.trees = []

    def _create_tree(self, X, y):
        def create_decision_tree(X, y, max_depth, depth=0):
            if len(set(y)) == 1 or depth == max_depth:
                return Counter(y).most_common(1)[0][0]

            best_feature, best_threshold, best_gini = None, None, float('inf')
            for feature in range(X.shape[1]):
                thresholds = np.unique(X[:, feature])
                for threshold in thresholds:
                    left = y[X[:, feature] <= threshold]
                    right = y[X[:, feature] > threshold]
                    if len(left) == 0 or len(right) == 0:
                        continue

                    gini = (len(left) / len(y)) * (1 - sum((np.sum(left == c) / len(left)) ** 2 for c in set(left))) + \
                           (len(right) / len(y)) * (1 - sum((np.sum(right == c) / len(right)) ** 2 for c in set(right)))

                    if gini < best_gini:
                        best_gini, best_feature, best_threshold = gini, feature, threshold

            if best_feature is None:
                return Counter(y).most_common(1)[0][0]

            left_indices = X[:, best_feature] <= best_threshold
            right_indices = X[:, best_feature] > best_threshold

            left_tree = create_decision_tree(X[left_indices], y[left_indices], max_depth, depth + 1)
            right_tree = create_decision_tree(X[right_indices], y[right_indices], max_depth, depth + 1)

            return {'feature': best_feature, 'threshold': best_threshold, 'left': left_tree, 'right': right_tree}

        return create_decision_tree(X, y, self.max_depth)

    def fit(self, X, y):
        n_samples = X.shape[0]
        for _ in range(self.n_trees_initial):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            feature_indices = np.random.choice(X.shape[1], self.max_features, replace=False)
            X_sample_subset = X_sample[:, feature_indices]

            tree = self._create_tree(X_sample_subset, y_sample)
            self.trees.append((tree, feature_indices))

    def predict_tree(self, tree, x):
        if not isinstance(tree, dict):
            return tree
        if x[tree['feature']] <= tree['threshold']:
            return self.predict_tree(tree['left'], x)
        else:
            return self.predict_tree(tree['right'], x)

    def predict(self, X):
        predictions = []
        for tree, feature_indices in self.trees:
            predictions.append([self.predict_tree(tree, x[feature_indices]) for x in X])

        predictions = np.array(predictions).T
        return [Counter(row).most_common(1)[0][0] for row in predictions]