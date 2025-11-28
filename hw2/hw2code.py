# hw2code.py
import numpy as np
from collections import Counter

def find_best_split(X, y, feature_types):
    """
    Находит лучшее разделение для узла дерева
    """
    n_samples, n_features = X.shape
    if n_samples <= 1:
        return None, None, float('inf')

    current_gini = gini_impurity(y)
    best_gini = float('inf')
    best_feature = None
    best_threshold = None

    for feature_idx in range(n_features):
        feature_values = X[:, feature_idx]
        feature_type = feature_types[feature_idx]

        if feature_type == 'real':
            # Для вещественных признаков
            sorted_indices = np.argsort(feature_values)
            sorted_features = feature_values[sorted_indices]
            sorted_y = y[sorted_indices]

            for i in range(1, n_samples):
                if sorted_features[i] != sorted_features[i-1]:
                    threshold = (sorted_features[i] + sorted_features[i-1]) / 2

                    left_y = sorted_y[:i]
                    right_y = sorted_y[i:]

                    gini = weighted_gini(left_y, right_y)

                    if gini < best_gini:
                        best_gini = gini
                        best_feature = feature_idx
                        best_threshold = threshold

        elif feature_type == 'categorical':
            # Для категориальных признаков
            categories = np.unique(feature_values)
            if len(categories) <= 1:
                continue

            # Сортируем категории по частоте положительного класса
            cat_means = []
            for cat in categories:
                mask = feature_values == cat
                if np.sum(mask) > 0:
                    cat_means.append(np.mean(y[mask]))
                else:
                    cat_means.append(0)

            sorted_categories = [cat for _, cat in sorted(zip(cat_means, categories))]

            # Перебираем возможные разделения
            for i in range(1, len(sorted_categories)):
                left_categories = set(sorted_categories[:i])

                left_mask = np.isin(feature_values, list(left_categories))
                left_y = y[left_mask]
                right_y = y[~left_mask]

                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                gini = weighted_gini(left_y, right_y)

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_idx
                    best_threshold = left_categories

    return best_feature, best_threshold, best_gini

def gini_impurity(y):
    """Вычисляет критерий Джини для массива меток"""
    if len(y) == 0:
        return 0
    p1 = np.mean(y == 1)
    p0 = 1 - p1
    return 1 - p0**2 - p1**2

def weighted_gini(left_y, right_y):
    """Вычисляет взвешенный критерий Джини для двух подмножеств"""
    n_left, n_right = len(left_y), len(right_y)
    n_total = n_left + n_right

    if n_total == 0:
        return 0

    gini_left = gini_impurity(left_y)
    gini_right = gini_impurity(right_y)

    return (n_left / n_total) * gini_left + (n_right / n_total) * gini_right

class Node:
    """Узел решающего дерева"""
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    """Решающее дерево для бинарной классификации"""

    def __init__(self, feature_types, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=None):
        self.feature_types = feature_types
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.root = None

    def fit(self, X, y):
        """Обучение дерева"""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        """Рекурсивное построение дерева"""
        n_samples = len(y)

        # Критерии остановки
        stop_condition = (
            n_samples < self.min_samples_split or
            (self.max_depth is not None and depth >= self.max_depth) or
            len(np.unique(y)) == 1 or
            n_samples < 2 * self.min_samples_leaf  # Нельзя разделить на два валидных листа
        )

        if stop_condition:
            return Node(value=self._most_common_label(y))

        # Находим лучшее разделение
        feature_idx, threshold, gini = find_best_split(X, y, self.feature_types)

        if feature_idx is None:
            return Node(value=self._most_common_label(y))

        # Создаем разделение
        left_mask = self._get_left_mask(X[:, feature_idx], feature_idx, threshold)
        right_mask = ~left_mask

        # Проверяем минимальное количество samples в листьях
        if (np.sum(left_mask) < self.min_samples_leaf or
            np.sum(right_mask) < self.min_samples_leaf):
            return Node(value=self._most_common_label(y))

        # Рекурсивно строим поддеревья
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature_idx=feature_idx, threshold=threshold,
                   left=left_subtree, right=right_subtree)

    def _get_left_mask(self, feature_column, feature_idx, threshold):
        """Получает маску для левого поддерева"""
        feature_type = self.feature_types[feature_idx]

        if feature_type == 'real':
            return feature_column <= threshold
        else:  # categorical
            return np.isin(feature_column, list(threshold))

    def _most_common_label(self, y):
        """Возвращает наиболее частую метку"""
        if len(y) == 0:
            return 0
        return Counter(y).most_common(1)[0][0]

    def predict(self, X):
        """Предсказание для массива X"""
        return np.array([self._predict_single(x, self.root) for x in X])

    def _predict_single(self, x, node):
        """Предсказание для одного образца"""
        if node.value is not None:
            return node.value

        feature_value = x[node.feature_idx]
        feature_type = self.feature_types[node.feature_idx]

        if feature_type == 'real':
            if feature_value <= node.threshold:
                return self._predict_single(x, node.left)
            else:
                return self._predict_single(x, node.right)
        else:  # categorical
            if feature_value in node.threshold:
                return self._predict_single(x, node.left)
            else:
                return self._predict_single(x, node.right)

    def get_depth(self):
        """Возвращает глубину дерева"""
        return self._get_node_depth(self.root)

    def _get_node_depth(self, node):
        """Рекурсивно вычисляет глубину узла"""
        if node.value is not None:
            return 0
        return 1 + max(self._get_node_depth(node.left), self._get_node_depth(node.right))
    def get_params(self, deep=True):
        return {
            'feature_types': self.feature_types,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'random_state': self.random_state
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
