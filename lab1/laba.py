import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# 1. EDA - Разведочный анализ данных
print("=== 1. EDA - Разведочный анализ данных ===")

# Загрузка данных
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(f"Размер тренировочных данных: {train_df.shape}")
print(f"Размер тестовых данных: {test_df.shape}")

# Основная информация о данных
print("\nИнформация о тренировочных данных:")
print(train_df.info())
print("\nПервые 5 строк:")
print(train_df.head())

# Статистика
print("\nСтатистика данных:")
print(train_df.describe())

# Проверка на пропущенные значения
print("\nПропущенные значения:")
print(train_df.isnull().sum())

# Анализ целевой переменной
plt.figure(figsize=(15, 12))

plt.subplot(3, 3, 1)
plt.hist(train_df['RiskScore'].dropna(), bins=50, alpha=0.7, color='blue')
plt.title('Распределение RiskScore')
plt.xlabel('RiskScore')
plt.ylabel('Частота')

# Анализ выбросов в целевой переменной
plt.subplot(3, 3, 2)
plt.boxplot(train_df['RiskScore'].dropna())
plt.title('Boxplot RiskScore')

# Матрица корреляций
plt.subplot(3, 3, 3)
numeric_cols = train_df.select_dtypes(include=[np.number]).columns
# Берем только первые 10 числовых признаков для читаемости
correlation_matrix = train_df[numeric_cols[:10]].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Матрица корреляций (первые 10 признаков)')

# Графики зависимостей некоторых признаков от целевой переменной
sample_features = ['Age', 'AnnualIncome', 'CreditScore', 'LoanAmount']
for i, feature in enumerate(sample_features):
    if feature in train_df.columns:
        plt.subplot(3, 3, i+4)
        plt.scatter(train_df[feature], train_df['RiskScore'], alpha=0.3)
        plt.xlabel(feature)
        plt.ylabel('RiskScore')
        plt.title(f'{feature} vs RiskScore')

plt.tight_layout()
plt.show()

# Обработка пропущенных значений
print("\n=== Обработка пропущенных значений ===")

def handle_missing_values(train_df, test_df):
    """Обработка пропущенных значений"""
    # Создаем копии данных
    train_clean = train_df.copy()
    test_clean = test_df.copy()

    # Разделяем числовые и категориальные признаки (исключая целевую переменную)
    numeric_features = train_clean.select_dtypes(include=[np.number]).columns
    # Убираем целевую переменную из числовых признаков для обработки
    if 'RiskScore' in numeric_features:
        numeric_features = numeric_features.drop('RiskScore')

    categorical_features = train_clean.select_dtypes(include=['object']).columns

    print(f"Обрабатываем числовые признаки: {list(numeric_features)}")
    print(f"Обрабатываем категориальные признаки: {list(categorical_features)}")

    # Для числовых признаков заполняем медианой
    numeric_imputer = SimpleImputer(strategy='median')
    train_clean[numeric_features] = numeric_imputer.fit_transform(train_clean[numeric_features])
    test_clean[numeric_features] = numeric_imputer.transform(test_clean[numeric_features])

    # Для категориальных признаков заполняем самым частым значением
    if len(categorical_features) > 0:
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        train_clean[categorical_features] = categorical_imputer.fit_transform(train_clean[categorical_features])
        test_clean[categorical_features] = categorical_imputer.transform(test_clean[categorical_features])

    # Для целевой переменной в тренировочных данных заполняем медианой
    if 'RiskScore' in train_clean.columns:
        target_imputer = SimpleImputer(strategy='median')
        train_clean['RiskScore'] = target_imputer.fit_transform(train_clean[['RiskScore']])

    return train_clean, test_clean

# Обрабатываем пропущенные значения
train_clean, test_clean = handle_missing_values(train_df, test_df)

print("Пропущенные значения после обработки:")
print(f"Тренировочные данные: {train_clean.isnull().sum().sum()}")
print(f"Тестовые данные: {test_clean.isnull().sum().sum()}")

# 2. Нормализация данных
print("\n=== 2. Нормализация данных ===")

class DataNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None
        self.min = None
        self.max = None

    def z_score_normalize(self, X):
        """Z-score нормализация"""
        if self.mean is None or self.std is None:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            # Защита от деления на ноль
            self.std[self.std == 0] = 1

        return (X - self.mean) / self.std

    def min_max_normalize(self, X):
        """Min-Max нормализация"""
        if self.min is None or self.max is None:
            self.min = np.min(X, axis=0)
            self.max = np.max(X, axis=0)
            # Защита от случая, когда min == max
            range_vals = self.max - self.min
            range_vals[range_vals == 0] = 1
            self.range = range_vals

        return (X - self.min) / self.range

# 3. Реализация линейной регрессии
print("\n=== 3. Реализация линейной регрессии ===")

class LinearRegressionModel:
    def __init__(self, learning_rate=0.01, n_iter=1000, method='analytic', batch_size=32):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.method = method
        self.batch_size = batch_size
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _add_intercept(self, X):
        """Добавляет столбец единиц для intercept"""
        return np.c_[np.ones(X.shape[0]), X]

    def fit_analytic(self, X, y):
        """Аналитическое решение (нормальное уравнение)"""
        X_with_intercept = self._add_intercept(X)

        # Нормальное уравнение: theta = (X^T * X)^(-1) * X^T * y
        try:
            theta = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
        except np.linalg.LinAlgError:
            # Если матрица вырождена, используем псевдообратную
            theta = np.linalg.pinv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y

        self.bias = theta[0]
        self.weights = theta[1:]

    def fit_gradient_descent(self, X, y):
        """Градиентный спуск"""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iter):
            # Предсказания
            y_pred = X @ self.weights + self.bias

            # Градиенты
            dw = (1/n_samples) * X.T @ (y_pred - y)
            db = (1/n_samples) * np.sum(y_pred - y)

            # Обновление параметров
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Сохранение потерь
            loss = np.mean((y_pred - y) ** 2)
            if i % 100 == 0:  # Сохраняем потери каждые 100 итераций
                self.loss_history.append(loss)

    def fit_stochastic_gd(self, X, y):
        """Стохастический градиентный спуск"""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iter):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for j in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[j:j+self.batch_size]
                y_batch = y_shuffled[j:j+self.batch_size]

                if len(X_batch) == 0:
                    continue

                # Предсказания для батча
                y_pred_batch = X_batch @ self.weights + self.bias

                # Градиенты для батча
                dw = (1/len(X_batch)) * X_batch.T @ (y_pred_batch - y_batch)
                db = (1/len(X_batch)) * np.sum(y_pred_batch - y_batch)

                # Обновление параметров
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            # Сохранение потерь после эпохи
            y_pred = X @ self.weights + self.bias
            loss = np.mean((y_pred - y) ** 2)
            self.loss_history.append(loss)

    def fit(self, X, y):
        """Основной метод обучения"""
        if self.method == 'analytic':
            self.fit_analytic(X, y)
        elif self.method == 'gradient_descent':
            self.fit_gradient_descent(X, y)
        elif self.method == 'stochastic_gd':
            self.fit_stochastic_gd(X, y)
        else:
            raise ValueError("Метод должен быть 'analytic', 'gradient_descent' или 'stochastic_gd'")

    def predict(self, X):
        """Предсказание"""
        if self.weights is None or self.bias is None:
            raise Exception("Модель не обучена. Сначала вызовите fit().")
        return X @ self.weights + self.bias

# 4. Кросс-валидация
print("\n=== 4. Реализация кросс-валидации ===")

class CrossValidation:
    @staticmethod
    def k_fold_cv(model, X, y, k=5, metric='mse'):
        """K-fold кросс-валидация"""
        n_samples = len(X)
        fold_size = n_samples // k
        indices = np.random.permutation(n_samples)
        scores = []

        for i in range(k):
            # Разделение на тренировочную и валидационную выборки
            start = i * fold_size
            end = (i + 1) * fold_size if i < k - 1 else n_samples

            val_indices = indices[start:end]
            train_indices = np.concatenate([indices[:start], indices[end:]])

            X_train, X_val = X[train_indices], X[val_indices]
            y_train, y_val = y[train_indices], y[val_indices]

            # Обучение и предсказание
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            # Вычисление метрики
            if metric == 'mse':
                score = Metrics.mse(y_val, y_pred)
            elif metric == 'mae':
                score = Metrics.mae(y_val, y_pred)
            elif metric == 'r2':
                score = Metrics.r2(y_val, y_pred)
            elif metric == 'mape':
                score = Metrics.mape(y_val, y_pred)

            scores.append(score)

        return np.mean(scores), np.std(scores)

    @staticmethod
    def leave_one_out_cv(model, X, y, metric='mse'):
        """Leave-one-out кросс-валидация"""
        n_samples = len(X)
        scores = []

        # Для больших данных используем только подвыборку
        sample_size = min(100, n_samples)
        indices = np.random.choice(n_samples, sample_size, replace=False)

        for i in indices:
            # Одна выборка для валидации, остальные для тренировки
            train_indices = np.array([j for j in indices if j != i])
            val_indices = np.array([i])

            X_train, X_val = X[train_indices], X[val_indices]
            y_train, y_val = y[train_indices], y[val_indices]

            # Обучение и предсказание
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            # Вычисление метрики
            if metric == 'mse':
                score = Metrics.mse(y_val, y_pred)
            elif metric == 'mae':
                score = Metrics.mae(y_val, y_pred)
            elif metric == 'r2':
                score = Metrics.r2(y_val, y_pred)
            elif metric == 'mape':
                score = Metrics.mape(y_val, y_pred)

            scores.append(score)

        return np.mean(scores), np.std(scores)

# 5. Метрики
print("\n=== 5. Реализация метрик ===")

class Metrics:
    @staticmethod
    def mse(y_true, y_pred):
        """Mean Squared Error"""
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def mae(y_true, y_pred):
        """Mean Absolute Error"""
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def r2(y_true, y_pred):
        """R-squared"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    @staticmethod
    def mape(y_true, y_pred):
        """Mean Absolute Percentage Error"""
        # Избегаем деления на ноль
        mask = y_true != 0
        if np.sum(mask) == 0:
            return 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# 6. Регуляризация (дополнительно)
print("\n=== 6. Реализация регуляризации ===")

class RegularizedLinearRegression(LinearRegressionModel):
    def __init__(self, learning_rate=0.01, n_iter=1000, method='gradient_descent',
                 regularization=None, alpha=0.1, p=2, batch_size=32):
        super().__init__(learning_rate, n_iter, method, batch_size)
        self.regularization = regularization
        self.alpha = alpha
        self.p = p

    def _regularization_penalty(self, weights):
        """Вычисление штрафа регуляризации"""
        if self.regularization == 'l1':
            return self.alpha * np.sum(np.abs(weights))
        elif self.regularization == 'l2':
            return self.alpha * np.sum(weights ** 2)
        elif self.regularization == 'l1_l2':
            return self.alpha * (0.5 * np.sum(weights ** 2) + 0.5 * np.sum(np.abs(weights)))
        elif self.regularization == 'lp':
            return self.alpha * np.sum(np.abs(weights) ** self.p)
        else:
            return 0

    def _regularization_gradient(self, weights):
        """Градиент регуляризации"""
        if self.regularization == 'l1':
            return self.alpha * np.sign(weights)
        elif self.regularization == 'l2':
            return self.alpha * 2 * weights
        elif self.regularization == 'l1_l2':
            return self.alpha * (weights + 0.5 * np.sign(weights))
        elif self.regularization == 'lp':
            # Аппроксимация для произвольного p
            epsilon = 1e-8
            return self.alpha * self.p * np.sign(weights) * (np.abs(weights) + epsilon) ** (self.p - 1)
        else:
            return np.zeros_like(weights)

    def fit_gradient_descent(self, X, y):
        """Градиентный спуск с регуляризацией"""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iter):
            # Предсказания
            y_pred = X @ self.weights + self.bias

            # Градиенты
            dw = (1/n_samples) * X.T @ (y_pred - y) + self._regularization_gradient(self.weights)
            db = (1/n_samples) * np.sum(y_pred - y)

            # Обновление параметров
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Сохранение потерь
            loss = np.mean((y_pred - y) ** 2) + self._regularization_penalty(self.weights)
            if i % 100 == 0:
                self.loss_history.append(loss)

# Подготовка данных для обучения
print("\n=== Подготовка данных для обучения ===")

# Выбираем только числовые признаки (исключая целевую переменную)
numeric_features = train_clean.select_dtypes(include=[np.number]).columns.drop('RiskScore', errors='ignore')

# Исключаем признаки, которые могут быть идентификаторами или не несут полезной информации
# Также исключаем признаки, которые сильно коррелируют или могут вызвать утечку данных
exclude_features = ['BaseInterestRate', 'InterestRate', 'MonthlyLoanPayment']  # Пример исключения
numeric_features = [col for col in numeric_features if col not in exclude_features]

print(f"Используемые признаки: {list(numeric_features)}")

X = train_clean[numeric_features].values
y = train_clean['RiskScore'].values

print(f"Размер X: {X.shape}, Размер y: {y.shape}")

# Проверяем на наличие выбросов в целевой переменной
print(f"\nСтатистика целевой переменной:")
print(f"Min: {np.min(y):.2f}, Max: {np.max(y):.2f}")
print(f"Mean: {np.mean(y):.2f}, Std: {np.std(y):.2f}")
print(f"Median: {np.median(y):.2f}")

# Обработка выбросов в целевой переменной - более безопасный подход
print("Обработка выбросов в целевой переменной...")

# Вместо логарифмирования используем обрезку выбросов
def remove_outliers_iqr(data, column):
    Q1 = np.percentile(data[column], 25)
    Q3 = np.percentile(data[column], 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Создаем копию данных без выбросов
    clean_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return clean_data

# Применяем обрезку выбросов
train_clean_no_outliers = remove_outliers_iqr(train_clean, 'RiskScore')
print(f"Размер данных после удаления выбросов: {train_clean_no_outliers.shape}")

# Используем данные без выбросов для обучения
X_clean = train_clean_no_outliers[numeric_features].values
y_clean = train_clean_no_outliers['RiskScore'].values

print(f"Размер X после удаления выбросов: {X_clean.shape}, Размер y: {y_clean.shape}")

# Нормализация
normalizer = DataNormalizer()
X_normalized = normalizer.z_score_normalize(X_clean)

# Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_clean, test_size=0.2, random_state=42)

print(f"Размеры данных: X_train {X_train.shape}, X_test {X_test.shape}")

# Тестирование реализации
print("\n=== Тестирование реализации ===")

print("Сравнение методов обучения:")

# Аналитический метод
print("\n1. Аналитический метод:")
lr_analytic = LinearRegressionModel(method='analytic')
lr_analytic.fit(X_train, y_train)
y_pred_analytic = lr_analytic.predict(X_test)

# Градиентный спуск
print("2. Градиентный спуск:")
lr_gd = LinearRegressionModel(method='gradient_descent', learning_rate=0.01, n_iter=1000)
lr_gd.fit(X_train, y_train)
y_pred_gd = lr_gd.predict(X_test)

# Стохастический градиентный спуск
print("3. Стохастический градиентный спуск:")
lr_sgd = LinearRegressionModel(method='stochastic_gd', learning_rate=0.01, n_iter=100, batch_size=32)
lr_sgd.fit(X_train, y_train)
y_pred_sgd = lr_sgd.predict(X_test)

# Sklearn для сравнения
print("4. Sklearn LinearRegression:")
lr_sklearn = LinearRegression()
lr_sklearn.fit(X_train, y_train)
y_pred_sklearn = lr_sklearn.predict(X_test)

# Сравнение метрик
methods = ['Аналитический', 'Градиентный спуск', 'Стохастический GD', 'Sklearn']
predictions = [y_pred_analytic, y_pred_gd, y_pred_sgd, y_pred_sklearn]

print("\nСравнение метрик:")
print("Метод\t\t\tMSE\t\tMAE\t\tR2\t\tMAPE")
for i, (method, y_pred) in enumerate(zip(methods, predictions)):
    mse_custom = Metrics.mse(y_test, y_pred)
    mae_custom = Metrics.mae(y_test, y_pred)
    r2_custom = Metrics.r2(y_test, y_pred)
    mape_custom = Metrics.mape(y_test, y_pred)

    print(f"{method:20} {mse_custom:.4f}\t\t{mae_custom:.4f}\t\t{r2_custom:.4f}\t\t{mape_custom:.4f}")

# Сравнение с sklearn метриками
print("\nСравнение с sklearn метриками:")
y_pred_test = lr_analytic.predict(X_test)

print(f"MSE - Custom: {Metrics.mse(y_test, y_pred_test):.4f}, Sklearn: {mean_squared_error(y_test, y_pred_test):.4f}")
print(f"MAE - Custom: {Metrics.mae(y_test, y_pred_test):.4f}, Sklearn: {mean_absolute_error(y_test, y_pred_test):.4f}")
print(f"R2 - Custom: {Metrics.r2(y_test, y_pred_test):.4f}, Sklearn: {r2_score(y_test, y_pred_test):.4f}")

# Кросс-валидация
print("\n=== Кросс-валидация ===")

cv = CrossValidation()
k_fold_score, k_fold_std = cv.k_fold_cv(LinearRegressionModel(method='analytic'), X_normalized, y_clean, k=5, metric='mse')
print(f"K-Fold CV (k=5) MSE: {k_fold_score:.4f} ± {k_fold_std:.4f}")

# Для больших данных LOO используем только подвыборку
loo_score, loo_std = cv.leave_one_out_cv(LinearRegressionModel(method='analytic'), X_normalized, y_clean, metric='mse')
print(f"LOO CV MSE (на 100 samples): {loo_score:.4f} ± {loo_std:.4f}")

# Тестирование регуляризации
print("\n=== Тестирование регуляризации ===")

regularization_types = ['l1', 'l2', 'l1_l2', 'lp']
for reg_type in regularization_types:
    p = 1.5 if reg_type == 'lp' else 2
    reg_lr = RegularizedLinearRegression(method='gradient_descent', regularization=reg_type, alpha=0.1, p=p)
    reg_lr.fit(X_train, y_train)
    y_pred_reg = reg_lr.predict(X_test)
    mse_reg = Metrics.mse(y_test, y_pred_reg)
    print(f"{reg_type.upper()} регуляризация MSE: {mse_reg:.4f}")

# Финальная модель для submission
print("\n=== Подготовка финальной модели ===")

# Используем всю тренировочную выборку для финального обучения
final_model = LinearRegressionModel(method='analytic')
final_model.fit(X_normalized, y_clean)

# Предобработка тестовых данных
X_test_final = test_clean[numeric_features].values
X_test_normalized = normalizer.z_score_normalize(X_test_final)

# Предсказания
final_predictions = final_model.predict(X_test_normalized)

# Создание submission файла
submission = pd.DataFrame({
    'Id': test_clean.index,
    'RiskScore': final_predictions
})

submission.to_csv('submission.csv', index=False)
print("Submission файл создан: submission.csv")

# Проверка качества на тренировочных данных
train_predictions = final_model.predict(X_normalized)
final_mse = Metrics.mse(y_clean, train_predictions)
final_r2 = Metrics.r2(y_clean, train_predictions)

print(f"\nФинальные метрики на тренировочных данных:")
print(f"MSE: {final_mse:.4f}")
print(f"R2: {final_r2:.4f}")

if final_mse < 25.00:
    print("✅ MSE < 25.00 - условие выполнено!")
else:
    print("❌ MSE > 25.00 - необходимо улучшить модель")
    print("Рекомендации:")
    print("1. Попробуйте feature engineering")
    print("2. Используйте регуляризацию")
    print("3. Попробуйте другие комбинации признаков")

# Визуализация прогресса обучения
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
if hasattr(lr_gd, 'loss_history') and lr_gd.loss_history:
    plt.plot(lr_gd.loss_history)
    plt.title('Градиентный спуск - История потерь')
    plt.xlabel('Итерация (каждые 100)')
    plt.ylabel('MSE')

plt.subplot(1, 3, 2)
if hasattr(lr_sgd, 'loss_history') and lr_sgd.loss_history:
    plt.plot(lr_sgd.loss_history)
    plt.title('Стохастический GD - История потерь')
    plt.xlabel('Эпоха')
    plt.ylabel('MSE')

plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred_analytic, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Истинные значения')
plt.ylabel('Предсказанные значения')
plt.title('Предсказания vs Истинные значения')

plt.tight_layout()
plt.show()

print("\n=== Лабораторная работа завершена! ===")