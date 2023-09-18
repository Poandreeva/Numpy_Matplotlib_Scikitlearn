# 1. Импорт необходимых библиотек
import pandas as pd
import sklearn

# 2. Загрузка базы данных
# Перечислим заголовки колонок с признаками
header = ['age','anaemia','creatinine_phosphokinase','diabetes',
          'ejection_fraction','high_blood_pressure','platelets',
          'serum_creatinine','serum_sodium','sex','smoking','time',
          'DEATH_EVENT']
# Сформируем DataFrame на основе загруженной базы данных (файл базы данных поместить в одну папку с файлом python)
data = pd.read_csv('heart_failure_clinical_records_dataset.csv', names=header)

# 3. Разделим данные на вектор признаков Х и целевую переменную Y
Y = data['DEATH_EVENT'].values
X = data.drop(columns=['DEATH_EVENT'])

# 4. Нормализуем значения вектора признаков
X = (X-X.min())/(X.max()-X.min())

# 5. Рандомизируем выборку и разделим на обучающую и тестовую выборки
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=1, stratify=Y)

# 6. Решим задачу классификации методом логистической регрессии
from sklearn.linear_model import LogisticRegression
LR_model = LogisticRegression(solver='liblinear')
# Обучим модель на обучающей выборке
LR_model.fit(X_train, Y_train)
# Предскажем класс тестовой выборки
LR_prediction = LR_model.predict(X_test)

# 7. Оценим точность классификации
LR_train_accuracy = LR_model.score(X_train, Y_train)
LR_test_accuracy = LR_model.score(X_test, Y_test)
print ('Точность модели на обучающей выборке: ', round (LR_train_accuracy, 2))
print ('Точность модели на тестовой выборке: ', round (LR_test_accuracy, 2))
# Точность модели на обучающей выборке:  0.86
# Точность модели на тестовой выборке:  0.8


# 8. Построим матрицу несоответствий
from sklearn.metrics import confusion_matrix
print('Матрица несоответствий метода LR:\n',confusion_matrix(LR_prediction, Y_test))
# Матрица несоответствий метода LR:
#  [[65 18]
#  [ 2 14]]


# 9. Метод k-ближайших соседей
from sklearn.neighbors import KNeighborsClassifier
KNN_model = KNeighborsClassifier(n_neighbors = 11)
KNN_model.fit(X_train,Y_train)
KNN_prediction = KNN_model.predict(X_test)
print('Матрица несоответствий метода KNN:\n', confusion_matrix(KNN_prediction, Y_test))
KNN_train_accuracy = KNN_model.score(X_train, Y_train)
KNN_test_accuracy = KNN_model.score(X_test, Y_test)
print ('Точность модели на обучающей выборке: ', round (KNN_train_accuracy, 2))
print ('Точность модели на тестовой выборке: ', round (KNN_test_accuracy, 2))
# Матрица несоответствий метода KNN:
#  [[64 28]
#  [ 3  4]]
# Точность модели на обучающей выборке:  0.76
# Точность модели на тестовой выборке:  0.69


# 10. Метод линейного дискриминантного анализа
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
LDA_model = LinearDiscriminantAnalysis()
LDA_model.fit(X_train, Y_train)
LDA_prediction = LDA_model.predict(X_test)
print('Матрица несоответствий метода LDA:\n', confusion_matrix(LDA_prediction, Y_test))
LDA_train_accuracy = LDA_model.score(X_train, Y_train)
LDA_test_accuracy = LDA_model.score(X_test, Y_test)
print ('Точность модели на обучающей выборке: ', round (LDA_train_accuracy, 2))
print ('Точность модели на тестовой выборке: ', round (LDA_test_accuracy, 2))
# Матрица несоответствий метода LDA:
#  [[63 16]
#  [ 4 16]]
# Точность модели на обучающей выборке:  0.85
# Точность модели на тестовой выборке:  0.8


# 11. Метод наивный байесовский
from sklearn.naive_bayes import GaussianNB
GNB_model = GaussianNB()
GNB_model.fit(X_train, Y_train)
GNB_prediction = GNB_model.predict(X_test)
print('Матрица несоответствий метода GNB:\n', confusion_matrix(GNB_prediction, Y_test))
GNB_train_accuracy = GNB_model.score(X_train, Y_train)
GNB_test_accuracy = GNB_model.score(X_test, Y_test)
print ('Точность модели на обучающей выборке: ', round (GNB_train_accuracy, 2))
print ('Точность модели на тестовой выборке: ', round (GNB_test_accuracy, 2))
# Матрица несоответствий метода GNB:
#  [[64 21]
#  [ 3 11]]
# Точность модели на обучающей выборке:  0.77
# Точность модели на тестовой выборке:  0.76


# 12. Метод дерева решений
from sklearn.tree import DecisionTreeClassifier
DTC_model = DecisionTreeClassifier()
DTC_model.fit(X_train, Y_train)
DTC_prediction = DTC_model.predict(X_test)
print('Матрица несоответствий метода DTC:\n', confusion_matrix(DTC_prediction, Y_test))
DTC_train_accuracy = DTC_model.score(X_train, Y_train)
DTC_test_accuracy = DTC_model.score(X_test, Y_test)
print ('Точность модели на обучающей выборке: ', round (DTC_train_accuracy, 2))
print ('Точность модели на тестовой выборке: ', round (DTC_test_accuracy, 2))
# Матрица несоответствий метода DTC:
#  [[52 11]
#  [15 21]]
# Точность модели на обучающей выборке:  1.0
# Точность модели на тестовой выборке:  0.74


# 13. Метод опорных векторов
from sklearn.svm import SVC
SVC_model = SVC(gamma='scale')
SVC_model.fit(X_train, Y_train)
SVC_prediction = SVC_model.predict(X_test)
print('Матрица несоответствий метода SVC:\n', confusion_matrix(SVC_prediction, Y_test))
SVC_train_accuracy = SVC_model.score(X_train, Y_train)
SVC_test_accuracy = SVC_model.score(X_test, Y_test)
print ('Точность модели на обучающей выборке: ', round (SVC_train_accuracy, 2))
print ('Точность модели на тестовой выборке: ', round (SVC_test_accuracy, 2))
# Матрица несоответствий метода SVC:
#  [[65 21]
#  [ 2 11]]
# Точность модели на обучающей выборке:  0.88
# Точность модели на тестовой выборке:  0.77

