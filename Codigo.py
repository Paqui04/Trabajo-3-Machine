import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.exceptions import UndefinedMetricWarning
import warnings

# Ignorar advertencias
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Cargar el dataset de Yeast desde el archivo local
file_path = 'yeast.data'  # Asegúrate de poner la ruta correcta al archivo descargado
df = pd.read_csv(file_path, sep=r'\s+', header=None)

# Asignar nombres a las columnas según la documentación del dataset
df.columns = ['SequenceName', 'mcg', 'gvh', 'alm', 'mit', 'erl', 'pox', 'vac', 'nuc', 'Class']

# Eliminar la columna de nombres de secuencias ya que no es útil para la clasificación
df = df.drop(['SequenceName'], axis=1)

# Convertir la variable de respuesta 'Class' a valores numéricos
label_encoders = {}
le = LabelEncoder()
df['Class'] = le.fit_transform(df['Class'])
label_encoders['Class'] = le

# Imputación de valores faltantes (aunque en este dataset no debería haber)
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Verificar si el dataset está desbalanceado
print(df['Class'].value_counts())

# Balancear el dataset usando SMOTE
X = df_imputed.drop('Class', axis=1)
y = df_imputed['Class']

# Ajustar k_neighbors en SMOTE para clases minoritarias
smote = SMOTE(k_neighbors=min(df['Class'].value_counts().min() - 1, 5), random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Dividir el dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Definir los modelos
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier(algorithm='SAMME')
}

# Afinar hiperparámetros y entrenar modelos
param_grids = {
    'Decision Tree': {'max_depth': [3, 5, 7, None]},
    'Random Forest': {'n_estimators': [10, 50, 100], 'max_depth': [3, 5, 7, None]},
    'AdaBoost': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}
}

best_models = {}
for model_name in models:
    grid_search = GridSearchCV(models[model_name], param_grids[model_name], cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_models[model_name] = grid_search.best_estimator_

# Evaluar modelos usando validación cruzada
for model_name in best_models:
    scores = cross_val_score(best_models[model_name], X_train, y_train, cv=5, scoring='accuracy')
    print(f"{model_name} Accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})")

# Evaluar el desempeño de los modelos
for model_name in best_models:
    y_pred = best_models[model_name].predict(X_test)
    print(f"{model_name} Classification Report:\n")
    print(classification_report(y_test, y_pred))
