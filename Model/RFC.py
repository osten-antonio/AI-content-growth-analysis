import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from scipy.sparse import csr_matrix, load_npz, vstack
import pickle

X_train=load_npz("X_train.npz")
X_test=load_npz("X_test.npz")

y_train=np.load("Y_train.npy")
y_test=np.load("Y_test.npy")

X = vstack([X_train,X_test])

y = np.concatenate((y_train,y_test),axis=0)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    print("\n" + "="*60)
    print(f"Fold {fold+1}/5")
    print("="*60)

    X_train_raw = X[train_idx]
    y_train_raw = y[train_idx]

    X_test_raw = X[test_idx]
    y_test_raw = y[test_idx]

    X_train = csr_matrix(X_train_raw)
    X_test = csr_matrix(X_test_raw)

    model = RandomForestClassifier(min_samples_leaf=1, min_samples_split=5, n_estimators=300, bootstrap=False, n_jobs=-1)
    model.fit(X_train, y_train_raw)

    train_preds = model.predict(X_train)
    train_probs = model.predict_proba(X_train)[:,1]
    
    train_acc = accuracy_score(y_train_raw, train_preds)
    train_auc = roc_auc_score(y_train_raw, train_probs)
    train_f1 = f1_score(y_train_raw, train_preds)

    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)[:,1]
    
    test_acc = accuracy_score(y_test_raw, test_preds)
    test_auc = roc_auc_score(y_test_raw, test_probs)
    test_f1 = f1_score(y_test_raw, test_preds)

    print(f"Train - Accuracy: {train_acc}, AUC: {train_auc}, F1: {train_f1}")
    print(f"Test  - Accuracy: {test_acc}, AUC: {test_auc}, F1: {test_f1}")

    filename = f'Tree fold {fold+1}.pkl'
    pickle.dump(model, open(filename, 'wb'))