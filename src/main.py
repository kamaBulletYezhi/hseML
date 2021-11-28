from data.readdataset import read_spam_dataset, read_cancer_dataset
from data.splitdataset import train_test_split
from reports.performance_metrics import plot_precision_recall, plot_roc_curve
from src.basedir import BASE_DIR


def cancer_results():
    path = BASE_DIR+"/reports/figures/cancer"
    cancer_dataset = BASE_DIR+"/data/raw/cancer.csv"
    X, y = read_cancer_dataset(cancer_dataset)
    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.9)
    print("wait, precision/recall is calculated")
    plot_precision_recall(X_train, y_train, X_test, y_test, path=path)
    print("wait, roc-auc is calculated")
    plot_roc_curve(X_train, y_train, X_test, y_test, max_k=10, path=path)
    print("graphs for CANCER have been saved to " + path)


def spam_results():
    path = BASE_DIR+"/reports/figures/spam"
    spam_dataset = BASE_DIR+"/data/raw/spam.csv"
    X, y = read_spam_dataset(spam_dataset)
    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.9)
    print("wait, precision/recall is calculated")
    plot_precision_recall(X_train, y_train, X_test, y_test, path=path)
    print("wait, roc-auc is calculated")
    plot_roc_curve(X_train, y_train, X_test, y_test, max_k=10, path=path)
    print("graphs for SPAM have been saved to " + path)


if __name__ == '__main__':
    cancer_results()
    spam_results()
