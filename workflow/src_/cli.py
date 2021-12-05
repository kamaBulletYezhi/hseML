import click
from src.data.readdataset import read_cancer_dataset, read_spam_dataset
from src.data.splitdataset import train_test_split
from src.reports.performance_metrics import plot_precision_recall, plot_roc_curve


@click.group()
def cli():
    pass


@click.command()
@click.argument("path_to_csv", type=click.Path())
@click.argument("path_to_pre_rec", type=click.Path())
@click.argument("path_to_roc", type=click.Path())
def read_and_graph_cancer(path_to_csv, path_to_pre_rec, path_to_roc):
    X, y = read_cancer_dataset(path_to_csv)
    Xy_train_test = train_test_split(X, y, ratio=0.9)
    plot_precision_recall(*Xy_train_test, path=path_to_pre_rec)
    plot_roc_curve(*Xy_train_test, max_k=10, path=path_to_roc)


@click.command()
@click.argument("path_to_csv", type=click.Path())
@click.argument("path_to_pre_rec", type=click.Path())
@click.argument("path_to_roc", type=click.Path())
def read_and_graph_spam(path_to_csv, path_to_pre_rec, path_to_roc):
    X, y = read_spam_dataset(path_to_csv)
    Xy_train_test = train_test_split(X, y, ratio=0.9)
    plot_precision_recall(*Xy_train_test, path=path_to_pre_rec)
    plot_roc_curve(*Xy_train_test, max_k=10, path=path_to_roc)


if __name__ == "__main__":
    cli()
