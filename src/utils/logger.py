import mlflow


def setup_mlflow(tracking_uri, experiment_name):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
