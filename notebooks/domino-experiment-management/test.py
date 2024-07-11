# create a new experiment (or name existing experiment to add more runs)
mlflow.set_experiment(experiment_name="custom-model")
# local dependencies
code_deps = [
    "prev-pipeline-mlflow/code/" + dep
    for dep in os.listdir("prev-pipeline-mlflow/code")
]
# name run -- this should be dynamic
run_name = "register-model-6"
with mlflow.start_run(run_name=run_name):
    # https://mlflow.org/docs/2.6.0/python_api/mlflow.pyfunc.html
    # log model and register
    mlflow.pyfunc.log_model(
        artifact_path="prev-pipeline-mlflow-artifacts",
        loader_module="application_package.mlflow_utils",
        data_path="prev-pipeline-mlflow/data",
        code_path=code_deps,
        registered_model_name="prev-pipeline-model",
    )
# end run
mlflow.end_run()