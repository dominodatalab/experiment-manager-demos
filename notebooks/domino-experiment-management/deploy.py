import mlflow
import os
os.chdir(os.getcwd() + "/src")

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
import mlflow
import mlflow.pyfunc
import os
os.chdir(os.getcwd() + "/src")

"""
Testing out the Domino experiment functionality (https://docs.dominodatalab.com/en/5.8/user_guide/da707d/track-and-monitor-experiments/#_auto_logging).
"""
# load model version
model_version = 14
model = mlflow.pyfunc.load_model(model_uri=f"models:/prev-pipeline-model/{model_version}")
# set experiment (or create experiment)
mlflow.set_experiment(experiment_name="sample-experiement")
with mlflow.start_run(run_name="sample-run", tags={"model-version": f"{model_version}"}):
    # sample predictions
    test = {
        "claim_number": ["ABC123", "DEF123"],
        "loss_cause": ["Cut/Puncture", "Cut/Puncture"],
        "loss_desc": ["employee cut hand with knife", "employee cut hand with knife"],
        "note_text": [
            "employee cut hand with knife in the kitchen. knife was dull and employee was not wearing gloves or hand protection",
            "called employee. did not answer",
        ]}

    # make predictions
    pred = model.predict(test)
    # count number in each class
    human_behavior_claims = len(
        [i for i in pred["loss_source_pred"] if i == "Human Behavior"]
    )
    physical_control_claims = len(pred["loss_source_pred"]) - human_behavior_claims
    preventable_claims = len(
        [i for i in pred["preventable_pred"] if i == "Preventable"]
    )
    non_preventable_claims = len(pred["preventable_pred"]) - preventable_claims
    # manually log parameters and metrics
    mlflow.log_param("batch_size", len(test["claim_number"]))
    mlflow.log_param("scored_data_size", len(pred["claim_number"]))
    mlflow.log_metric("human_behavior_claims", human_behavior_claims)
    mlflow.log_metric("physical_control_claims", physical_control_claims)
    mlflow.log_metric("preventable_claims", preventable_claims)
    mlflow.log_metric("non_preventable_claims", non_preventable_claims)
# end run
mlflow.end_run()