
# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from google.cloud import aiplatform


aiplatform.init(
    # your Google Cloud Project ID or number
    # environment default used is not set
    project='pyro-284215',

    # the Vertex AI region you will use
    # defaults to us-central1
    location='us-central1',

    # Google Cloud Storage bucket in same region as location
    # used to stage artifacts
    staging_bucket='gs://raj_ag_test_bucket',

    # the name of the experiment to use to track
    # logged metrics and parameters
    experiment='my-test-experiment',

    # description of the experiment above
    experiment_description='testing for using Vertex AI'
)



# def create_custom_job(
#     project: str,
#     location: str,
#     staging_bucket: str,
#     display_name: str,
#     script_path: str,
#     config_file: str,
#     container_uri: str,
#     replica_count: int = 1,
#     machine_type: str = "n1-highmem-16",
#     accelerator_type: str = "NVIDIA_TESLA_T4",
#     accelerator_count: int = 4,
#     boot_disk_type: str = "pd-ssd",
#     boot_disk_size_gb: int = 100,
#     labels: dict[str, str] | None = None,
#     experiment: str | None = None,
#     experiment_run: str | None = None,
# ) -> None:
#     aiplatform.init(project=project, location=location, staging_bucket=staging_bucket)

#     args = ["fit", "--config", config_file]

#     worker_pool_specs = aiplatform.utils.worker_spec_utils._DistributedTrainingSpec.chief_worker_pool(
#         replica_count=replica_count,
#         machine_type=machine_type,
#         accelerator_count=accelerator_count,
#         accelerator_type=accelerator_type,
#         boot_disk_type=boot_disk_type,
#         boot_disk_size_gb=boot_disk_size_gb,
#     ).pool_specs

#     for spec_order, spec in enumerate(worker_pool_specs):
#         if not spec:
#             continue

#         command = [
#             "sh",
#             "-c",
#             # f"python3 {script_path}",
#             f"python3 {script_path} {' '.join(args)}",
#         ]

#         spec["container_spec"] = {
#             "image_uri": container_uri,
#             "command": command,
#         }

#     job = aiplatform.CustomJob(
#         display_name=display_name,
#         worker_pool_specs=worker_pool_specs,
#         # base_output_dir=base_output_dir,
#         project=project,
#         location=location,
#         # credentials=credentials,
#         labels=labels,
#         # encryption_spec_key_name=encryption_spec_key_name,
#         staging_bucket=staging_bucket,
#     )

#     job.run(
#         # experiment=experiment,
#         # experiment_run=experiment_run,
#     )


# if __name__ == "__main__":
#     create_custom_job(
#         project="dsp-cell-annotation-service",
#         location="us-central1",
#         staging_bucket="gs://dsp-cell-annotation-service",
#         #  display_name="onepass324",
#         #  script_path="/gcs/dsp-cell-annotation-service/scvid/config_files/onepass_mean_var_std.py",
#         #  config_file="/gcs/dsp-cell-annotation-service/scvid/config_files/onepass_mean_var_std.yaml",
#         display_name="ppca324node9gpu4accum9",
#         script_path="/gcs/dsp-cell-annotation-service/scvid/config_files/probabilistic_pca.py",
#         config_file="/gcs/dsp-cell-annotation-service/scvid/config_files/probabilistic_pca.yaml",
#         #  display_name="node3gpu4accum1",
#         #  script_path="/gcs/dsp-cell-annotation-service/scvid/config_files/grad.py",
#         #  config_file="/gcs/dsp-cell-annotation-service/scvid/config_files/probabilistic_pca.yaml",
#         # container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest",
#         container_uri="us-central1-docker.pkg.dev/dsp-cell-annotation-service/scvid-docker-repo/scvid:lightning-cli",
#         # container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest",
#         # container_uri="pytorch/pytorch:latest",
#         replica_count=9,
#         accelerator_count=4,
#         #  labels: dict[str, str] | None = None,
#         #  experiment="onepass",
#         #  experiment_run: str | None = None,
#     )
