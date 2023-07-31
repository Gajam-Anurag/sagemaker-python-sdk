# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Contains the SageMaker Experiment utility methods."""
from __future__ import absolute_import

import logging
import os

import mimetypes
import urllib
from functools import wraps
from typing import Optional

import numpy
import PIL.Image
import matplotlib.figure
import plotly.graph_objects
import pandas

from sagemaker import Session
from sagemaker.apiutils import _utils
from sagemaker.experiments._environment import _RunEnvironment, _EnvironmentType
from sagemaker.experiments.trial_component import _TrialComponent
from sagemaker.utilities.search_expression import Filter, Operator, SearchExpression
from sagemaker.utils import retry_with_backoff


def resolve_artifact_name(file_path):
    """Resolve artifact name from given file path.

    If not specified, will auto create one.

    Args:
        file_path (str): Path to the file.

    Returns:
        str: The resolved artifact name.
    """
    _, filename = os.path.split(file_path)
    if filename:
        return filename

    return _utils.name("artifact")


def guess_media_type(file_path):
    """Infer the media type of a file based on its file name.

    Args:
        file_path (str): Path to the file.

    Returns:
        str: The guessed media type.
    """
    file_url = urllib.parse.urljoin("file:", urllib.request.pathname2url(file_path))
    guessed_media_type, _ = mimetypes.guess_type(file_url, strict=False)
    return guessed_media_type


def verify_length_of_true_and_predicted(true_labels, predicted_attrs, predicted_attrs_name):
    """Verify if lengths match between lists of true labels and predicted attributes.

    Args:
        true_labels (list or array): The list of the true labels.
        predicted_attrs (list or array): The list of the predicted labels/probabilities/scores.
        predicted_attrs_name (str): The name of the predicted attributes.

    Raises:
        ValueError: If lengths mismatch between true labels and predicted attributes.
    """
    if len(true_labels) != len(predicted_attrs):
        raise ValueError(
            "Lengths mismatch between true labels and {}: "
            "({} vs {}).".format(predicted_attrs_name, len(true_labels), len(predicted_attrs))
        )


def validate_invoked_inside_run_context(func):
    """A Decorator to force the decorated method called under Run context."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        self_instance = args[0]
        if not self_instance._inside_load_context and not self_instance._inside_init_context:
            raise RuntimeError("This method should be called inside context of 'with' statement.")
        return func(*args, **kwargs)

    return wrapper


def is_already_exist_error(error):
    """Check if the error indicates resource already exists

    Args:
        error (dict): The "Error" field in the response of the
            `botocore.exceptions.ClientError`
    """
    return error["Code"] == "ValidationException" and "already exists" in error["Message"]


def get_tc_and_exp_config_from_job_env(
        environment: _RunEnvironment,
        sagemaker_session: Session,
) -> dict:
    """Retrieve an experiment config from the job environment.

    Args:
        environment (_RunEnvironment): The run environment object with job specific data.
        sagemaker_session (sagemaker.session.Session): Session object which
            manages interactions with Amazon SageMaker APIs and any other
            AWS services needed. If not specified, one is created using the
            default AWS configuration chain.
    """
    job_name = environment.source_arn.split("/")[-1]
    if environment.environment_type == _EnvironmentType.SageMakerTrainingJob:
        job_response = retry_with_backoff(
            callable_func=lambda: sagemaker_session.describe_training_job(job_name),
            num_attempts=4,
        )
    elif environment.environment_type == _EnvironmentType.SageMakerProcessingJob:
        job_response = retry_with_backoff(
            callable_func=lambda: sagemaker_session.describe_processing_job(job_name),
            num_attempts=4,
        )
    else:  # environment.environment_type == _EnvironmentType.SageMakerTransformJob
        job_response = retry_with_backoff(
            callable_func=lambda: sagemaker_session.describe_transform_job(job_name),
            num_attempts=4,
        )

    job_exp_config = job_response.get("ExperimentConfig", dict())
    from sagemaker.experiments.run import RUN_NAME

    if job_exp_config.get(RUN_NAME, None):
        return job_exp_config
    raise RuntimeError(
        "Not able to fetch RunName in ExperimentConfig of the sagemaker job. "
        "Please make sure the ExperimentConfig is correctly set."
    )


def verify_load_input_names(
        run_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
):
    """Verify the run_name and the experiment_name inputs in load_run.

    Args:
        run_name (str): The run_name supplied by the user (default: None).
        experiment_name (str): The experiment_name supplied by the user
            (default: None).

    Raises:
        ValueError: If run_name is supplied while experiment_name is not.
    """
    if not run_name and experiment_name:
        logging.warning(
            "No run_name is supplied. Ignoring the provided experiment_name "
            "since it only takes effect along with run_name. "
            "Will load the Run object from the job environment or current Run context."
        )
    if run_name and not experiment_name:
        raise ValueError(
            "Invalid input: experiment_name is missing when run_name is supplied. "
            "Please supply a valid experiment_name when the run_name is not None."
        )


def is_run_trial_component(trial_component_name: str, sagemaker_session: Session) -> bool:
    """Check if a trial component is generated by `sagemaker.experiments.Run`

    Args:
        trial_component_name (str): The name of the trial component.
        sagemaker_session (sagemaker.session.Session): Session object which
            manages interactions with Amazon SageMaker APIs and any other
            AWS services needed. If not specified, one is created using the
            default AWS configuration chain.

    Returns:
        bool: Indicate whether the trial component is created by
            `sagemaker.experiments.Run` or not.
    """
    search_filter = Filter(
        name="TrialComponentName",
        operator=Operator.EQUALS,
        value=trial_component_name,
    )
    search_expression = SearchExpression(filters=[search_filter])

    def search():
        return list(
            _TrialComponent.search(
                search_expression=search_expression,
                max_results=1,  # TrialComponentName is unique in an account
                sagemaker_session=sagemaker_session,
            )
        )[0]

    try:
        tc_search_res = retry_with_backoff(search, 4)
        from sagemaker.experiments.run import RUN_TC_TAG

        if not tc_search_res.tags or RUN_TC_TAG not in tc_search_res.tags:
            return False
        return True
    except Exception as ex:  # pylint: disable=broad-except
        logging.warning(
            "Failed to inspect the type of the trial component (%s), due to (%s)",
            trial_component_name,
            str(ex),
        )
        return False


def verify_type_and_form_of_image(image):
    """Verify object is ndarray or PIL.Image.Image and ndarray has valid dimension and channel

    Args:
        image (numpy.ndarray or PIL.Image.Image): An Image object of type ndarray or PIL

    Raises:
        TypeError: If Image object is not in the supported object format.
        ValueError: If ndarray object is not in the given dimension or channel.
    """

    if not isinstance(image, (numpy.ndarray, PIL.Image.Image)):
        raise TypeError(
            "Logged Image must be of numpy.ndarray or PIL.Image.Image but found type: {}."
            .format(type(image))
        )

    if isinstance(image, numpy.ndarray):
        if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
            raise ValueError(
                "Invalid channel length: {}, Image channel has to be either Grayscale(1) "
                "or RGB(3) or RGBA(4)".format(image.shape[2])
            )
        if len(image.shape) < 2 or len(image.shape) > 3:
            raise ValueError(
                "Invalid dimension: {}, Image has to be either 2D or 3D ndarray".format(image.ndim)
            )


def verify_type_of_figure_and_name(figure, name):
    """Verify object is matplotlib or plotly Figure and verify file extension in supported formats

    Args:
        figure(matplotlib.figure.Figure, plotly.graph_objects.Figure): the figure object of type
        matplotlib.figure.Figure, plotly.graph_objects.Figure
        name (str): the name of the artifact file

    Raises:
        TypeError: If figure is not in the supported object format or
            if the artifact name not listed in the supported Image formats
    """

    if not isinstance(figure, (matplotlib.figure.Figure, plotly.graph_objects.Figure)):
        raise TypeError(
            "Logged figure must be of matplotlib.figure.Figure or plotly.graph_objects.Figure "
            "but found type: {}.".format(type(figure))
        )

    if name:
        extension = os.path.splitext(name)[1]

        if isinstance(figure, matplotlib.figure.Figure):
            supported_formats = ['.eps', '.jpeg', '.jpg', '.pdf', '.pgf', '.png', '.ps',
                                 '.raw', '.rgba', '.svg', '.svgz', '.tif', '.tiff', '.webp']
        elif isinstance(figure, plotly.graph_objects.Figure):
            supported_formats = ['.png', '.jpg', '.jpeg', '.webp', '.svg', '.pdf', '.html']

        if extension not in supported_formats:
            raise TypeError(
                "Format '{}' is not supported for {} (Supported formats: {})".
                format(extension[1:], type(figure), supported_formats)
            )


def verify_type_of_text_and_name(text, name):
    """Verify the text is string and verify artifact file extension in supported formats

    Args:
        text (str): the text artifact in the form of string
        name (str): the name of the artifact file

    Raises:
        TypeError: If text is not in the supported object format or
            if the artifact name not listed in the supported text formats
    """

    if not isinstance(text, str):
        raise TypeError(
            "Text artifact must be of string but found type: {}. ".format(type(text))
        )

    if name:
        extension = os.path.splitext(name)[1]

        supported_formats = ['.txt', '.csv', '.htm', '.html', '.ini',
                             '.log', '.md', '.rtf', '.yaml', '.yml']

        if extension not in supported_formats:
            raise TypeError(
                "Format '{}' is not supported (Supported formats: {})"
                .format(extension[1:], supported_formats)
            )


def verify_type_of_table_and_name(table, name):
    """Verify object is dict or DataFrame and verify file extension in supported formats

    Args:
        table (dict or pandas.DataFrame): the table artifact of type dict or pandas.DataFrame
        name (str): the name of the artifact file

    Raises:
        TypeError: If table object is not in the supported object format or
            if the artifact name not listed in the supported json format
    """

    if not isinstance(table, (dict, pandas.DataFrame)):
        raise TypeError(
            "table artifact must be of dictionary or pandas.DataFrame but found type: {}"
            .format(type(table))
        )

    if name:
        extension = os.path.splitext(name)[1]

        if extension != '.json':
            raise TypeError(
                "Format '{}' is not supported (Supported formats: [.json])".format(extension[1:])
            )


def verify_type_of_dictionary_and_name(dictionary, name):
    """Verify object is dictionary and verify the artifact file extension in supported formats

    Args:
        dictionary (dict): the artifact in the form of dictionary.
        name (str): The name of the artifact file

    Raises:
        TypeError: If dictionary is not in the supported object format or
            if the artifact name not listed in the supported formats
    """
    if not isinstance(dictionary, dict):
        raise TypeError(
            "The artifact must be of dictionary but found type: {}.".format(type(dictionary))
        )

    if name:
        extension = os.path.splitext(name)[1]

        supported_formats = ['.json', '.yml', '.yaml']

        if extension not in supported_formats:
            raise TypeError(
                "Format '{}' is not supported (Supported formats: {})"
                .format(extension[1:], supported_formats)
            )
