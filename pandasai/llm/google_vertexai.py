"""Google VertexAI

This module is to run the Google VertexAI LLM.
To read more on VertexAI:
https://cloud.google.com/vertex-ai/docs/generative-ai/learn/generative-ai-studio.

Example:
    Use below example to call Google VertexAI

    >>> from pandasai.llm import GoogleVertexAI

"""
from typing import Optional

from pandasai.helpers.memory import Memory

from ..exceptions import UnsupportedModelError
from ..helpers.optional import import_dependency
from .base import BaseGoogle


class GoogleVertexAI(BaseGoogle):
    """Google Vertexai LLM
    BaseGoogle class is extended for Google Vertexai model.
    The default model support at the moment is gemini-flash-latest.
    However, user can choose to use any other model from the list.
    """

    _supported_generative_models = [
        "gemini-flash-latest",
        "gemini-flash-lite-latest",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-pro",
        "gemini-3-flash-preview",
        "gemini-3-pro-preview",
    ]

    def __init__(
        self, project_id: str, location: str, model: Optional[str] = None, **kwargs
    ):
        """
        A init class to implement the Google Vertexai Models

        Args:
            project_id (str): GCP project
            location (str): GCP project Location
            model Optional (str): Model to use Default to gemini-flash-latest
            **kwargs: Arguments to control the Model Parameters
        """

        self.model = model or "gemini-flash-latest"

        self._configure(project_id, location)
        self.project_id = project_id
        self.location = location
        self._set_params(**kwargs)

    def _configure(self, project_id: str, location: str):
        """
        Configure Google VertexAi. Set value `self.vertexai` attribute.

        Args:
            project_id (str): GCP Project.
            location (str): Location of Project.

        Returns:
            None.

        """

        err_msg = "Install google-genai >= 1.0 for Google Vertexai"
        genai_module = import_dependency("google.genai", extra=err_msg)
        self.client = genai_module.Client(
            vertexai=True, project=project_id, location=location
        )

    def _valid_params(self):
        """Returns if the Parameters are valid or Not"""
        return super()._valid_params() + ["model"]

    def _validate(self):
        """
        A method to Validate the Model

        """

        super()._validate()

        if not self.model:
            raise ValueError("model is required.")

    def _generate_text(self, prompt: str, memory: Optional[Memory] = None) -> str:
        """
        Generates text for prompt.

        Args:
            prompt (str): A string representation of the prompt.

        Returns:
            str: LLM response.

        """
        self._validate()

        updated_prompt = self.prepend_system_prompt(prompt, memory)

        self.last_prompt = updated_prompt

        if self.model in self._supported_generative_models:
            completion = self.client.models.generate_content(
                model=self.model,
                contents=updated_prompt,
                config={
                    "max_output_tokens": self.max_output_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                },
            )
            return completion.text

        elif self.model in self._supported_code_chat_models:
            # Note: For simplicity and consistency with the new SDK,
            # we use generate_content even for chat models as it's the standard.
            # If specific 'chat' features are needed, they can be added later.
            completion = self.client.models.generate_content(
                model=self.model,
                contents=updated_prompt,
                config={
                    "max_output_tokens": self.max_output_tokens,
                    "temperature": self.temperature,
                },
            )
            return completion.text

        else:
            raise UnsupportedModelError(self.model)

    @property
    def type(self) -> str:
        return "google-vertexai"
