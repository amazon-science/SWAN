# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
bedrock_utils.py

AWS Bedrock API wrapper for LLM inference.

Provides BedrockManager, a unified interface that routes to the Anthropic
invoke_model API or the Amazon Nova converse API depending on the model ID.
Includes exponential-backoff retry logic for throttling and transient errors.
"""

import boto3
import json
import logging
from time import sleep

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class BedrockManager:
    def __init__(self, region: str, model_id: str):
        self._region = region
        self._model_id = model_id
        self._bedrock = boto3.client("bedrock-runtime", region_name=self._region)
        self._max_retries = 20
        self.base_sleep_time = 3
        self.throttling_sleep_time = 10

    def _invoke_with_retries(self, body: dict):
        attempt = 0
        response = None
        
        while attempt < self._max_retries:
            try:
                logger.debug(f"Attempt {attempt+1}/{self._max_retries}")
                response = self._bedrock.invoke_model(
                    body=json.dumps(body),
                    modelId=self._model_id,
                    accept="application/json",
                    contentType="application/json"
                )
                break
            except self._bedrock.exceptions.ThrottlingException as te:
                logger.warning(f"ThrottlingException: {te}")
                attempt += 1
                sleep(self.throttling_sleep_time * attempt)
            except self._bedrock.exceptions.ServiceUnavailableException as se:
                logger.warning(f"ServiceUnavailableException: {se}")
                attempt += 1
                sleep(self.base_sleep_time * attempt)
            except Exception as e:
                logger.warning(f"Unexpected error: {e}")
                attempt += 1
                sleep(self.base_sleep_time * attempt)
        
        if response is None:
            raise RuntimeError(f"Failed to invoke model after {self._max_retries} attempts.")
        return response
    

    def _converse_with_retries(self, messages: list, system: list = None):
        """
        Used for Amazon models that rely on the converse API (e.g. Nova).
        """
        attempt = 0
        response = None

        while attempt < self._max_retries:
            try:
                logger.debug(f"Attempt {attempt+1}/{self._max_retries} [converse]")
                response = self._bedrock.converse(
                    modelId=self._model_id,
                    messages=messages,
                    system=system
                )
                break
            except self._bedrock.exceptions.ThrottlingException as te:
                logger.warning(f"ThrottlingException: {te}")
                attempt += 1
                sleep(self.throttling_sleep_time * attempt)
            except self._bedrock.exceptions.ServiceUnavailableException as se:
                logger.warning(f"ServiceUnavailableException: {se}")
                attempt += 1
                sleep(self.base_sleep_time * attempt)
            except Exception as e:
                logger.warning(f"Unexpected error: {e}")
                attempt += 1
                sleep(self.base_sleep_time * attempt)

        if response is None:
            raise RuntimeError(f"Failed to converse with model after {self._max_retries} attempts.")
        return response
    
    def generate(self, 
                 user_text: str, 
                 system_text: str = None, 
                 max_tokens: int = 256, 
                 temperature: float = 0.7, 
                 top_p: float = 0.9) -> str:
        """
        Generate a completion using Bedrock.
        Routes to the Anthropic invoke_model API or the Amazon Nova converse API
        depending on the model ID.
        """
        is_nova_model = ("amazon.nova" in self._model_id or "us.amazon.nova" in self._model_id)

        if is_nova_model:
            # Amazon Nova approach: use 'converse'
            system_payload = None
            if system_text:
                system_payload = [{"text": system_text}]
            
            response = self._converse_with_retries(
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": user_text}]
                    }
                ],
                system=system_payload
            )

            content_list = response["output"]["message"]["content"]
            # Join if more than one text piece
            nova_text = " ".join(item["text"] for item in content_list).strip()
            return nova_text

        else:
            # Anthropic approach: use 'invoke_model'
            body = {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "messages": [
                    {"role": "user", "content": user_text}
                ],
                "anthropic_version": "bedrock-2023-05-31"
            }
            # Claude Sonnet 4.5 / Haiku 4.5 reject requests that set both
            # temperature and top_p. Keep temperature (the more common knob for
            # this codebase) and drop top_p for those models.
            if "claude-sonnet-4-5" in self._model_id or "claude-haiku-4-5" in self._model_id:
                body.pop("top_p", None)
            if system_text:
                body["system"] = system_text

            response = self._invoke_with_retries(body)
            response_body = json.loads(response["body"].read())
            
            content = response_body.get("content", [])
            if isinstance(content, list):
                # If content is a list of {"text": "..."} items
                text_segments = [item.get("text", "") for item in content]
                full_text = " ".join(text_segments).strip()
            else:
                # If content is a string or something else
                full_text = str(content).strip()

            return full_text


if __name__ == "__main__":
    manager = BedrockManager(
        region="us-east-1",
        model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    )
    test_prompt = "Describe a quiet morning in the countryside."
    result = manager.generate(user_text=test_prompt, max_tokens=128, temperature=0.7, top_p=0.9)
    print(f"Prompt: {test_prompt}\nResponse: {result}")
