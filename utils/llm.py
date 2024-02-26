import boto3
from botocore.config import Config

import json
import os
import logging
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock


class LLM:

    titan_inference_configuration = {
        "temperature": 0.1,
        "topP": 1,
        "maxTokenCount": 4096,
    }

    claude_inference_configuration = {
        "temperature": 0.1,
        "top_p": 0.999,
        "top_k": 250,
        "max_tokens_to_sample": 200,
    }

    current_working_directory = os.getcwd()

    bedrock_embedding_model_id = "amazon.titan-embed-text-v1"

    def call_llm_llama(self, payload, bedrock_model_id):
        print(f"Model Id: {bedrock_model_id}")
        print("---calling llm")

        boto3_bedrock = self.setup_bedrock_runtime()

        body = json.dumps(payload)
        accept = "application/json"
        contentType = "application/json"

        try:
            response = boto3_bedrock.invoke_model(
                body=body,
                modelId=bedrock_model_id,
                accept=accept,
                contentType=contentType,
            )
            response_body = json.loads(response.get("body").read())
        except Exception as e:
            print(e)

        return response_body

    def call_llm(self, prompt, inference_configuration, bedrock_model_id):

        print(f"Model Id: {bedrock_model_id}")
        print("---calling llm")
        # print(inference_configuration)

        # inference_configuration["stop_sequences"][0] = inference_configuration["stop_sequences"][0].encode("utf-8").decode("unicode_escape")

        boto3_bedrock = self.setup_bedrock_runtime()

        inference_configuration["prompt"] = prompt

        body = json.dumps(inference_configuration)
        accept = "application/json"
        contentType = "application/json"

        try:
            response = boto3_bedrock.invoke_model(
                body=body,
                modelId=bedrock_model_id,
                accept=accept,
                contentType=contentType,
            )
            response_body = json.loads(response.get("body").read())["completion"]
            response_body = response_body.replace("<summary>", "")
            response_body = response_body.replace("</summary>", "")
            return response_body
        except Exception as e:
            print(e)

    def list_foundation_models(self):
        boto3_bedrock = self.setup_bedrock_runtime()
        models_list = boto3_bedrock.list_foundation_models()

        return models_list

    def call_titan_embeddings(self, content_to_embed):
        boto3_bedrock = self.setup_bedrock_runtime()

        embedder = BedrockEmbeddings(
            client=boto3_bedrock, model_id=self.bedrock_embedding_model_id
        )

        embeddings = embedder.embed_query(content_to_embed)

        return embeddings

    def setup_bedrock_runtime(self):
        session = boto3.Session()
        config = Config(read_timeout=2000)
        # use default public bedrock service endpoint url
        bedrock = session.client(
            service_name="bedrock-runtime",
            region_name="us-west-2",
            config=config,
            # endpoint_url='https://prod.us-west-2.dataplane.bedrock.aws.dev'
        )
        return bedrock

    def setup_langchain_bedrock_claude_v2_1(
        self, bedrock_model_id, inference_configuration
    ):

        langchain_bedrock_claude = Bedrock(
            model_id=bedrock_model_id,
            client=self.setup_bedrock_runtime(),
            verbose=True,
            model_kwargs=inference_configuration,
            cache=False,
        )
        return langchain_bedrock_claude

    def setup_langchain_bedrock_llama(self, bedrock_model_id, inference_configuration):

        langchain_bedrock_llama = Bedrock(
            model_id=bedrock_model_id,
            client=self.setup_bedrock_runtime(),
            verbose=True,
            model_kwargs=inference_configuration,
            cache=False,
        )

        return langchain_bedrock_llama

    def setup_langchain_bedrock_claude_instant(
        self, bedrock_model_id, inference_configuration
    ):
        print(inference_configuration)

        langchain_bedrock_claude_instant = Bedrock(
            model_id=bedrock_model_id,
            client=self.setup_bedrock_runtime(),
            verbose=True,
            model_kwargs=inference_configuration,
            cache=False,
        )

        return langchain_bedrock_claude_instant

    def setup_bedrock_service(self):
        session = boto3.Session()

        # use default public bedrock service endpoint url
        bedrock = session.client(service_name="bedrock", region_name="us-west-2")
        return bedrock
