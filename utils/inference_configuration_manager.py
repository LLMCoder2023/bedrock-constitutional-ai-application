import json


class InferenceConfigurationManager:

    def configuration_selector(model_id):
        configuration = None

        with open("./utils/llm_configurations.json") as json_file:
            configuration_data = json.load(json_file)
            configuration = json.dumps(configuration_data[f"{model_id}"])
        return configuration
