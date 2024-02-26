import re
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain import PromptTemplate
from langchain.chains import ConversationChain
from langchain.schema.output_parser import BaseLLMOutputParser


class MyOutputParser(BaseLLMOutputParser):
    def __init__(self):
        super().__init__()

    def parse_result(self, text):
        print("Entering Output Parser...")
        # print(text[0]['text'])
        text = text[0].text
        text = text.replace("Model: ", "Assistant: ")
        # if self.MERGE_CONSECUTIVE_MESSAGES:
        #    prompt = self._merge_consecutive_messages(prompt)

        # self.file_writer_response('DEBUG', text)

        PREFIX_MODEL = "Model:"
        PREFIX_USER = "Human:"
        PREFIX_SYSTEM = "System:"
        PREFIX_ASSISTANT = "Assistant:"
        PREFIX_INITIAL_RESPONSE = "Initial response:"
        PREFIX_APPLYING = "Applying"
        PREFIX_CRITIQUE = "Critique:"
        PREFIX_CRITIQUE_REQUEST = "Critique Request:"
        PREFIX_UPDATED_RESPONSE = "Updated response:"
        PREFIX_UPDATES_DONE = "Updated response: No revisions needed."
        PREFIX_NO_REVISIONS_NEEDED = "No revisions needed."

        # text = "Received below is a conversation between a human and an AI assistant."

        starter_pattern = r"^Received below is a conversation between"
        starter_match = re.search(starter_pattern, text)

        # Recognize Anthropic Claude Prefixes
        find_human_prompt = re.search(f"^{PREFIX_USER}", text)
        find_system_prompt = re.search(f"^{PREFIX_SYSTEM}", text)
        find_assistant_prompt = re.search(f"^{PREFIX_ASSISTANT}", text)

        # Find ConsitutionalChain Reponse Keys
        find_initial_response = re.search(f"^{PREFIX_INITIAL_RESPONSE}", text)
        find_applying = re.search(f"^{PREFIX_APPLYING} .+?\.{3}'", text)
        find_critique = re.search(f"^{PREFIX_CRITIQUE}", text)
        find_updated_response = re.search(f"^{PREFIX_UPDATED_RESPONSE}", text)
        find_updates_done = re.search(f"^{PREFIX_UPDATES_DONE}", text)
        find_no_revisions_needed = re.search(f"^{PREFIX_NO_REVISIONS_NEEDED}", text)

        # if find_critique:
        #    text = PREFIX_HUMAN + text

        if find_initial_response:
            text = PREFIX_ASSISTANT + text
        """
        if find_applying
        if find_critique
        if find_critique_request

        if find_updated_response:
        if find_updates_done:
        if find_no_revisions_needed:
            text = PREFIX_ASSISTANT + text

        else:
            response_type = 'error'
            #self.file_writer_response(response_type, text)
            raise OutputParserException(f"^Could not parse LLM output: `{text}`")
        """

        # self.file_writer_response(response_type, text)
        text = text.replace("Model:", "Assistant:")
        return text
