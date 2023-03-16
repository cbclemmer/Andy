from gpt import GptCompletion, completion_config
from util import open_file, stringify_conversation

class Summary(GptCompletion):
    def __init__(self, api_key, org_key):
        super().__init__(api_key, org_key, 'text-davinci-003')
        self._summary_prompt = open_file('prompts/prompt_executive_summary.txt')
        self.prompt_tokens = self._encoding.encode(self._summary_prompt)

    def summarize(self, conversation):
        conv_s = stringify_conversation(conversation)
        prompt = self._summary_prompt.replace('<<INPUT>>', conv_s)
        return self.complete(prompt, completion_config)