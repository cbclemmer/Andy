from gpt import GptCompletion, completion_config
from util import open_file, stringify_conversation

class Salience(GptCompletion):
    def __init__(self, api_key, org_key):
        super().__init__(api_key, org_key, 'text-davinci-003')
        self._salience_prompt = open_file('prompts/prompt_salience.txt')
        self._prompt_tokens = len(self._encoding.encode(self._salience_prompt))

    def get_salient_points(self, conversation):
        conv_s = stringify_conversation(conversation)
        prompt = self._salience_prompt.replace('<<INPUT>>', conv_s)
        return self.complete(prompt, completion_config)