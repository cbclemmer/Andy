import tiktoken
import requests
import json

completion_config = {
    'temperature': 0,
    'max_tokens': 400,
    'stop': ['USER:', 'RAVEN:']
}

class GptCompletion:
    def __init__(self, api_key, org, model) -> None:
        self.api_key = api_key
        self.org = org
        self.model = model
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self._encoding = tiktoken.encoding_for_model(self.model)

    def complete(self, prompt, config={}):
        defaultConfig = {
            "model": self.model,
            "prompt": prompt
        }

        defaultConfig.update(config)
        res = requests.post('https://api.openai.com/v1/completions', 
            headers={
                'Authorization': 'Bearer ' + self.api_key,
                'OpenAI-Organization': self.org
            },
            json = defaultConfig
        ).json()
        if res.get('error') is not None:
            if 'overloaded' in res.get('error').get('message'):
                print('ERROR: Server overloaded. Retrying request...')
                self.complete(self, prompt, config)
            raise SyntaxError("error getting chat message: " + res.get('error').get('message'))
        # self.prompt_tokens += res.get('usage').get('prompt_tokens')
        # self.completion_tokens += res.get('usage').get('completion_tokens')
        # self.total_tokens += res.get('usage').get('total_tokens')
        print('TOKENS')
        print(res.get('usage').get('prompt_tokens'))
        print(res.get('usage').get('completion_tokens'))
        print(res.get('usage').get('total_tokens'))
        msg = res.get('choices')[0].get('text')
        return msg

class Chat:
    def __init__(
        self,
        api_key,
        org,
        model,
        config={ }
    ):
        self.api_key = api_key
        self.org = org
        self.model = model
        self.config = config
        self._encoding = tiktoken.encoding_for_model(self.model)

        # total tokens of the conversation updated every chat message
        self._total_tokens = 0
        self._messages = []

    def add_message(self, msg, role):
        self._messages.append({ 'role': role, 'content': msg})
    
    def send(self, msg, role="user"):
        if self.api_key is None or \
            self.org is None or \
            self.model is None:
            raise AssertionError("API key, org, or model is not defined")

        self.add_message(msg, role)
        return self.run()

    def run(self):
        cfg = {
            "model": self.model,
            "messages": self._messages
        }
        cfg.update(self.config)
        res = requests.post('https://api.openai.com/v1/chat/completions', 
            headers={
                'Authorization': 'Bearer ' + self.api_key,
                'OpenAI-Organization': self.org
            },
            json = cfg
        ).json()
        if res.get('error') is not None:
            if 'overloaded' in res.get('error').get('message'):
                print('ERROR: Server overloaded. Retrying request...')
                self.run(self)
            raise SyntaxError("error getting chat message: " + res.get('error').get('message'))
        self._total_tokens = res.get('usage').get('total_tokens')
        choice = res.get('choices')[0].get('message')
        self._messages.append(choice)
        return choice.get('content')

class EmbeddingFactory:
    def __init__(self, api_key, org_key):
        self.api_key = api_key
        self.org_key = org_key

    def get_embedding(self, data):
        d = {
            "model": 'text-embedding-ada-002',
            "input": json.dumps(data)
        }
        res = requests.post('https://api.openai.com/v1/embeddings', 
            headers={
                'Authorization': 'Bearer ' + self.api_key,
                'OpenAI-Organization': self.org_key
            },
            json=d
        ).json()
        if res.get('error') is not None:
            if 'overloaded' in res.get('error').get('message'):
                print('ERROR: Server overloaded. Retrying request...')
                self.get_embedding(self, data)
            raise SyntaxError("Error getting embedding: " + res.get('error').get('message'))
        return res.get('data')[0].get('embedding')