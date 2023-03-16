from gpt import GptCompletion, EmbeddingFactory, completion_config
from util import open_file, get_closest_embeddings

class Concept(GptCompletion):
    def __init__(self, api_key, org_key):
        super().__init__(api_key, org_key, 'text-davinci-003')
        self.embeddingFactory = EmbeddingFactory(api_key, org_key)
        self._retrieve_prompt = open_file('prompts/prompt_concept_retrieve.txt')
        self._update_prompt = open_file('prompts/prompt_concept_update.txt')

    def retrieve_concepts(self, salient_points):
        prompt = self._retrieve_prompt.replace('<<INPUT>>', salient_points)
        concept_keys = self.complete(prompt, completion_config).split('\n')
        concepts = ''
        concept_list = []
        for c in concept_keys:
            print('Attempting to retrieve concept: ' + c)
            q_embed = self.embeddingFactory.get_embedding(c)
            most_similar = get_closest_embeddings('concepts', q_embed, 1, concept_list, 'data')
            if len(most_similar) == 0:
                continue
            concept_list.append(most_similar[0][1]['data'])
            concepts += '\n- ' + most_similar[0][1]['data']
        return concepts
    
    def update_concepts(self, salient_points, retrieved_concepts, bot_message, user_message):
        prompt = self._update_prompt\
            .replace('<<SALIENT_POINTS>>', salient_points)\
            .replace('<<CONCEPTS>>', retrieved_concepts)\
            .replace('<<CHAT_MESSAGE>>', bot_message)\
            .replace('<<USER_MESSAGE>>', user_message)
        concepts_to_update = self.complete(prompt, completion_config).split('\n')
        add_concepts = []
        update_concepts = []
        for c in concepts_to_update:
            c_embed_o = self.embed_concept(c)
            if 'add' in c.lower():
                c_embed_o['data'] = c_embed_o['data'].replace('Add', '')
                add_concepts.append(c_embed_o)
            if 'update' in c.lower():
                c_embed_o['data'] = c_embed_o['data'].replace('Update', '')
                c_to_update = get_closest_embeddings('concepts', c_embed_o['embedding'], 1)
                if len(c_to_update) == 0:
                    continue
                c_embed_o['file_name'] = c_to_update[0][1]['file_name']
                update_concepts.append(c_embed_o)
        return {
            'add': add_concepts,
            'update': update_concepts
        }

    def embed_concept(self, concept):
        c_embed = self.embeddingFactory.get_embedding(concept)
        return {
            'data': concept,
            'embedding': c_embed
        }