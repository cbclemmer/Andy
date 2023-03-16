from gpt import EmbeddingFactory
from util import get_closest_embeddings

class Memory(EmbeddingFactory):
    def __init__(self, api_key, org_key):
        super().__init__(api_key, org_key)

    def get_memories(self, query, top_n=3):
        q_embed = self.get_embedding(query)
        most_similar = get_closest_embeddings('embeddings', q_embed, top_n)
        memories = ''
        for o in most_similar:
            memories += '\n- ' + o[1]['salient_points']
        return memories