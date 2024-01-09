from sentence_transformers import SentenceTransformer, util


class SimilarityMetric:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def calculate(self, original, altered):
        original_embeddings = self.model.encode(original)
        altered_embeddings = self.model.encode(altered)

        cosine_similarities = util.cos_sim(original_embeddings, altered_embeddings)

        return cosine_similarities

    def __call__(self, original, altered):
        return self.calculate(original, altered)

