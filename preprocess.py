import tqdm
import torch

from config import Config


try:

    title_word_embeddings = torch.load("title_embeddings.pt")
    title_word_embeddings = title_word_embeddings.to(device)
    print("Loaded title word embeddings")

except:

    def labelled_sentences(node_dict):
        sentences = []
        for node in tqdm(node_dict.keys()):
            sentences.append(TaggedDocument(node_dict[node].split(), [node]))

        return sentences

    sentences = labelled_sentences(title_dict)

    model_dbow = Doc2Vec(sentences, vector_size=VECTOR_SIZE, window=2, min_count=1, workers=8)
    model_dbow.build_vocab(sentences)

    for epoch in range(100):
        model_dbow.train(sentences, total_examples=model_dbow.corpus_count, epochs=model_dbow.epochs)
        model_dbow.alpha -= 0.002
        model_dbow.min_alpha = model_dbow.alpha


    def get_vectors(node_dict, model):
        vectors = []
        for node in tqdm(node_dict.keys()):
            vectors.append(model.dv[node])

        return vectors

    title_vectors = get_vectors(title_dict, model_dbow)

    # Save vectors

    title_word_embeddings = torch.tensor(title_vectors).to(device)
   
    torch.save(title_word_embeddings, "title_word_embeddings.pt")

    print("Saved title word embeddings")



