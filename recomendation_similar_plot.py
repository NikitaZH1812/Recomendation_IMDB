import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

metadata = pd.read_csv("movies_metadata.csv", low_memory=False)

tfidf = TfidfVectorizer(stop_words='english')

metadata['overview'] = metadata['overview'].fillna('')

tfidf_matrix = tfidf.fit_transform(metadata['overview'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()


def get_recommendations(title, cosine_sim=cosine_sim):

    movie_indices = indices[title]

    similarity_score = list(enumerate(cosine_sim[movie_indices]))

    similarity_score = sorted(similarity_score,
                              key=lambda x: x[1], reverse=True)

    similarity_score = similarity_score[1:11]

    movie_indices = [i[0] for i in similarity_score]

    return metadata['title'].iloc[movie_indices]


result = open("result.txt", "w")
result.write(str(get_recommendations("The Maze Runner")))
result.close()
