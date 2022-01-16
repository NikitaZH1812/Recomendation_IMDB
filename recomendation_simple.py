import pandas as pd

metadata = pd.read_csv("movies_metadata.csv", low_memory=False)

mean_voite_whole_dataset = metadata["vote_average"].mean()

minimum_voites_required = metadata['vote_count'].quantile(0.90)

filtered_movies = metadata.copy().loc[metadata['vote_count'] >= minimum_voites_required]


def weighted_rating(x, minimum_voites_required=minimum_voites_required,
                    mean_voite_whole_dataset=mean_voite_whole_dataset):
    votes_count = x['vote_count']
    votes_avarage = x['vote_average']
    return (votes_count/(votes_count+minimum_voites_required) * votes_avarage) + (minimum_voites_required/(minimum_voites_required+votes_count) * mean_voite_whole_dataset)


filtered_movies['score'] = filtered_movies.apply(weighted_rating, axis=1)

filtered_movies = filtered_movies.sort_values('score', ascending=False)

result = open("result.txt", "w")
result.write(str(filtered_movies[['title', 'vote_count',
                                  'vote_average', 'score']].head(10)))
result.close()
