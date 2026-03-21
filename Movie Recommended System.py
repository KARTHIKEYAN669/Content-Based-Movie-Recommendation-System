import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

data={
    "Movie":[
        "Avengers",
        "Iron Man",
        "Captain America",
        "Thor",
        "Spider Man",
        "Batman",
        "Superman"
    ],
    "Genre":[
        "Action Superhero",
        "Action Superhero",
        "Action Superhero",
        "Action Fantasy",
        "Action Superhero",
        "Action Dark",
        "Action Superhero"
    ]
}

df=pd.DataFrame(data)
print(df)

vectorizer=CountVectorizer()
genre_matrix=vectorizer.fit_transform(df["Genre"])

similarity=cosine_similarity(genre_matrix)
df["Movie"]=df["Movie"].str.lower()
def recommend(movie_name):
    movie_name=movie_name.lower()
    
    if movie_name not in df["Movie"].values:
        print("Movie not found")
        return
    
    movie_index=df[df["Movie"]==movie_name].index[0]
    similarity_scores=list(enumerate(similarity[movie_index]))
    similarity_scores=sorted(similarity_scores,key=lambda x:x[1],reverse=True)
    print(f"Recommended movies for {movie_name}:")

    for i in similarity_scores[1:4]:
        print(df.iloc[i[0]]["Movie"])

recommend("Avengers")