import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import numpy as np
import csv

# Lendo Dados
dados = pd.read_csv('https://raw.githubusercontent.com/Cizika/tiktok-popularity-analysis/main/tiktok.csv')

# Removendo dados duplicados
dados_new = dados.drop_duplicates('track_id')

artists = np.unique(dados_new["artist_id"].to_numpy()).tolist()
credentials = SpotifyClientCredentials(client_id="f385d6971d5b4d0baa6c02d700251a8f", client_secret="4695782d5b30448787175babfcdf6294")
spotify = spotipy.Spotify(client_credentials_manager=credentials)

j = 0
for i in range(50, 2101, 50):
    results = spotify.artists(artists[j:i])

    j = i

    for artist in results["artists"]:
        with open("artist_data.csv", 'a', newline='') as artist_csv:
            artist_csv_append = csv.writer(artist_csv)
            artist_csv_append.writerow([artist["id"], artist["popularity"]])
