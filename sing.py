
import os
from time import sleep
from src import parse_data

song_lyrics = parse_data.ReadAndCanonicalizeLyricsData('./data')

unigram_counts, bigram_counts = parse_data.CountGrams(song_lyrics)


words = parse_data.BayesianWords(unigram_counts, bigram_counts, 1000)

for n in [0, 40, 80, 120]:
  parse_data.Speak(words[n+0:n+10], s=200, p=25)
  parse_data.Speak(words[n+10:n+20], s=200, p=50)
  sleep(1.0)
  parse_data.Speak(words[n+20:n+30], s=100, p=75)
  parse_data.Speak(words[n+30:n+40], s=300, p=50)
  sleep(1.5)
