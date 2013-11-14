
import os
from time import sleep
from src import parse_data

song_lyrics = parse_data.ReadAndCanonicalizeLyricsData('./data')

unigram_counts, bigram_counts = parse_data.CountGrams(song_lyrics)


words = parse_data.BayesianWords(unigram_counts, bigram_counts, 1000)

s=200
parse_data.Speak(words[0:10], s=200, p=25)
parse_data.Speak(words[10:20], s=200, p=50)
sleep(1.0)
parse_data.Speak(words[20:30], s=100, p=75)
parse_data.Speak(words[30:40], s=100, p=50)
sleep(1.0)
