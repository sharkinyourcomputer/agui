from os import system
from os import path
from glob import glob
import numpy as np
from scipy.io import wavfile


def ReadAndCanonicalizeLyricsData(data_path):
  lyrics_files = glob(path.join(data_path, '*.txt'))
  song_lyrics = {}
  for filename in lyrics_files:
    print 'Reading %s' % filename
    with open(filename) as f:
      lines = f.read().split('\n')
      assert lines[0].startswith('TITLE:')
      title = Canonicalize(lines[0][len('TITLE:'):])
      song_lyrics[title] = filter(None, map(Canonicalize, lines[1:]))
      print 'Title: %s\n# Lines: %d\n' % (title, len(song_lyrics[title]))
  return song_lyrics

def Canonicalize(line):
  return line.lower().translate(None, ',.[]-;?!()\'"').strip()


def CountGrams(song_lyrics):
  unigram_counts = {}
  bigram_counts = {}
  for title, lines in song_lyrics.iteritems():
    _CountGrams([x.strip() for x in title.split(' ')],
        unigram_counts, bigram_counts)
    for line in lines:
      _CountGrams([x.strip() for x in line.split(' ')],
          unigram_counts, bigram_counts)
  return unigram_counts, bigram_counts

def _CountGrams(unigrams, unigram_counts, bigram_counts):
  for ug in unigrams:
    unigram_counts[ug] = unigram_counts.get(ug, 0.0) + 1.0
  for w1, w2 in zip(unigrams[:-1], unigrams[1:]):
    w1_bgcnts = bigram_counts.setdefault(w1, {})
    w1_bgcnts[w2] = w1_bgcnts.get(w2, 0.0) + 1.0


def BayesianWords(unigram_counts, bigram_counts, n_words):
  unigrams, ucounts = zip(*sorted(filter(
      lambda (k, v): k in bigram_counts,
      unigram_counts.items())))
  prior = np.array(ucounts) / sum(ucounts)
  prior_pdf = np.array([np.sum(prior[:n]) for n in range(len(unigrams))])

  bigram_pdfs = {}
  for w1, w1_bgcnts in bigram_counts.iteritems():
    w2strs, w2counts = zip(*sorted(w1_bgcnts.items()))
    w2pdf = np.array(w2counts) / sum(w2counts)
    bigram_pdfs[w1] = (
        w2strs,
        np.array([np.sum(w2pdf[:n]) for n in range(len(w2strs))]))
    #print '%d bigrams for %s' % (len(w2strs), w1)

  first_word_index = np.searchsorted(prior_pdf, np.random.random_sample())
  words = [unigrams[min(len(unigrams)-1, first_word_index)]]
  for n in range(1, n_words):
    if words[-1] in bigram_pdfs:
      bigram_strs, bigram_pdf = bigram_pdfs[words[-1]]
      idx = np.searchsorted(bigram_pdf, np.random.random_sample())
      words.append(bigram_strs[min(len(bigram_strs)-1, idx)])
    else:
      # Pick from the prior.
      idx = np.searchsorted(prior_pdf, np.random.random_sample())
      words.append(unigrams[min(len(unigrams)-1, idx)])
  return words


def Speak(words, wavfile, speed=80, pitch=50):
  system('bin\espeak -w %s -v +m4 -s %d -p %d "%s"' % (
      wavfile, speed, pitch, ' '.join(words)))


def GetRawWave(wavfile):
  w = wave.open(wavfile, 'rb')
  nchannels, sampwidth, framerate, nframes, comptype, _ = w.getparams()
  assert (
      (nchannels, sampwidth, framerate, comptype)
      == (1, 2, 22050, 'NONE'))
  frames = w.readframes(nframes)
  w.close()
  return frames
