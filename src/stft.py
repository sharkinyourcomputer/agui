"Shut the fuck transform. Rasta don't work for no CIA."

import numpy as np
from numpy.fft import rfft, irfft
import scipy.io.wavfile


def ReadAndConvert(wavfile):
  rate, iwav = scipy.io.wavfile.read(wavfile)
  assert len(iwav.shape) == 1, iwav.shape
  assert iwav.dtype == np.int16
  # It doesn't seem to matter what type we put into fft, complex128 comes out.
  return np.array(iwav, dtype=np.float32), rate

  
def ConvertAndWrite(wavform, rate, wavfile):
  maxval = np.max(np.abs(wavform))
  assert maxval > 0, (maxval, wavform.size)
  normalized_wavform = wavform / maxval
  int_data = np.array((2 ** 15) * normalized_wavform, dtype=np.int16)
  scipy.io.wavfile.write(wavfile, rate, int_data)


def STFT(x, nfft=1024, nskip=256):
  "I'm gonna put it on. Anywhere, anytime. I'm not boasting. I'm just toasting."
  h = Sinc(nfft)
  return [
    rfft(h * Zpad(nfft, x[w_0 : min(x.size, w_0+nfft)]))
    for w_0 in xrange(0, x.size, nskip)]


def STIFT(stft_windows, nfft, nskip):
  "Sum contributions from each window. No normalization."
  r = np.zeros(nskip * (len(stft_windows) - 1) + nfft)
  for w, w_0 in zip(stft_windows, xrange(0, r.size, nskip)):
    r[w_0 : w_0+nfft] += irfft(w, nfft)
  return r


def Sinc(N, k=3.):
  "Centered samples of sinc(x). k is max abs(x)"
  assert not N % 2, N
  x = k * np.arange(-N/2., N/2, 1, dtype=np.float32) / (N/2.)
  h = np.sin(np.pi * x) / x  # Warning: invalid value encountered in divide
  h[x==0] = 1.0
  return h


def Zpad(N, x):
  # Use np.pad ...
  return (x if x.size == N else
    np.hstack((x, np.zeros(N-x.size))))


def test():
  import os
  s, f = ReadAndConvert(os.path.join('testdata', 'foo.wav'))
  nfft, nskip = 1024, 256
  stft_windows = STFT(s, nfft, nskip)
  r = np.hstack((
    STIFT(stft_windows, nfft, nskip),
    STIFT(stft_windows, nfft/2, nskip/2),
    STIFT(stft_windows, nfft/4, nskip/4),
  ))
  ConvertAndWrite(r, f, os.path.join('testdata', 'out.wav'))


if __name__ == '__main__':
  test()
