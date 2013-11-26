"Shut the fuck transform. Rasta don't work for no CIA."
import os
from functools import partial
import numpy as np
import scipy.io.wavfile
import scipy.signal
from scipy.signal import hamming
from numpy.fft import rfftfreq, rfft, irfft, fftshift
import plot


def ReadAndConvert(wavfile):
  rate, iwav = scipy.io.wavfile.read(wavfile)
  assert len(iwav.shape) == 1, iwav.shape
  assert iwav.dtype == np.int16
  # It doesn't seem to matter what type we put into fft, complex128 comes out.
  return np.array(iwav, dtype=np.float32) / (2.0 ** 16), rate

  
def ConvertAndWrite(wavform, rate, wavfile):
  maxval = np.max(np.abs(wavform))
  assert maxval > 0, (maxval, wavform.size)
  normalized_wavform = wavform / maxval
  int_data = np.array((2 ** 15) * normalized_wavform, dtype=np.int16)
  scipy.io.wavfile.write(wavfile, rate, int_data)


def STFT(s, nfft=1024, nskip=512):
  "I'm gonna put it on. Anywhere, anytime. I'm not boasting. I'm just toasting."
  h = sinc(nfft)
  return [
      rfft(h * zpad(nfft, s[w_0 : min(s.size, w_0+nfft)]))
      for w_0 in xrange(0, s.size, nskip)]


def STIFT(stft_windows, nfft, nskip):
  "Sum contributions from each window. No normalization."
  r = np.zeros(nskip * (len(stft_windows) - 1) + nfft)
  h = hamming(nfft)
  for w, w_0 in zip(stft_windows, xrange(0, r.size, nskip)):
    r[w_0 : w_0+nfft] += irfft(w, nfft) / h
  return r


def sinc(N, k=3.):
  "Centered samples of sinc(x). k is max abs(x)"
  assert not N % 2, N
  x = k * np.arange(-N/2., N/2, 1, dtype=np.float32) / (N/2.)
  zero = np.argwhere(x == 0)
  x[zero] = 1.0 # Avoids "Warning: invalid value encountered in divide"
  h = np.sin(np.pi * x) / x
  h[zero] = 1.0
  return h


def zpad(N, x):
  # Use np.pad ...
  return (x if x.size == N else
    np.hstack ((x, np.zeros(N-x.size))))


def EstimatePitch(S, rate):
  "Very simple. Maximum value of abs(spectrum)."
  w = np.argmax(np.abs(S))
  r = rfftfreq(S.size, 1.0/rate)[w]
  return r


def Resample(S, ns, nr, p, rate):
  w_s = rfftfreq(ns, 1.0/rate)
  d_s  = 1.*rate / ns
  w_r = rfftfreq(nr, 1.0/rate)
  d_r = 1.*rate / nr 
  # A linear interpolation of the two nearest frequencies, via the pitch
  # transformation p. I.e.
  # R(w) = a_f * S(floor(p * w / d_s) * d_s)
  #      + a_c * S(ceil(p * w / d_s) * d_s )
  i_wfloor = np.floor(w_r/p / d_s).astype(np.int)
  i_wceil = np.ceil(w_r/p / d_s).astype(np.int)
  w_floor = w_s[i_wfloor]
  w_ceil = w_s[i_wceil]
  a_floor0, a_ceil0 = (w_r/p - w_floor), (w_ceil - w_r/p)
  assert all(a_floor0 >= 0), a_floor0[a_floor0 < 0]
  assert all(a_ceil0 >= 0), a_ceil0[a_ceil0 < 0]
  a_floor = a_floor0 / (a_floor0 + a_ceil0 + 1e-6)
  a_ceil = a_ceil0 / (a_floor0 + a_ceil0 + 1e-6)

  absS = np.abs(S)
  angleS = np.angle(S)

  absR = a_floor * absS[i_wfloor] + a_ceil * absS[i_wceil]
  angleR = a_floor * angleS[i_wfloor] + a_ceil * angleS[i_wceil]

  R = absR# * np.exp(1j * angleR)

  assert R.size == w_r.size, (R.size, S.size, ns, nr)
  return R


def testKickTires():
  s, f = ReadAndConvert(os.path.join('testdata', 'foo_s80_p95.wav'))
  s = s[1024 : 15*1024]
  nfft, nskip = 1024, 256
  stft_windows = STFT(s, nfft, nskip)
  r = np.hstack((
    STIFT(stft_windows, nfft, nskip),
    STIFT(stft_windows, nfft/2, nskip/2),
    STIFT(stft_windows, nfft*2, nskip*2),
  ))
  ConvertAndWrite(r, f, os.path.join('testout', 'KickTires.wav'))


def testPlotPitches():
  for name in ['foo_s80_p50', 'foo_s80_p75', 'foo_s80_p95']:
    s, rate = ReadAndConvert(os.path.join('testdata', '%s.wav' % name))
    freqs, S = rfftfreq(s.size, 1.0/rate), np.log(np.abs(rfft(s)))
    plot.LinePlotHtml(
        'testout', 'testPitch_%s' % name,
        'Spectrum of %s' % name,
        [['freq', name]] + [[f, Sk] for f, Sk in zip(freqs, S)],
        logx=True, xticks_at_A=True)


def testResample():
  s, rate = ReadAndConvert(os.path.join('testdata', 'foo_s80_p95.wav'))
  s = s[1024 : 15*1024]
  nfft, nskip = 1024, 512
  Sws = STFT(s, nfft, nskip)
  nrfft, nrskip = 4096, 2048
  target_pitch = 880.0 # Try to sing an A.

  Rws = []
  for nw, Sw in enumerate(Sws):
    freqs = rfftfreq(nfft, 1.0/rate)
    plot.LinePlotHtml(
        'testout', 'testResampleFrame%dIn' % nw,
        'Spectrum of input frame %d' % nw,
        [['freq', str(nw)]] + [[f, Swk] for f, Swk in zip(freqs, np.abs(Sw))],
        logx=True, xticks_at_A=True)
    #p = target_pitch / EstimatePitch(Sw, rate)
    p=1.0
    Rw = Resample(Sw, nfft, nrfft, p, rate)
    freqs = rfftfreq(nrfft, 1.0/rate)
    plot.LinePlotHtml(
        'testout', 'testResampleFrame%dOut' % nw,
        'Spectrum of ouput frame %d' % nw,
        [['freq', str(nw)]] + [[f, Rwk] for f, Rwk in zip(freqs, np.abs(Rw))],
        logx=True, xticks_at_A=True)
    Rws.append(Rw)
  ConvertAndWrite(
      STIFT(Rws, nrfft, nrskip), rate,
      os.path.join('testout', 'testResample.wav'))


def test():
  testKickTires()
  testPlotPitches()
  testResample()


if __name__ == '__main__':
  test()
