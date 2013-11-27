"Shut the fuck transform. Rasta don't work for no CIA."
import os
import math
from functools import partial
import numpy as np
import scipy.io.wavfile
import scipy.signal
from scipy.signal import hamming
from numpy.fft import rfftfreq, rfft, irfft, fftshift, ifftshift
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
  h = hamming(nfft)
  return [
      rfft(zpad(nfft, s[w_0 : min(s.size, w_0+nfft)]))
      for w_0 in xrange(0, s.size, nskip)]


def STIFT(stft_windows, nfft, nskip):
  "Sum contributions from each window. No normalization."
  r = np.zeros(nskip * (len(stft_windows) - 1) + nfft)
  h = hamming(nfft)
  for w, w_0 in zip(stft_windows, xrange(0, r.size, nskip)):
    r[w_0 : w_0+nfft] += irfft(w, nfft) * h
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
  i_wfloor = (np.floor(w_r[1:]/p / d_s) - 1).astype(np.int)
  i_wceil = (np.ceil(w_r[1:]/p / d_s) - 1).astype(np.int)
  
  w_floor = w_s[1+i_wfloor]
  w_ceil = w_s[1+i_wceil]
  a_floor0, a_ceil0 = (w_r[1:]/p - w_floor), (w_ceil - w_r[1:]/p)
  # Some freqs may be exact matches, e.g. if nr == 2*ns, every other one.
  exact = (a_floor0 == 0.0) & (a_ceil0 == 0.0)
  # Ignore the DC Component and any exact matches in the interpolation.
  R = np.hstack((np.array([S[0]]), np.zeros(w_r.size-1)))
  R[1:][exact] = S[1+i_wfloor[exact]]
  # The rest: interpolate mag/phase from the nearest sampled frequencies.
  absS, angS = np.abs(S[1:]), np.angle(S[1:])
  intrp = ~exact
  # Basic floor/ceil linear interpolation ...
  i_fl, i_cl = i_wfloor[intrp], i_wceil[intrp]
  afl0, acl0 = a_floor0[intrp], a_ceil0[intrp]
  afl = 1.0 - afl0 / (afl0 + acl0)
  acl = 1.0 - acl0 / (afl0 + acl0)
  absR = afl * absS[i_fl] + acl * absS[i_cl]
  angR = afl * angS[i_fl] + acl * angS[i_cl]
  R[1:][intrp] = absR * np.exp(1j * angR)
  assert R.size == w_r.size, (R.size, w_r.size, S.size, ns, nr)
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


def PlotPitches(S, rate, dirpath='testout', name='pitches', title=''):
  freqs = rfftfreq(S.size, 1.0/rate)
  plot.LinePlotHtml(
      dirpath, name, title,
      [['freq', name]] + [[f, Sk] for f, Sk in zip(freqs, S)],
      logx=True, xticks_at_A=True)


def testPlotPitches():
  for name in ['foo_s80_p50', 'foo_s80_p75', 'foo_s80_p95']:
    s, rate = ReadAndConvert(os.path.join('testdata', '%s.wav' % name))
    S = np.abs(rfft(s))
    PlotPitches(S, rate, name='testPlotPitches_%s' % name,
        title='Spectrum of %s' % name)


def testResample():
  s, rate = ReadAndConvert(os.path.join('testdata', 'foo_s80_p95.wav'))
  s = scipy.signal.resample(s, 4*len(s))
  rate = 4*rate
  ConvertAndWrite(s, rate, os.path.join('testout', 'foo_resampled.wav'))
  s = s[0 : 15*2048]
  nfft, nskip = 1024, 512
  Sws = STFT(s, nfft, nskip)
  nrfft, nrskip = 4096, 2048
  target_pitch = 880. #440.0 * 2 ** (7.0 / 12)

  Rws = []
  for nw, Sw in enumerate(Sws):
    freqs = rfftfreq(nfft, 1.0/rate)
    epitch = EstimatePitch(Sw, rate)

    if epitch == 0.0: 
      p = 1
    else:
      p = target_pitch / epitch
      print '%d: epitch=%f p=%f' % (nw, epitch, p)

    PlotPitches(np.abs(Sw), rate, name='testResampleFrame%dInMag' % nw,
                title='Spectrum of input frame %d (%f)' % (nw, epitch))
    PlotPitches(np.angle(Sw), rate, name='testResampleFrame%dInPhase' % nw,
                title='Phase of input frame %d' % nw)

    Rw = Resample(Sw, nfft, nrfft, p, rate)

    Rws.append(Rw)

    erpitch = EstimatePitch(Rw, rate)
    PlotPitches(np.abs(Rw), rate, name='testResampleFrame%dOutMag' % nw,
                title='Spectrum of output frame %d (%f)' % (nw, erpitch))
    PlotPitches(np.angle(Rw), rate, name='testResampleFrame%dOutPhase' % nw,
                title='Phase of output frame %d' % nw)
  r = STIFT(Rws, nrfft, nrskip)
  r = scipy.signal.resample(r, int(np.floor(r.size/2)))
  rate = rate/2

  ConvertAndWrite(
      r, rate,
      os.path.join('testout', 'testResample.wav'))


def test():
  testKickTires()
  testPlotPitches()
  testResample()


if __name__ == '__main__':
  test()
