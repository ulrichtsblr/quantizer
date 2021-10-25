from quantizer.utils import *
from quantizer.wavetable import *
import numpy as np
from scipy.io import wavfile
from typing import List


class Clavier:

    def __init__(self):
        self.clavier = None

    def load(self, clavier: List[tuple]) -> List[tuple]:
        self.clavier = clavier
        return self.clavier


class Stream:

    def __init__(self, wavetable: Wavetable, claviers: List[list], bpm: int = None, line_level: float = 1.0):
        self.wavetable = wavetable
        self.cat = [chord for clavier in claviers for chord in clavier]
        self.bpm = bpm
        self.line_level = line_level
        self.stream = np.array([])

    def get_stream(self):
        return self.stream

    def get_bpm(self):
        return self.bpm

    def set_bpm(self, bpm: int):
        self.bpm = bpm

    def bounce(self):
        assert self.bpm
        for i in range(len(self.cat)):
            nsamples = beats2samples(n=self.cat[i][0], bpm=self.bpm, fs=self.wavetable.get_fs())
            y = np.zeros(nsamples)
            for j in range(1, len(self.cat[i])):
                f = midi2freq(m=self.cat[i][j])
                y += self.wavetable.read(f=f, nsamples=nsamples)
            self.stream = self.line_level * np.concatenate([self.stream, y], axis=None)


class Sequencer:

    def __init__(self, streams: List[Stream], bpm: int = 128, fs: int = 44100):
        self.streams = streams
        self.bpm = bpm
        self.fs = fs
        self.mix = None

    def _mix(self):

        if not self.streams[0].get_bpm():
            self.streams[0].set_bpm(self.bpm)
        self.streams[0].bounce()
        self.mix = self.streams[0].get_stream()

        if len(self.streams) > 1:
            for i in range(1, len(self.streams)):
                if not self.streams[i].get_bpm():
                    self.streams[i].set_bpm(self.bpm)
                self.streams[i].bounce()
                self.mix += self.streams[i].get_stream()

    def export(self, file_name: str = "sequencer.wav", master_level: float = 1.0):
        self._mix()
        wavfile.write(file_name, self.fs, minmaxscale_i16(stream=master_level * self.mix))
