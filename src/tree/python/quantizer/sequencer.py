from quantizer.kernel.kontext import kntxt
from quantizer.kernel.util import (
    Boolean, Scalar, Sequence, Integer, String, List, beats2samples, bounce,
    db2mag, midi2freq, midi2mag, minmaxscale_i16, transpose
)
from quantizer.kernel.waveform import Stream
from copy import deepcopy
import numpy as np
from uuid import uuid1
import yaml


class Event:

    nsamples_default = beats2samples(
        beats=1.0,
        bpm=kntxt().bpm,
        fs=kntxt().fs,
    )
    f_default = midi2freq(36)

    def __init__(
            self,
            idle: Boolean = False,
            nsamples: Scalar = nsamples_default,
            f: Scalar = f_default,
            p: Scalar = 0,
            a: Scalar = 1,
            cc: Sequence = None,
    ):
        self.idle = idle
        self.nsamples = nsamples
        self.f = f
        self.p = p
        self.a = a
        self.cc = cc

    def get_idle(self):
        return self.idle

    def get_nsamples(self):
        return self.nsamples

    def get_f(self):
        return self.f

    def get_p(self):
        return self.p

    def get_a(self):
        return self.a

    def get_cc(self):
        return self.cc

    def set_f(self, f):
        self.f = f

    def set_p(self, p):
        self.p = p

    def set_a(self, a):
        self.a = a

    def set_cc(self, cc):
        self.cc = cc


class Pattern:

    def __init__(
        self,
        identifier: String = str(uuid1()),
        events: List = None,
        patterns: Sequence = None,
    ):
        self.identifier = identifier
        self.events = []
        if events:
            self.events = events
        if patterns:
            for p in patterns:
                events = p.get_events()
                for e in events:
                    self.events.append(e)

    def get_identifier(self):
        return self.identifier

    def get_events(self):
        return self.events

    def append(self, event: Event):
        self.events.append(event)

    def transpose(self, oc=0, st=0, ct=0):
        events = deepcopy(self.events)
        for e in events:
            e.set_f(transpose(e.get_f(), *[oc, st, ct]))
        return Pattern(events=events)

    def matmul(self, other, r):
        if isinstance(other, Integer):
            events = []
            for _ in range(other):
                events += deepcopy(self.events)
            return Pattern(events=events)
        elif isinstance(other, Pattern):
            if r:
                events = deepcopy(other.get_events())
                for e in self.events:
                    events.append(deepcopy(e))
                return Pattern(events=events)
            else:
                events = deepcopy(self.events)
                for e in other.get_events():
                    events.append(deepcopy(e))
                return Pattern(events=events)
        else:
            raise RuntimeError

    def __matmul__(self, other):
        return self.matmul(other, r=False)

    def __rmatmul__(self, other):
        return self.matmul(other, r=True)


class Sequencer:

    class YAMLEvent(Event):

        def __init__(
            self,
            idle: Boolean = False,
            beats: Scalar = 1.0,
            f: Integer = 36,
            a: Integer = 127,
            *cc,
        ):
            super().__init__(
                idle=idle,
                nsamples=beats2samples(
                    beats=beats,
                    bpm=kntxt().bpm,
                    fs=kntxt().fs,
                ),
                f=midi2freq(f),
                a=midi2mag(a),
                cc=tuple(map(float, cc)),
            )

    def __init__(
        self,
        streams: List = None,
        line_levels: List = None,
        master_level: Scalar = 0
    ):
        self.streams = streams
        self.line_levels = line_levels
        self.master_level = master_level
        self.patterns = dict()

    def get_pattern(self, identifier):
        return self.patterns[identifier]

    def mix(self):
        mix = Sequencer._mix(
            streams=self.streams,
            line_levels=self.line_levels
        )
        mix = db2mag(self.master_level) * mix.get_ndarray()
        return mix

    def export(self, file_path: str = "sequencer.wav"):
        bounce(minmaxscale_i16(self.mix()), file_path, kntxt().fs)

    def load_yaml(self, file_path: str):
        manuscript = yaml.load(open(file_path, 'r'), Loader=yaml.BaseLoader)
        for k, v in manuscript.items():
            pattern = Pattern(identifier=k)

            # pattern is a single event
            if len(v) == 0 or not isinstance(v[0], list):
                pattern.append(self._parse_yaml_event(v))

            # pattern contains multiple events
            else:
                for e in v:
                    pattern.append(self._parse_yaml_event(e))

            self.patterns[pattern.get_identifier()] = pattern

    @staticmethod
    def _mix(streams: Sequence, line_levels: Sequence = None) -> Stream:
        dim = []
        for stream in streams:
            dim.append(len(stream))
        streams = Sequencer._pad(unpadded_streams=streams, n=max(dim))
        if line_levels:
            streams = Sequencer._line_level(streams, line_levels=line_levels)
        mix = np.zeros(max(dim))
        for stream in streams:
            mix += stream.get_ndarray()
        return Stream(mix)

    @staticmethod
    def _pad(unpadded_streams, n):
        padded_streams = []
        for stream in unpadded_streams:
            if len(stream) != n:
                s = np.concatenate(
                    [stream.get_array(), np.zeros(n - len(stream))]
                )
            else:
                s = stream.get_array()
            padded_streams.append(Stream(s))
        return padded_streams

    @staticmethod
    def _line_level(streams, line_levels):
        assert len(streams) == len(line_levels)
        ll_streams = []
        for i in range(len(streams)):
            s = db2mag(line_levels[i]) * streams[i].get_array()
            ll_streams.append(Stream(s))
        return ll_streams

    def _parse_yaml_event(self, e: List) -> Event:
        if len(e) < 3:
            if len(e) == 0:
                args = [True]
                event = self.YAMLEvent(*args)
            else:
                args = [True, float(e[0])]
                event = self.YAMLEvent(*args)
        elif len(e) == 3:
            if '-' in e[1] and '-' in e[2]:
                args = [False, float(e[0]), 36, 127]
                event = self.YAMLEvent(*args)
            elif '-' in e[1]:
                args = [False, float(e[0]), 36, float(e[2])]
                event = self.YAMLEvent(*args)
            elif '-' in e[2]:
                args = [False, float(e[0]), float(e[1]), 127]
                event = self.YAMLEvent(*args)
            else:
                args = [False, float(e[0]), float(e[1]), float(e[2])]
                event = self.YAMLEvent(*args)
        else:
            if '-' in e[1] and '-' in e[2]:
                args = [False, float(e[0]), 36, 127] + e[3:]
                event = self.YAMLEvent(*args)
            elif '-' in e[1]:
                args = [False, float(e[0]), 36, float(e[2])] + e[3:]
                event = self.YAMLEvent(*args)
            elif '-' in e[2]:
                args = [False, float(e[0]), float(e[1]), 127] + e[3:]
                event = self.YAMLEvent(*args)
            else:
                args = [False, float(e[0]), float(e[1]), float(e[2])] + e[3:]
                event = self.YAMLEvent(*args)
        return event
