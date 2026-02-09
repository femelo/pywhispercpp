#!/usr/bin/env python

"""
A simple example showcasing how to use pywhispercpp to transcribe a recording.
"""

import argparse
import importlib.metadata
import logging

import sounddevice as sd

import pywhispercpp.constants
from pywhispercpp.model import Model


__version__ = importlib.metadata.version("pywhispercpp")

__header__ = f"""
===================================================================
PyWhisperCpp
A simple example of transcribing a recording, based on whisper.cpp
Version: {__version__}               
===================================================================
"""


class Recording:
    """
    Recording class

    Example usage
    ```python
    from pywhispercpp.examples.recording import Recording

    myrec = Recording(5)
    myrec.start()
    ```
    """

    def __init__(self, duration: int, model: str = "tiny.en", **model_params):
        self.duration = duration
        self.sample_rate = pywhispercpp.constants.WHISPER_SAMPLE_RATE
        self.channels = 1
        self.pwcpp_model = Model(model, print_realtime=True, **model_params)

    def start(self):
        logging.info(f"Start recording for {self.duration}s ...")
        recording = sd.rec(
            int(self.duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
        )
        sd.wait()
        logging.info("Duration finished")
        _res = self.pwcpp_model.transcribe(recording)
        self.pwcpp_model.print_timings()


def _main():
    print(__header__)
    parser = argparse.ArgumentParser(description="", allow_abbrev=True)
    # Positional args
    parser.add_argument("duration", type=int, help="Duration in seconds")
    parser.add_argument(
        "-m",
        "--model",
        default="tiny.en",
        type=str,
        help="Whisper.cpp model, default to %(default)s",
    )

    args = parser.parse_args()

    myrec = Recording(duration=args.duration, model=args.model)
    myrec.start()


if __name__ == "__main__":
    _main()
