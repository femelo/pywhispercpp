#!/usr/bin/env python

"""
A simple Command Line Interface to test the package
"""

import argparse
import importlib.metadata
import logging
import pprint

import pywhispercpp.constants as constants
import pywhispercpp.utils as utils
from pywhispercpp.model import Model

__version__ = importlib.metadata.version("pywhispercpp")

__header__ = f"""
PyWhisperCpp
A simple Command Line Interface to test the package
Version: {__version__}               
====================================================
"""

logger = logging.getLogger("pywhispercpp-cli")
logger.setLevel(logging.DEBUG)
logging.basicConfig()


def _get_params(args) -> dict:
    """
    Helper function to get params from argparse as a `dict`
    """
    params = {}
    inv_params_mapping = {v: k for k, v in constants.PARAMS_MAPPING.items()}
    for arg in args.__dict__:
        if arg in constants.PARAMS_SCHEMA and getattr(args, arg) is not None:
            params[arg] = getattr(args, arg)
        elif arg in inv_params_mapping:
            arg_ = inv_params_mapping[arg]
            if "." in arg_:
                arg_, subarg_ = arg_.split(".")
                if arg_ not in params:
                    params[arg_] = {}
                params[arg_][subarg_] = getattr(args, arg)
            else:
                params[arg_] = getattr(args, arg)
        else:
            pass
    return params


def run(args):
    logger.info(f"Running with model `{args.model}`")
    params = _get_params(args)
    logger.info(
        f"Running with params\n{pprint.pformat(params, indent=2, sort_dicts=False)}"
    )
    m = Model(model=args.model, **params)
    logger.info(
        f"System info:\n  n_threads = {m.get_params()['n_threads']}"
        f"\n  processors = {args.processors}"
        f"\n  other = {m.system_info()}"
    )
    for file in args.media_file:
        try:
            logger.info(f"Processing file {file} ...")
            segs = m.transcribe(
                file,
                n_processors=int(args.processors) if args.processors else None,
            )
            m.print_timings()
        except KeyboardInterrupt:
            logger.info("Transcription manually stopped")
            break
        except Exception as e:
            logger.error(f"Error while processing file {file}: {e}")
        finally:
            if segs:
                # output stuff
                if args.output_txt:
                    logger.info("Saving result as a txt file ...")
                    txt_file = utils.output_txt(segs, file)
                    logger.info(f"txt file saved to {txt_file}")
                if args.output_vtt:
                    logger.info("Saving results as a vtt file ...")
                    vtt_file = utils.output_vtt(segs, file)
                    logger.info(f"vtt file saved to {vtt_file}")
                if args.output_srt:
                    logger.info("Saving results as a srt file ...")
                    srt_file = utils.output_srt(segs, file)
                    logger.info(f"srt file saved to {srt_file}")
                if args.output_csv:
                    logger.info("Saving results as a csv file ...")
                    csv_file = utils.output_csv(segs, file)
                    logger.info(f"csv file saved to {csv_file}")
                else:
                    for i, seg in enumerate(segs):
                        logger.info(f"Segment {i:02}: {seg}")


def main():
    print(__header__)
    parser = argparse.ArgumentParser(description="", allow_abbrev=True)
    # Positional args
    parser.add_argument(
        "media_file",
        type=str,
        nargs="+",
        help="The path of the media file or a list of files separated by space",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="tiny",
        help="Path to the `ggml` model, or just the model name",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "--processors",
        type=int,
        default=1,
        help="number of processors to use during computation",
    )
    parser.add_argument(
        "-otxt",
        "--output-txt",
        action="store_true",
        help="output result in a text file",
    )
    parser.add_argument(
        "-ovtt", "--output-vtt", action="store_true", help="output result in a vtt file"
    )
    parser.add_argument(
        "-osrt", "--output-srt", action="store_true", help="output result in a srt file"
    )
    parser.add_argument(
        "-ocsv", "--output-csv", action="store_true", help="output result in a csv file"
    )

    # add params from PARAMS_SCHEMA
    for param in constants.PARAMS_SCHEMA:
        param_fields = constants.PARAMS_SCHEMA[param]
        type_ = param_fields["type"]
        descr_ = param_fields["description"]
        default_ = param_fields["default"]
        if type_ is dict:
            for dft_key, dft_val in default_.items():
                map_key = f"{param}.{dft_key}"
                map_val = constants.PARAMS_MAPPING.get(map_key)
                if map_val:
                    parser.add_argument(
                        f"--{map_val.replace('_', '-')}",
                        type=type(dft_val),
                        default=dft_val,
                        help=map_key,
                    )
        elif param in constants.PARAMS_MAPPING:
            mapped_param = constants.PARAMS_MAPPING[param]
            parser.add_argument(
                f"--{mapped_param.replace('_', '-')}",
                type=type_,
                default=default_,
                help=descr_,
            )
        else:
            parser.add_argument(
                f"--{param.replace('_', '-')}",
                type=type_,
                default=default_,
                help=descr_,
            )

    args, _ = parser.parse_known_args()
    run(args)


if __name__ == "__main__":
    main()
