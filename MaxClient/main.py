import argparse
import time

import fluteline

import watson_streaming
import watson_streaming.utilities


def parse_arguments():
    parser = argparse.ArgumentParser(description="doc")
    parser.add_argument('credentials', help='credentials.json')
    return parser.parse_args()


def main():
    args = parse_arguments()
    settings = {
        'inactivity_timeout': -1,  # Don't kill me after 30 seconds
        'interim_results': True,
    }

    nodes = [
        watson_streaming.utilities.MicAudioGen(),
        watson_streaming.Transcriber(settings, args.credentials),
        watson_streaming.utilities.Printer(),
    ]

    fluteline.connect(nodes)
    fluteline.start(nodes)

    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        pass
    finally:
        fluteline.stop(nodes)


if __name__ == '__main__':
    main()