import os

import click
import pandas as pd


def get_incidents(video):
    res = []

    if video['start1'] != -1:
        res.append(
            (video['start1'], video['end1'])
        )

    if video['start2'] != -1:
        res.append(
            (video['start2'], video['end2'])
        )

    return res


def overlap(inc, seg):
    return inc[1] >= seg[0] and seg[1] >= inc[0]


def get_in_anomaly(incidents, timestamp, segment_length):
    for inc in incidents:
        if overlap(inc, (timestamp, timestamp + segment_length)):
            return True

    return False


def generate_happens_at(video, incidents, timestamp, segment_length):
    return 'happensAt(\'{video}\', {value}, {timestamp}).'.format(
        video=video['path'],
        value='something' if get_in_anomaly(incidents, timestamp, segment_length) else 'nothing',
        timestamp=timestamp
    )


def generate_initiated_at(video, incidents, timestamp, segment_length):
    return 'initiatedAt(videoOnly(\'{video}\') = {value}, {timestamp}).'.format(
        video=video['path'],
        value='true' if get_in_anomaly(incidents, timestamp, segment_length) else 'false',
        timestamp=timestamp
    )


def generate_holds_at(video, incidents, timestamp, segment_length):
    return 'holdsAt(videoOnly(\'{video}\') = {value}, {timestamp}).'.format(
        video=video['path'],
        value='true' if get_in_anomaly(incidents, timestamp, segment_length) else 'false',
        timestamp=timestamp
    )


@click.command()
@click.argument('temporal_annotations_file', required=True, type=click.Path(exists=True))
@click.option('--step', required=False, default=8)
@click.option('--segment_length', required=False, default=16)
def generate_data(temporal_annotations_file, step, segment_length):
    temp_annot = pd.read_csv(
        temporal_annotations_file,
        delimiter='  ',
        header=None,
        names=['filename', 'video_type', 'start1', 'end1', 'start2', 'end2'],
        engine='python'
    )

    # Generate the path from the video_type and filename
    temp_annot['path'] = temp_annot['video_type'] + '/' + temp_annot['filename']

    for i, video in temp_annot.iterrows():
        incidents = get_incidents(video)

        n_frames_path = 'jpg/{}/n_frames'.format(video['path'].split('.')[0])
        if not os.path.exists(n_frames_path):
            continue
        with open(n_frames_path) as f:
            n_frames = int(f.read().strip())

        for timestamp in range(0, n_frames, step):
            print(
                # generate_happens_at(video, incidents, timestamp, segment_length)
                # generate_initiated_at(video, incidents, timestamp, segment_length)
                generate_holds_at(video, incidents, timestamp, segment_length)
            )


if __name__ == '__main__':
    generate_data()
