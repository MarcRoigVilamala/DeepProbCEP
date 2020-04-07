import os
import re
import sys

import click

from examples.NIPS.UrbanSounds8K.SequenceDetection.run import run_linear

sys.path.append('../../../')


@click.command()
@click.argument('start_path', type=click.Path(exists=True, dir_okay=True, file_okay=False))
@click.option('--scenario', default='')
@click.option('--noise', default='')
def execute_scenarios(start_path, scenario, noise):
    for folder in sorted(os.listdir(start_path)):
        if folder.startswith('scenario') and re.search(scenario, folder):
            print("#######################################################################################")
            print(folder)

            prob_ec_cached = '{}/{}/prob_ec_cached.pl'.format(start_path, folder)
            if not os.path.isfile(prob_ec_cached):
                prob_ec_cached = '{}/ProbLogFiles/prob_ec_cached.pl'.format(start_path)

            event_defs = '{}/{}/event_defs.pl'.format(start_path, folder)
            if not os.path.isfile(event_defs):
                event_defs = '{}/ProbLogFiles/event_defs.pl'.format(start_path)

            for subfolder in sorted(os.listdir(start_path + folder)):
                # if subfolder != 'noise_1_00':
                #     continue

                if re.search(noise, subfolder) and os.path.isdir('{}/{}/{}'.format(start_path, folder, subfolder)):
                    print('===================================================================================')
                    print(subfolder)

                    run_linear(
                        '{}/{}/{}/init_train_data.txt'.format(start_path, folder, subfolder),
                        '{}/{}/{}/init_sound_test_data.txt'.format(start_path, folder, subfolder),
                        [
                            prob_ec_cached,
                            event_defs
                        ],
                        problog_train_files=[
                            '{}/{}/{}/in_train_data.txt'.format(start_path, folder, subfolder)
                        ],
                        problog_test_files=[
                            '{}/{}/{}/in_test_data.txt'.format(start_path, folder, subfolder)
                        ]
                    )


if __name__ == '__main__':
    execute_scenarios()
