import pandas as pd
import os
import numpy as np
import soundfile
import torch

from torch.utils.data import Dataset

DATASET_PATH = r'/Users/kolai/Data/speech_commands_v0.02/'

SAMPLING_RATE = 16000
SAMPLES_DURATION_SEC = 1
SAMPLES_LEN = int(SAMPLING_RATE*SAMPLES_DURATION_SEC)

DIGIT_SUBSETS = {'train': 0, 'valid': 1, 'test': 2}
DIGIT_LABELS = {'TARGET': 1, 'NON_TARGET': -1, 'BACKGROUND_NOISE': 0}

TARGET_LABEL = 'stop'
BACKGROUND_NOISE_LABEL = '_background_noise_'


def prepare_ref_dataset_csv(name, n_samples, target_ratio=0.20, non_target_ratio=0.45, background_noise_ratio=0.35,
                            test_ratio=0.20, valid_ratio=0.20):
    np.random.seed(42)
    labels = [f for f in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, f))]

    # prepare dataframe with references
    ref_df = pd.DataFrame(columns=['path', 'label', 'start', 'stop', 'subset'])

    # select target utterances
    n_target_samples = int(n_samples * target_ratio)
    wav_files = [os.path.join(TARGET_LABEL, f)
                 for f in os.listdir(os.path.join(DATASET_PATH, TARGET_LABEL))
                 if f.endswith('.wav')]
    assert len(wav_files) > n_target_samples, \
        f'Not enough target slices ({len(wav_files)} from {n_target_samples})'
    ref_targets_df = pd.DataFrame({'path': wav_files[:n_target_samples], 'label': DIGIT_LABELS['TARGET'],
                                   'start': -1, 'stop': -1})
    ref_df = ref_df.append(ref_targets_df, ignore_index=True)

    # select non target utterances
    n_non_target_samples = int(n_samples * non_target_ratio)
    non_target_label = [la for la in labels if la not in [TARGET_LABEL, BACKGROUND_NOISE_LABEL]]
    wav_files = []
    for label in non_target_label:
        wav_files += [os.path.join(label, f) for f in os.listdir(os.path.join(DATASET_PATH, label)) if
                      f.endswith('.wav')]

    wav_files = np.random.choice(wav_files, n_non_target_samples, replace=False)
    ref_non_targets = pd.DataFrame({'path': wav_files, 'label': DIGIT_LABELS['NON_TARGET'], 'start': -1, 'stop': -1})
    ref_df = ref_df.append(ref_non_targets, ignore_index=True)

    # select background noise
    n_background_noise_samples = int(background_noise_ratio * n_samples)

    wav_files = [os.path.join(BACKGROUND_NOISE_LABEL, f)
                 for f in os.listdir(os.path.join(DATASET_PATH, BACKGROUND_NOISE_LABEL))
                 if f.endswith('.wav')]
    wav_files_slices = []
    slice_size = int(SAMPLING_RATE * SAMPLES_DURATION_SEC)
    slice_step = slice_size // 2
    for wav_file in wav_files:
        info = soundfile.info(os.path.join(DATASET_PATH, wav_file))
        assert info.samplerate == SAMPLING_RATE, f'Sample rate should be {SAMPLING_RATE}, got {info.samplerate}'
        wav_files_slices += [(wav_file, DIGIT_LABELS['BACKGROUND_NOISE'], ind - slice_size, ind)
                             for ind in range(slice_size, int(info.duration * SAMPLING_RATE), slice_size // slice_step)]

    assert len(wav_files_slices) > n_background_noise_samples, \
        f'Not enough background noise slices ({len(wav_files_slices)} from {n_background_noise_samples})'
    inds = np.random.choice(range(len(wav_files_slices)), n_background_noise_samples, replace=False)
    wav_files_slices = [wav_files_slices[i] for i in inds]
    ref_background_noise = pd.DataFrame(wav_files_slices, columns=['path', 'label', 'start', 'stop'])
    ref_df = ref_df.append(ref_background_noise, ignore_index=True)

    # divide into train, validation, test
    ref_df = ref_df.sample(frac=1, ignore_index=True, random_state=42)
    n_samples = len(ref_df)
    n_train_samples = int(n_samples * (1 - valid_ratio - test_ratio))
    n_valid_samples = int(n_samples * valid_ratio)
    ref_df['subset'] = DIGIT_SUBSETS['train']
    ref_df.loc[n_train_samples:, 'subset'] = DIGIT_SUBSETS['valid']
    ref_df.loc[n_train_samples + n_valid_samples:, 'subset'] = DIGIT_SUBSETS['test']

    csv_path = os.path.join(DATASET_PATH, f'ref_{name}_{n_samples}.csv')
    ref_df.to_csv(csv_path, index=False)
    return csv_path


class SCDataset(Dataset):
    def __init__(self, ref_dataset_csv_path, subset='train', transform=None, transform_label=None):
        self.ref_df = pd.read_csv(ref_dataset_csv_path).query(f'subset=={DIGIT_SUBSETS[subset]}')
        self.transform = transform
        self.transform_label = transform_label

    def __getitem__(self, n):
        sample_ref = self.ref_df.iloc[n]
        file_path = os.path.join(DATASET_PATH, sample_ref['path'])
        if sample_ref['label'] == DIGIT_LABELS['BACKGROUND_NOISE']:
            x, _sr = soundfile.read(file_path, start=sample_ref['start'], stop=sample_ref['stop'])
        else:
            x, _sr = soundfile.read(file_path)
            if len(x) < SAMPLES_LEN:
                x = np.concatenate([x, np.zeros(SAMPLES_LEN-len(x))])
        if self.transform:
            x = self.transform(x)

        label = 1 if sample_ref['label']==DIGIT_LABELS['TARGET'] else 0
        if self.transform_label:
            label = self.transform_label(label)
        return x, label

    def __len__(self):
        return len(self.ref_df)


if __name__ == '__main__':
    # prepare dataset
    csv_path = prepare_ref_dataset_csv('sc_v2', 17000)
    print(csv_path)

    # # load as numpy array
    # dataset = SCDataset('/Users/kolai/Data/speech_commands_v0.01/ref_small_1000.csv', 'train')
    # from solution.utils import play
    # play(dataset[0][0])
    #
    # # # load and transform
    # from torchvision.transforms import Compose
    # from solution.preprocessing import Spectrogram, NormalizeSpec, ToTensor
    # transform = Compose([Spectrogram(), NormalizeSpec(), ToTensor()])
    # dataset = SCDataset('/Users/kolai/Data/speech_commands_v0.01/ref_small_1000.csv', 'train', transform=transform)
    # print(dataset[0])
    # print(dataset[0][0].shape)
