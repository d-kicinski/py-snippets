import logging
import time
from pathlib import Path
from typing import Tuple

import tensorflow as tf
import soundfile

AUDIO_LENGTH = 16_000


class AudioDataset:
    def __init__(self, data_root: Path):
        self._files = list(map(str, data_root.rglob("*.wav")))
        self._index = 0

    def __len__(self):
        return len(self._files)

    def __next__(self):
        if self._index >= len(self._files):
            raise StopIteration
        value = self._files[self._index]
        self._index += 1

        label = value.split("/")[1]

        return value, label

    def __iter__(self):
        return self

    def __call__(self):
        return self


def generator(data_root: Path):
    files = list(data_root.rglob("*.wav"))
    for f in files:
        yield f


def iter_dataset(dataset: tf.data.Dataset):
    for audio_data, label in dataset:
        pass
    print(audio_data.shape)


def to_spectrogram(audio_data: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    spectrogram = tf.signal.stft(audio_data, frame_length=1024, frame_step=512)
    spectrogram = tf.abs(spectrogram)
    return spectrogram, label


def to_waveform(path: tf.string, label: tf.string) -> Tuple[tf.Tensor, tf.Tensor]:
    audio, _ = tf.audio.decode_wav(tf.io.read_file(path))
    audio = tf.squeeze(audio, axis=-1)
    audio = tf.pad(audio, paddings=((0, AUDIO_LENGTH - len(audio)),))
    return audio, label


def dataset():
    logging.getLogger().setLevel(logging.ERROR)

    data_root = Path("speech_commnads")
    audio_dataset = AudioDataset(data_root)
    print(f"AudioDataset: {len(audio_dataset)}")
    dataset = tf.data.Dataset.from_generator(
        generator=audio_dataset,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string))
    )
    dataset = dataset \
        .take(1000) \
        .shuffle(buffer_size=1000) \
        .map(to_waveform) \
        .batch(10) \
        .map(to_spectrogram) \
        .cache()

    for epoch in range(5):
        start = time.time()
        iter_dataset(dataset)
        end = time.time()
        print(f"epoch {epoch}: {end - start}")


if __name__ == '__main__':
    dataset()
