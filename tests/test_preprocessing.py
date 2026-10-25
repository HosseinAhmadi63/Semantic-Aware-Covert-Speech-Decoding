import csv

import numpy as np
import pytest
from scipy.io import wavfile

from semantic_speech.preprocessing import (
    _repair,
    _robust_scale,
    _waveform,
    log_mel,
    prepare_manifest,
    preprocess_recording,
)


def recording_data():
    import mne
    rng = np.random.default_rng(32)
    times = np.arange(6000) / 500
    names = mne.channels.make_standard_montage('standard_1020').ch_names[:63]
    eeg = np.sin(2 * np.pi * 10 * times)[None] * 2e-5
    eeg = eeg + rng.normal(0, 5e-6, (63, len(times)))
    emg = rng.normal(0, 5e-5, (2, len(times)))
    return eeg, emg, names


def test_repair_and_robust_scale():
    values = np.array([[1., np.nan, 3., 4.], [4., 4., 4., 4.]])
    repaired, invalid = _repair(values)
    np.testing.assert_allclose(repaired[0], [1, 2, 3, 4])
    np.testing.assert_array_equal(invalid, [False, True, False, False])
    scaled = _robust_scale(repaired)
    assert np.isfinite(scaled).all()
    np.testing.assert_allclose(scaled[1], 0)
    np.testing.assert_allclose(np.median(scaled[0]), 0)
    with pytest.raises(ValueError, match='two finite'):
        _repair(np.full((2, 10), np.nan))


def test_continuous_preprocessing_preserves_rate_and_channel_contract():
    eeg, emg, names = recording_data()
    eeg[0, 4000] = np.nan
    emg[0, 4001] = np.inf
    clean_eeg, clean_emg, quality = preprocess_recording(eeg, emg, channel_names=names,
        config={'ica': False, 'eeg_taps': 101, 'emg_taps': 101, 'quality_threshold': 20})
    assert clean_eeg.shape == (63, 2400)
    assert clean_emg.shape == (2, 2400)
    assert np.isfinite(clean_eeg).all() and np.isfinite(clean_emg).all()
    assert quality['eeg_repaired_samples'] == 1
    assert quality['emg_repaired_samples'] == 1
    assert any(start <= 8 < stop for start, stop in quality['bad_intervals'])
    assert not quality['ica_applied']
    assert quality['output_sfreq'] == 200


def test_log_mel_exact_shape_and_padding():
    waveform = np.sin(2 * np.pi * 240 * np.arange(48000) / 48000)
    representation = log_mel(waveform)
    assert representation.shape == (1, 64, 301)
    assert representation.dtype == np.float32
    assert np.isfinite(representation).all()
    assert abs(float(representation.mean())) < 1e-5
    assert abs(float(representation.std()) - 1) < 1e-5
    with pytest.raises(ValueError, match='48000'):
        log_mel(waveform, 16000)


def test_waveform_integer_units(tmp_path):
    path = tmp_path / 'audio.wav'
    wavfile.write(path, 48000, np.array([-32768, 0, 32767], dtype=np.int16))
    waveform, sample_rate = _waveform(path)
    np.testing.assert_allclose(waveform, [-1, 0, 32767 / 32768])
    assert sample_rate == 48000


def test_prepare_manifest_aligns_overt_audio_and_covert_absence(tmp_path):
    eeg, emg, names = recording_data()
    np.savez(tmp_path / 'recording.npz', eeg=eeg, emg=emg, sfreq=500,
             channel_names=np.asarray(names))
    wavfile.write(tmp_path / 'audio.wav', 48000, np.zeros(48000, dtype=np.int16))
    manifest = tmp_path / 'trials.csv'
    fields = ['subject', 'run', 'trial', 'label', 'mode', 'phase', 'onset', 'recording', 'audio']
    with manifest.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(dict(zip(fields, ['S01', 'R01', '1', 'water', 'text', 'overt', 1, 'recording.npz', 'audio.wav'])))
        writer.writerow(dict(zip(fields, ['S01', 'R01', '1', 'water', 'text', 'covert', 5, 'recording.npz', ''])))
    summary = prepare_manifest(manifest, tmp_path / 'prepared.npz',
                               {'ica': False, 'eeg_taps': 101, 'emg_taps': 101, 'quality_threshold': 20})
    assert summary['trials'] == 2
    with np.load(tmp_path / 'prepared.npz', allow_pickle=False) as data:
        assert data['x'].shape == (2, 65, 600)
        assert data['audio'].shape == (2, 1, 64, 301)
        assert not np.any(data['audio'][1])
        assert data['phase'].tolist() == ['overt', 'covert']
        assert data['y'].tolist() == [0, 0]
        assert data['trial'].tolist() == ['1', '1']
