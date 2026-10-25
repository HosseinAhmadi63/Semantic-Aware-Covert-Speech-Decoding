from fractions import Fraction
from pathlib import Path
import csv

import numpy as np
from scipy import signal
from scipy.io import wavfile


LABELS = ('water', 'toilet', 'light', 'pain')


def _repair(values):
    array = np.asarray(values, dtype=np.float64).copy()
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError('Signals must have shape channels by samples.')
    invalid = ~np.isfinite(array)
    for row, bad in zip(array, invalid):
        good = ~bad
        if good.sum() < 2:
            raise ValueError('Every channel requires at least two finite samples.')
        if bad.any():
            row[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(good), row[good])
    return array, invalid.any(axis=0)


def _robust_z(values, axis=0):
    median = np.median(values, axis=axis, keepdims=True)
    mad = np.median(np.abs(values - median), axis=axis, keepdims=True)
    tolerance = np.maximum(np.abs(median) * 1e-6, np.finfo(float).eps)
    return (values - median) / np.maximum(1.4826 * mad, tolerance)


def _intervals(mask, sfreq):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return [[float(a / sfreq), float(b / sfreq)]
            for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def _quality(eeg, invalid, sfreq, threshold):
    window = max(2, int(round(sfreq)))
    starts = list(range(0, eeg.shape[1], window))
    blocks = [eeg[:, start:min(start + window, eeg.shape[1])] for start in starts]
    blocks = [(start, block) for start, block in zip(starts, blocks) if block.shape[1] > 1]
    ptp = np.stack([np.ptp(block, axis=1) for _, block in blocks])
    variance = np.stack([np.var(block, axis=1) for _, block in blocks])
    channel_ptp = np.median(ptp, axis=0)
    channel_var = np.median(variance, axis=0)
    tiny = np.finfo(float).tiny
    flat = (channel_ptp <= max(np.median(channel_ptp) * 1e-6, tiny))
    noisy = ((_robust_z(np.log(np.maximum(channel_ptp, tiny)), axis=0) > threshold)
             | (_robust_z(np.log(np.maximum(channel_var, tiny)), axis=0) > threshold))
    bad_channels = flat | noisy
    if (~bad_channels).sum() < 2:
        raise ValueError('At least two usable EEG channels are required.')
    window_bad = ((_robust_z(np.log(np.maximum(ptp[:, ~bad_channels], tiny)), axis=0) > threshold)
                  | (_robust_z(np.log(np.maximum(variance[:, ~bad_channels], tiny)), axis=0) > threshold)).any(axis=1)
    bad_samples = invalid.copy()
    for (start, block), bad in zip(blocks, window_bad):
        if bad:
            bad_samples[start:start + block.shape[1]] = True
    return bad_channels, bad_samples


def _fir_filter(values, frequencies, taps, sfreq):
    taps = int(taps)
    if taps < 3 or taps % 2 != 1:
        raise ValueError('FIR tap counts must be odd integers of at least three.')
    if values.shape[1] <= 3 * (taps - 1):
        raise ValueError(f'Recording requires more than {3 * (taps - 1)} samples for zero-phase filtering.')
    coefficients = signal.firwin(taps, frequencies, pass_zero=False, fs=sfreq, window='hamming')
    return signal.filtfilt(coefficients, [1.0], values, axis=-1, padlen=3 * (taps - 1))


def _robust_scale(values):
    median = np.median(values, axis=1, keepdims=True)
    q25, q75 = np.percentile(values, [25, 75], axis=1, keepdims=True)
    scale = q75 - q25
    return np.divide(values - median, scale, out=np.zeros_like(values), where=scale > np.finfo(float).eps)


def preprocess_recording(eeg, emg, sfreq=500, channel_names=None, config=None):
    config = {} if config is None else config
    sfreq = float(sfreq)
    if not np.isfinite(sfreq) or sfreq <= 200:
        raise ValueError('The input sampling rate must exceed 200 Hz.')
    eeg, eeg_invalid = _repair(eeg)
    emg, emg_invalid = _repair(emg)
    if eeg.shape[0] != 63 or emg.shape[0] != 2 or eeg.shape[1] != emg.shape[1]:
        raise ValueError('Recordings must contain 63 EEG and two EMG channels with identical sample counts.')
    if channel_names is None or len(channel_names) != 63 or len(set(channel_names)) != 63:
        raise ValueError('Exactly 63 distinct EEG channel names are required.')
    names = [str(name) for name in channel_names]
    threshold = float(config.get('quality_threshold', 6.0))
    if threshold <= 0:
        raise ValueError('quality_threshold must be positive.')
    eeg = signal.detrend(eeg, axis=-1, type='linear')
    bad_channels, bad_samples = _quality(eeg, eeg_invalid | emg_invalid, sfreq, threshold)
    eeg = _fir_filter(eeg, [0.5, 40.0], config.get('eeg_taps', 1501), sfreq)
    eeg -= eeg[~bad_channels].mean(axis=0, keepdims=True)
    eeg -= eeg.mean(axis=1, keepdims=True)
    quality = {
        'input_sfreq': sfreq,
        'output_sfreq': 200,
        'bad_channels': [names[i] for i in np.flatnonzero(bad_channels)],
        'bad_intervals': _intervals(bad_samples, sfreq),
        'eeg_repaired_samples': int(eeg_invalid.sum()),
        'emg_repaired_samples': int(emg_invalid.sum()),
        'ica_applied': bool(config.get('ica', True)),
        'ica_excluded_components': [],
    }
    if bad_channels.any() or config.get('ica', True):
        import mne
        raw = mne.io.RawArray(eeg, mne.create_info(names, sfreq, ch_types='eeg'), verbose=False)
        raw.info['bads'] = quality['bad_channels']
        if quality['bad_intervals']:
            onset = [item[0] for item in quality['bad_intervals']]
            duration = [item[1] - item[0] for item in quality['bad_intervals']]
            raw.set_annotations(mne.Annotations(onset, duration, ['BAD_quality'] * len(onset)))
        if config.get('ica', True):
            frontal = [name for name in ('Fp1', 'Fp2', 'AF7', 'AF8') if name in names and name not in raw.info['bads']]
            if not frontal:
                raise ValueError('ICA requires a usable frontal channel named Fp1, Fp2, AF7, or AF8.')
            if (~bad_samples).sum() < sfreq * 5:
                raise ValueError('ICA requires at least five seconds of usable continuous EEG.')
            ica = mne.preprocessing.ICA(n_components=0.99, method='fastica',
                                        random_state=int(config.get('seed', 2026)), max_iter=1000)
            ica.fit(raw, picks='eeg', reject_by_annotation=True, verbose=False)
            excluded = set()
            for name in frontal:
                indices, _ = ica.find_bads_eog(raw, ch_name=name, reject_by_annotation=True, verbose=False)
                excluded.update(indices)
            ica.exclude = sorted(excluded)
            ica.apply(raw, verbose=False)
            quality['ica_excluded_components'] = ica.exclude
        if bad_channels.any():
            raw.set_montage('standard_1020', on_missing='raise', verbose=False)
            raw.interpolate_bads(reset_bads=True, verbose=False)
        eeg = raw.get_data()
    emg = signal.detrend(emg, axis=-1, type='linear')
    emg = _fir_filter(emg, [20.0, 100.0], config.get('emg_taps', 331), sfreq)
    notch_frequency = float(config.get('notch_frequency', 50.0))
    b, a = signal.iirnotch(notch_frequency, float(config.get('notch_q', 30.0)), fs=sfreq)
    emg = signal.filtfilt(b, a, emg, axis=-1)
    ratio = Fraction(200 / sfreq).limit_denominator(10000)
    if not np.isclose(sfreq * ratio.numerator / ratio.denominator, 200):
        raise ValueError('The sampling rate cannot be accurately resampled to 200 Hz.')
    eeg = signal.resample_poly(_robust_scale(eeg), ratio.numerator, ratio.denominator, axis=-1)
    emg = signal.resample_poly(_robust_scale(emg), ratio.numerator, ratio.denominator, axis=-1)
    return eeg.astype(np.float32), emg.astype(np.float32), quality


def log_mel(audio, sample_rate=48000):
    import librosa
    audio = np.asarray(audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or not np.isfinite(audio).all() or not len(audio):
        raise ValueError('Audio must be a nonempty finite mono or sample-by-channel waveform.')
    if int(sample_rate) != 48000:
        raise ValueError('Overt audio must use the paper sampling rate of 48000 Hz.')
    audio = audio.astype(np.float64)
    audio = np.pad(audio[:144000], (0, max(0, 144000 - len(audio))))
    mel = librosa.feature.melspectrogram(y=audio, sr=48000, n_fft=1024, hop_length=480,
                                        n_mels=64, fmin=50, fmax=8000, power=2.0,
                                        center=True, pad_mode='constant')
    values = librosa.power_to_db(mel, ref=1.0)
    scale = values.std()
    values = (values - values.mean()) / scale if scale > np.finfo(float).eps else np.zeros_like(values)
    if values.shape != (64, 301):
        raise ValueError(f'Unexpected audio representation shape: {values.shape}.')
    return values[None].astype(np.float32)


def _waveform(path):
    sfreq, values = wavfile.read(path)
    if np.issubdtype(values.dtype, np.signedinteger):
        values = values.astype(np.float64) / max(abs(np.iinfo(values.dtype).min), np.iinfo(values.dtype).max)
    elif np.issubdtype(values.dtype, np.unsignedinteger):
        info = np.iinfo(values.dtype)
        center = (int(info.max) + 1) / 2
        values = (values.astype(np.float64) - center) / center
    return values, sfreq


def prepare_manifest(manifest_path, output_path, config=None):
    config = {} if config is None else config
    manifest_path = Path(manifest_path).resolve()
    with manifest_path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        required = {'subject', 'run', 'trial', 'label', 'mode', 'phase', 'onset', 'recording', 'audio'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Manifest requires columns: {", ".join(sorted(required))}.')
        rows = list(reader)
    rows = [row for row in rows if row['phase'] in ('overt', 'covert')]
    if not rows:
        raise ValueError('Manifest contains no overt or covert trials.')
    grouped = {}
    seen = set()
    for row in rows:
        if row['label'] not in LABELS or row['mode'] not in ('text', 'image'):
            raise ValueError('Trial labels and stimulus modalities must match the documented schema.')
        key = tuple(row[key] for key in ('subject', 'run', 'trial', 'label', 'mode', 'phase'))
        if any(not row[key] for key in ('subject', 'run', 'trial', 'recording')) or key in seen:
            raise ValueError('Trials require nonempty identifiers and unique subject/run/trial/label/mode/phase keys.')
        seen.add(key)
        onset = float(row['onset'])
        if not np.isfinite(onset) or onset < 0:
            raise ValueError('Trial onsets must be finite, nonnegative seconds.')
        if row['phase'] == 'overt' and not row['audio']:
            raise ValueError('Every overt trial requires an aligned trial-specific WAV file.')
        recording = (manifest_path.parent / row['recording']).resolve()
        grouped.setdefault(recording, []).append(row)
    prepared = {key: [] for key in ('x', 'audio', 'y', 'phase', 'subject', 'run', 'trial', 'mode')}
    quality_records = []
    excluded = []
    canonical_names = None
    for recording, markers in grouped.items():
        with np.load(recording, allow_pickle=False) as archive:
            required = {'eeg', 'emg', 'sfreq', 'channel_names'}
            if not required.issubset(archive.files):
                raise ValueError(f'Recording {recording.name} is missing required arrays.')
            names = archive['channel_names'].astype(str).tolist()
            if canonical_names is None:
                canonical_names = names
            elif names != canonical_names:
                raise ValueError('All recordings must use the same EEG channel names in the same order.')
            eeg, emg, quality = preprocess_recording(archive['eeg'], archive['emg'],
                float(archive['sfreq'].item()), names, config)
        owners = {(row['subject'], row['run']) for row in markers}
        if len(owners) != 1:
            raise ValueError('Each continuous recording must belong to exactly one subject and run.')
        quality_records.append({'recording': str(recording), **quality})
        for row in markers:
            onset = float(row['onset'])
            start = int(round(onset * 200))
            stop = start + 600
            if stop > eeg.shape[1]:
                raise ValueError(f'Trial {row["trial"]} extends beyond its continuous recording.')
            if any(onset < end and onset + 3 > begin for begin, end in quality['bad_intervals']):
                excluded.append({key: row[key] for key in ('subject', 'run', 'trial', 'phase')})
                continue
            if row['phase'] == 'overt':
                waveform, audio_sfreq = _waveform((manifest_path.parent / row['audio']).resolve())
                spectrogram = log_mel(waveform, audio_sfreq)
            else:
                spectrogram = np.zeros((1, 64, 301), dtype=np.float32)
            prepared['x'].append(np.concatenate((eeg[:, start:stop], emg[:, start:stop])))
            prepared['audio'].append(spectrogram)
            prepared['y'].append(LABELS.index(row['label']))
            for key in ('phase', 'subject', 'run', 'trial', 'mode'):
                prepared[key].append(row[key])
    if not prepared['x']:
        raise ValueError('No usable trials remain after signal-quality exclusions.')
    arrays = {key: np.asarray(values, dtype=np.float32 if key in ('x', 'audio') else np.int64 if key == 'y' else str)
              for key, values in prepared.items()}
    output_path = Path(output_path)
    if output_path.suffix.lower() != '.npz':
        raise ValueError('The prepared dataset output must use the .npz extension.')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    return {'trials': len(arrays['y']), 'excluded_trials': excluded, 'recordings': quality_records,
            'output': str(output_path.resolve())}
