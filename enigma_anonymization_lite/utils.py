#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 15:37:59 2023

@author: nugenta
"""
from collections import OrderedDict
from pkg_resources import parse_version
import numpy as np
import mne
from mne import Epochs
from mne.io.constants import FIFF
from mne.io import BaseRaw
from mne.channels.channels import  _get_meg_system
from mne.chpi import get_chpi_info
from mne.utils import logger
import mne.preprocessing
from mne_bids.utils import _write_json, _infer_eeg_placement_scheme
from mne_bids.config import IGNORED_CHANNELS


def _sidecar_json_patched(raw, task, manufacturer, fname, datatype,
                  emptyroom_fname=None, overwrite=False):
    """Create a sidecar json file depending on the suffix and save it.
    The sidecar json file provides meta data about the data
    of a certain datatype.
    Parameters
    ----------
    raw : mne.io.Raw
        The data as MNE-Python Raw object.
    task : str
        Name of the task the data is based on.
    manufacturer : str
        Manufacturer of the acquisition system. For MEG also used to define the
        coordinate system for the MEG sensors.
    fname : str | mne_bids.BIDSPath
        Filename to save the sidecar json to.
    datatype : str
        Type of the data as in ALLOWED_ELECTROPHYSIO_DATATYPE.
    emptyroom_fname : str | mne_bids.BIDSPath
        For MEG recordings, the path to an empty-room data file to be
        associated with ``raw``. Only supported for MEG.
    overwrite : bool
        Whether to overwrite the existing file.
        Defaults to False.
    """
    sfreq = raw.info['sfreq']
    try:
        powerlinefrequency = raw.info['line_freq']
        powerlinefrequency = ('n/a' if powerlinefrequency is None else
                              powerlinefrequency)
    except KeyError:
        raise ValueError(
            "PowerLineFrequency parameter is required in the sidecar files. "
            "Please specify it in info['line_freq'] before saving to BIDS, "
            "e.g. by running: "
            "    raw.info['line_freq'] = 60"
            "in your script, or by passing: "
            "    --line_freq 60 "
            "in the command line for a 60 Hz line frequency. If the frequency "
            "is unknown, set it to None")

    if isinstance(raw, BaseRaw):
        rec_type = 'continuous'
    elif isinstance(raw, Epochs):
        rec_type = 'epoched'
    else:
        rec_type = 'n/a'

    # determine whether any channels have to be ignored:
    n_ignored = len([ch_name for ch_name in
                     IGNORED_CHANNELS.get(manufacturer, list()) if
                     ch_name in raw.ch_names])
    # all ignored channels are trigger channels at the moment...

    n_megchan = len([ch for ch in raw.info['chs']
                     if ch['kind'] == FIFF.FIFFV_MEG_CH])
    n_megrefchan = len([ch for ch in raw.info['chs']
                        if ch['kind'] == FIFF.FIFFV_REF_MEG_CH])
    n_eegchan = len([ch for ch in raw.info['chs']
                     if ch['kind'] == FIFF.FIFFV_EEG_CH])
    n_ecogchan = len([ch for ch in raw.info['chs']
                      if ch['kind'] == FIFF.FIFFV_ECOG_CH])
    n_seegchan = len([ch for ch in raw.info['chs']
                      if ch['kind'] == FIFF.FIFFV_SEEG_CH])
    n_eogchan = len([ch for ch in raw.info['chs']
                     if ch['kind'] == FIFF.FIFFV_EOG_CH])
    n_ecgchan = len([ch for ch in raw.info['chs']
                     if ch['kind'] == FIFF.FIFFV_ECG_CH])
    n_emgchan = len([ch for ch in raw.info['chs']
                     if ch['kind'] == FIFF.FIFFV_EMG_CH])
    n_miscchan = len([ch for ch in raw.info['chs']
                      if ch['kind'] == FIFF.FIFFV_MISC_CH])
    n_stimchan = len([ch for ch in raw.info['chs']
                      if ch['kind'] == FIFF.FIFFV_STIM_CH]) - n_ignored
    n_dbschan = len([ch for ch in raw.info['chs']
                     if ch['kind'] == FIFF.FIFFV_DBS_CH])
    nirs_channels = [ch for ch in raw.info['chs'] if
                     ch['kind'] == FIFF.FIFFV_FNIRS_CH]
    n_nirscwchan = len(nirs_channels)
    n_nirscwsrc = len(np.unique([ch["ch_name"].split(" ")[0].split("_")[0]
                                 for ch in nirs_channels]))
    n_nirscwdet = len(np.unique([ch["ch_name"].split(" ")[0].split("_")[1]
                                 for ch in nirs_channels]))

    # Set DigitizedLandmarks to True if any of LPA, RPA, NAS are found
    # Set DigitizedHeadPoints to True if any "Extra" points are found
    # (DigitizedHeadPoints done for Neuromag MEG files only)
    digitized_head_points = False
    digitized_landmark = False
    if datatype == 'meg' and raw.info['dig'] is not None:
        for dig_point in raw.info['dig']:
            if dig_point['kind'] in [FIFF.FIFFV_POINT_NASION,
                                     FIFF.FIFFV_POINT_RPA,
                                     FIFF.FIFFV_POINT_LPA]:
                digitized_landmark = True
            elif dig_point['kind'] == FIFF.FIFFV_POINT_EXTRA and \
                    raw.filenames[0].endswith('.fif'):
                digitized_head_points = True
    software_filters = {
        'SpatialCompensation': {
            'GradientOrder': raw.compensation_grade
        }
    }

    # Compile cHPI information, if any.
    system, _ = _get_meg_system(raw.info)
    chpi = None
    hpi_freqs = []
    if (datatype == 'meg' and
            parse_version(mne.__version__) > parse_version('0.23')):
        # We need to handle different data formats differently
        if system == 'CTF_275':
            try:
                mne.chpi.extract_chpi_locs_ctf(raw)
                chpi = True
            except RuntimeError:
                chpi = False
                logger.info('Could not find cHPI information in raw data.')
        elif system == 'KIT':
            try:
                mne.chpi.extract_chpi_locs_kit(raw)
                chpi = True
            except (RuntimeError, ValueError, TypeError):
                chpi = False
                logger.info('Could not find cHPI information in raw data, or you imported a KIT file as FIF.')
        elif system in ['122m', '306m']:
            n_active_hpi = mne.chpi.get_active_chpi(raw, on_missing='ignore')
            chpi = bool(n_active_hpi.sum() > 0)
            if chpi:
                hpi_freqs, _, _ = get_chpi_info(info=raw.info,
                                                on_missing='ignore')
                hpi_freqs = list(hpi_freqs)

    elif datatype == 'meg':
        logger.info('Cannot check for & write continuous head localization '
                    'information: requires MNE-Python >= 0.24')

    # Define datatype-specific JSON dictionaries
    ch_info_json_common = [
        ('TaskName', task),
        ('Manufacturer', manufacturer),
        ('PowerLineFrequency', powerlinefrequency),
        ('SamplingFrequency', sfreq),
        ('SoftwareFilters', 'n/a'),
        ('RecordingDuration', raw.times[-1]),
        ('RecordingType', rec_type)]

    ch_info_json_meg = [
        ('DewarPosition', 'n/a'),
        ('DigitizedLandmarks', digitized_landmark),
        ('DigitizedHeadPoints', digitized_head_points),
        ('MEGChannelCount', n_megchan),
        ('MEGREFChannelCount', n_megrefchan),
        ('SoftwareFilters', software_filters)]

    if chpi is not None:
        ch_info_json_meg.append(('ContinuousHeadLocalization', chpi))
        ch_info_json_meg.append(('HeadCoilFrequency', hpi_freqs))

    if emptyroom_fname is not None:
        ch_info_json_meg.append(('AssociatedEmptyRoom', str(emptyroom_fname)))

    ch_info_json_eeg = [
        ('EEGReference', 'n/a'),
        ('EEGGround', 'n/a'),
        ('EEGPlacementScheme', _infer_eeg_placement_scheme(raw)),
        ('Manufacturer', manufacturer)]

    ch_info_json_ieeg = [
        ('iEEGReference', 'n/a'),
        ('ECOGChannelCount', n_ecogchan),
        ('SEEGChannelCount', n_seegchan + n_dbschan)]

    ch_info_json_nirs = [
        ('Manufacturer', manufacturer)
    ]

    ch_info_ch_counts = [
        ('EEGChannelCount', n_eegchan),
        ('EOGChannelCount', n_eogchan),
        ('ECGChannelCount', n_ecgchan),
        ('EMGChannelCount', n_emgchan),
        ('MiscChannelCount', n_miscchan),
        ('TriggerChannelCount', n_stimchan)]

    ch_info_ch_counts_nirs = [
        ('NIRSChannelCount', n_nirscwchan),
        ('NIRSSourceOptodeCount', n_nirscwsrc),
        ('NIRSDetectorOptodeCount', n_nirscwdet)
    ]

    # Stitch together the complete JSON dictionary
    ch_info_json = ch_info_json_common
    if datatype == 'meg':
        append_datatype_json = ch_info_json_meg
    elif datatype == 'eeg':
        append_datatype_json = ch_info_json_eeg
    elif datatype == 'ieeg':
        append_datatype_json = ch_info_json_ieeg
    elif datatype == 'nirs':
        append_datatype_json = ch_info_json_nirs
        ch_info_ch_counts.extend(ch_info_ch_counts_nirs)

    ch_info_json += append_datatype_json
    ch_info_json += ch_info_ch_counts
    ch_info_json = OrderedDict(ch_info_json)

    _write_json(fname, ch_info_json, overwrite)

    return fname