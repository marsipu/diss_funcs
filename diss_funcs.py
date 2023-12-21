import itertools
import re
import sys
from collections import OrderedDict
from functools import reduce
from itertools import combinations
from os import mkdir
from os.path import isdir, join

import mne
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.pyplot import colormaps
from mne.minimum_norm import source_induced_power
from mne.stats import (
    permutation_cluster_test,
    permutation_cluster_1samp_test,
    bonferroni_correction,
)
from mne_pipeline_hd.functions.operations import calculate_gfp, find_6ch_binary_events
from mne_pipeline_hd.pipeline.loading import Group
from mne_pipeline_hd.pipeline.loading import MEEG
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import ttest_1samp
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from statannotations.Annotator import Annotator

figsize = [9, 3]


##############################################################
# Preparation
##############################################################


def combine_labels(fsmri, label_combinations):
    """This combines existing labels to bigger labels with a new name."""
    for new_label_name, label_names in label_combinations.items():
        labels = fsmri.get_labels(label_names)
        new_label = labels[0]
        for i in range(1, len(labels)):
            new_label += labels[i]
        new_label.name = new_label_name
        # Overwrites per default
        new_label.save(
            join(fsmri.subjects_dir, fsmri.name, "label", f"{new_label_name}.label")
        )
        print(f"Created new label: {new_label_name}")


def change_trig_channel_type(meeg, trig_channel):
    """Changes the channel-type of the Load-Cell-channel to 'stim'."""
    raw = meeg.load_raw()

    if trig_channel in raw.ch_names:
        if raw.get_channel_types(trig_channel) != "stim":
            print(f"Changing {trig_channel} to stim")
            raw.set_channel_types({trig_channel: "stim"})

            if trig_channel in raw.ch_names:
                raw.rename_channels({trig_channel: "LoadCell"})
                print(f"{meeg.name}: Rename Trigger-Channel")

            if trig_channel in meeg.bad_channels:
                meeg.bad_channels.remove(trig_channel)
                print(f"{meeg.name}: Removed Trigger-Channel from bad_channels")

            if trig_channel in raw.info["bads"]:
                raw.info["bads"].remove(trig_channel)
                print(f'{meeg.name}: Removed Trigger-Channel from info["bads"]')

            meeg.save_raw(raw)
        else:
            print(f"{trig_channel} already stim")
    else:
        print(f"{trig_channel} not in raw")


def rereference_eog(meeg, eog_tuple, eogecg_target):
    """EEG-channels are rereferenced to construct a single EOG-channel."""
    if eogecg_target == "Raw (Unfiltered)":
        raw = meeg.load_raw()
    else:
        raw = meeg.load_filtered()

    # Remove old channels
    for old_ch_name in [f"EOG BP{idx}" for idx in range(10)]:
        if old_ch_name in raw.ch_names:
            raw = raw.drop_channels(old_ch_name)
            print(f"Dropped existing channel: {old_ch_name}")

    # Set Bipolar reference
    ch_name = f"EOG BP"
    if ch_name in raw.ch_names:
        raw = raw.drop_channels(ch_name)
        print(f"Dropped existing channel: {ch_name}")

    mne.set_bipolar_reference(
        raw, eog_tuple[0], eog_tuple[1], ch_name=ch_name, drop_refs=False, copy=False
    )
    raw.set_channel_types({ch_name: "eog"})

    if eogecg_target == "Raw (Unfiltered)":
        meeg.save_raw(raw)
    else:
        meeg.save_filtered(raw)


def get_dig_eegs(meeg, n_eeg_channels, eeg_dig_first=True):
    """
    Function to get EEG-Montage from digitized EEG-Electrodes
    (without them having be labeled as EEG during Digitization)

    Notes
    -----
    By Laura Doll, adapted by Martin Schulz
    """
    raw = meeg.load_raw()

    if 3 not in set([int(d["kind"]) for d in raw.info["dig"]]):
        ch_pos = dict()
        hsp = None
        all_extra_points = [dp for dp in raw.info["dig"] if int(dp["kind"]) == 4]
        if eeg_dig_first:
            eeg_points = all_extra_points[:n_eeg_channels]
        else:
            eeg_points = all_extra_points[-n_eeg_channels:]
        for dp in eeg_points:
            ch_pos[f'EEG {dp["ident"]:03}'] = dp["r"]

        hsp_points = [dp["r"] for dp in all_extra_points]

        if len(hsp_points) > 0:
            hsp = np.asarray(hsp_points)

        lpa = [
            dp["r"]
            for dp in raw.info["dig"]
            if int(dp["kind"]) == 1 and dp["ident"] == 1
        ][0]
        nasion = [
            dp["r"]
            for dp in raw.info["dig"]
            if int(dp["kind"]) == 1 and dp["ident"] == 2
        ][0]
        rpa = [
            dp["r"]
            for dp in raw.info["dig"]
            if int(dp["kind"]) == 1 and dp["ident"] == 3
        ][0]

        hpi = np.asarray([dp["r"] for dp in raw.info["dig"] if int(dp["kind"]) == 2])

        montage = mne.channels.make_dig_montage(ch_pos, nasion, lpa, rpa, hsp, hpi)

        print(
            f"Added {n_eeg_channels} EEG-Channels to montage, "
            f"{len(all_extra_points) - n_eeg_channels} Head-Shape-Points remaining"
        )

        raw.set_montage(montage, on_missing="raise")
    else:
        print("EEG channels already added here")

    meeg.save_raw(raw)


def pl1_laser_event_correction(meeg):
    """A mistake during data-acquistion of the first subject of Experiment 2 is corrected."""
    if meeg.name != "pl1_laser2J":
        print("Only for pl1_laser2J")
        return
    events = meeg.load_events()
    # Change id from 11 to 32
    events[events[:, 2] == 11, 2] = 32
    # Determined 1142 ms mean latency between 32 and 11 from other laser-measurements
    events[events[:, 2] == 32, 0] -= 1142
    meeg.save_events(events)


##############################################################
# Ratings
##############################################################
def get_ratings(meeg, target_event_id):
    """Ratings encoded into trigger-signals 10-19 are decoded and saved as .csv-file."""
    events = meeg.load_events()

    file_name = "ratings_meta"
    file_path = join(meeg.save_dir, f"{meeg.name}_{meeg.p_preset}_{file_name}.csv")
    rating_meta_pd = pd.DataFrame([], columns=["time", "id", "rating"], dtype=int)

    # Get Ratings from Triggers 10-19
    pre_ratings = np.copy(
        events[np.nonzero(np.logical_and(10 <= events[:, 2], events[:, 2] <= 19))]
    )
    first_idx = np.nonzero(np.diff(pre_ratings[:, 0], axis=0) < 200)[0]
    last_idx = first_idx + 1
    ratings = pre_ratings[first_idx]
    ratings[:, 2] = (ratings[:, 2] - 10) * 10 + pre_ratings[last_idx][:, 2] - 10

    # Get time sample from target_event_id
    target_events = events[np.nonzero(events[:, 2] == target_event_id)]
    for rating in ratings:
        # Get time from previous target_event_id
        try:
            rating_time = target_events[
                np.nonzero(target_events[:, 0] - rating[0] < 0)
            ][-1][0]
        except IndexError:
            pass
        else:
            # Make sure there are no duplicates (because of missing events)
            if rating_time not in list(rating_meta_pd["time"]):
                rating_value = rating[2]
                rating_dict = {
                    "time": rating_time,
                    "id": target_event_id,
                    "rating": rating_value,
                }
                meta_series = pd.Series(rating_dict)
                rating_meta_pd = pd.concat(
                    [rating_meta_pd, meta_series.to_frame().T],
                    axis=0,
                    ignore_index=True,
                )
    print(f"Ratings: {list(rating_meta_pd['rating'])}")
    rating_meta_pd.to_csv(file_path)


def get_ratings_laser(meeg, laser_target_event_id):
    """Ratings encoded into trigger-signals 1-10 are decoded and saved as.csv-file."""
    events = meeg.load_events()

    file_name = "ratings_meta"
    file_path = join(meeg.save_dir, f"{meeg.name}_{meeg.p_preset}_{file_name}.csv")
    rating_meta_pd = pd.DataFrame([], columns=["time", "id", "rating"], dtype=int)

    # Get Ratings from Triggers 10-19
    pre_ratings = np.copy(
        events[np.nonzero(np.logical_and(1 <= events[:, 2], events[:, 2] <= 10))]
    )
    first_idx = np.nonzero(np.diff(pre_ratings[:, 0], axis=0) < 200)[0]
    last_idx = first_idx + 1
    ratings = pre_ratings[first_idx]
    ratings[:, 2] = (ratings[:, 2] - 1) * 10 + pre_ratings[last_idx][:, 2] - 1

    # Get time sample from target_event_id
    target_events = events[np.nonzero(events[:, 2] == laser_target_event_id)]
    for rating in ratings:
        # Get time from previous target_event_id
        try:
            rating_time = target_events[
                np.nonzero(target_events[:, 0] - rating[0] < 0)
            ][-1][0]
        except IndexError:
            pass
        else:
            # Make sure there are no duplicates (because of missing events)
            if rating_time not in list(rating_meta_pd["time"]):
                rating_value = rating[2]
                rating_dict = {
                    "time": rating_time,
                    "id": laser_target_event_id,
                    "rating": rating_value,
                }
                meta_series = pd.Series(rating_dict)
                rating_meta_pd = pd.concat(
                    [rating_meta_pd, meta_series.to_frame().T],
                    axis=0,
                    ignore_index=True,
                )
    print(f"Ratings: {list(rating_meta_pd['rating'])}")
    rating_meta_pd.to_csv(file_path)


def _add_events_meta(epochs, meta_pd, meta_key):
    """Make sure, that meat-data is assigned to correct epoch
    (requires parameter "time" and "id" to be included in meta_pd)
    """
    meta_pd_filtered = meta_pd.loc[
        meta_pd["id"].isin(epochs.event_id.values())
        & meta_pd["time"].isin(epochs.events[:, 0])
        ]

    metatimes = [int(t) for t in meta_pd_filtered["time"]]

    # Add missing values as NaN
    for miss_ix in np.nonzero(np.isin(epochs.events[:, 0], metatimes, invert=True))[0]:
        miss_time, miss_id = epochs.events[miss_ix, [0, 2]]
        meta_pd_filtered = pd.concat(
            [
                meta_pd_filtered,
                pd.Series({"time": miss_time, "id": miss_id, meta_key: np.nan})
                .to_frame()
                .T,
            ],
            axis=0,
            ignore_index=True,
        )

    meta_pd_filtered = meta_pd_filtered.sort_values(
        "time", ascending=True, ignore_index=True
    )

    # Integrate into existing metadata
    if isinstance(epochs.metadata, pd.DataFrame):
        meta_pd_filtered = pd.merge(epochs.metadata, meta_pd_filtered, how="inner")

    if len(meta_pd_filtered) > 0:
        epochs.metadata = meta_pd_filtered
    else:
        raise RuntimeWarning("No metadata fits to this epochs!")


def add_ratings_meta(meeg):
    """Ratings are added as metadata to epochs."""
    epochs = meeg.load_epochs()
    file_name = "ratings_meta"
    file_path = join(meeg.save_dir, f"{meeg.name}_{meeg.p_preset}_{file_name}.csv")
    ratings_pd = pd.read_csv(file_path, index_col=0)

    _add_events_meta(epochs, ratings_pd, "ratings")
    meeg.save_epochs(epochs)


def remove_metadata(meeg):
    """Metadata is removed from epochs."""
    epochs = meeg.load_epochs()
    epochs.metadata = None
    meeg.save_epochs(epochs)


##############################################################
# Events
##############################################################


def find_6ch_binary_events(meeg, min_duration, shortest_event, adjust_timeline_by_msec):
    raw = meeg.load_raw()  # No copy to consume less memory

    # Binary Coding of 6 Stim Channels in Biomagenetism Lab Heidelberg
    # prepare arrays
    events = np.ndarray(shape=(0, 3), dtype=np.int32)
    evs = list()
    evs_tol = list()

    # Find events for each stim channel, append sample values to list
    evs.append(
        mne.find_events(
            raw,
            min_duration=min_duration,
            shortest_event=shortest_event,
            stim_channel=["STI 001"],
        )[:, 0]
    )
    evs.append(
        mne.find_events(
            raw,
            min_duration=min_duration,
            shortest_event=shortest_event,
            stim_channel=["STI 002"],
        )[:, 0]
    )
    evs.append(
        mne.find_events(
            raw,
            min_duration=min_duration,
            shortest_event=shortest_event,
            stim_channel=["STI 003"],
        )[:, 0]
    )
    evs.append(
        mne.find_events(
            raw,
            min_duration=min_duration,
            shortest_event=shortest_event,
            stim_channel=["STI 004"],
        )[:, 0]
    )
    evs.append(
        mne.find_events(
            raw,
            min_duration=min_duration,
            shortest_event=shortest_event,
            stim_channel=["STI 005"],
        )[:, 0]
    )
    evs.append(
        mne.find_events(
            raw,
            min_duration=min_duration,
            shortest_event=shortest_event,
            stim_channel=["STI 006"],
        )[:, 0]
    )

    for i in evs:
        # delete events in each channel,
        # which are too close too each other (1ms)
        too_close = np.where(np.diff(i) <= 1)
        if np.size(too_close) >= 1:
            print(
                f"Two close events (1ms) at samples "
                f"{i[too_close] + raw.first_samp}, first deleted"
            )
            i = np.delete(i, too_close, 0)
            evs[evs.index(i)] = i

        # add tolerance to each value
        i_tol = np.ndarray(shape=(0, 1), dtype=np.int32)
        for t in i:
            i_tol = np.append(i_tol, t - 1)
            i_tol = np.append(i_tol, t)
            i_tol = np.append(i_tol, t + 1)

        evs_tol.append(i_tol)

    # Get events from combinated Stim-Channels
    equals = reduce(
        np.intersect1d,
        (evs_tol[0], evs_tol[1], evs_tol[2], evs_tol[3], evs_tol[4], evs_tol[5]),
    )
    # elimnate duplicated events
    too_close = np.where(np.diff(equals) <= 1)
    if np.size(too_close) >= 1:
        equals = np.delete(equals, too_close, 0)
        equals -= 1  # correction, because of shift with deletion

    for q in equals:
        if (
                q not in events[:, 0]
                and q not in events[:, 0] + 1
                and q not in events[:, 0] - 1
        ):
            events = np.append(events, [[q, 0, 63]], axis=0)

    for a, b, c, d, e in combinations(range(6), 5):
        equals = reduce(
            np.intersect1d, (evs_tol[a], evs_tol[b], evs_tol[c], evs_tol[d], evs_tol[e])
        )
        too_close = np.where(np.diff(equals) <= 1)
        if np.size(too_close) >= 1:
            equals = np.delete(equals, too_close, 0)
            equals -= 1

        for q in equals:
            if (
                    q not in events[:, 0]
                    and q not in events[:, 0] + 1
                    and q not in events[:, 0] - 1
            ):
                events = np.append(
                    events,
                    [[q, 0, int(2 ** a + 2 ** b + 2 ** c + 2 ** d + 2 ** e)]],
                    axis=0,
                )

    for a, b, c, d in combinations(range(6), 4):
        equals = reduce(
            np.intersect1d, (evs_tol[a], evs_tol[b], evs_tol[c], evs_tol[d])
        )
        too_close = np.where(np.diff(equals) <= 1)
        if np.size(too_close) >= 1:
            equals = np.delete(equals, too_close, 0)
            equals -= 1

        for q in equals:
            if (
                    q not in events[:, 0]
                    and q not in events[:, 0] + 1
                    and q not in events[:, 0] - 1
            ):
                events = np.append(
                    events, [[q, 0, int(2 ** a + 2 ** b + 2 ** c + 2 ** d)]], axis=0
                )

    for a, b, c in combinations(range(6), 3):
        equals = reduce(np.intersect1d, (evs_tol[a], evs_tol[b], evs_tol[c]))
        too_close = np.where(np.diff(equals) <= 1)
        if np.size(too_close) >= 1:
            equals = np.delete(equals, too_close, 0)
            equals -= 1

        for q in equals:
            if (
                    q not in events[:, 0]
                    and q not in events[:, 0] + 1
                    and q not in events[:, 0] - 1
            ):
                events = np.append(
                    events, [[q, 0, int(2 ** a + 2 ** b + 2 ** c)]], axis=0
                )

    for a, b in combinations(range(6), 2):
        equals = np.intersect1d(evs_tol[a], evs_tol[b])
        too_close = np.where(np.diff(equals) <= 1)
        if np.size(too_close) >= 1:
            equals = np.delete(equals, too_close, 0)
            equals -= 1

        for q in equals:
            if (
                    q not in events[:, 0]
                    and q not in events[:, 0] + 1
                    and q not in events[:, 0] - 1
            ):
                events = np.append(events, [[q, 0, int(2 ** a + 2 ** b)]], axis=0)

    # Get single-channel events
    for i in range(6):
        for e in evs[i]:
            if (
                    e not in events[:, 0]
                    and e not in events[:, 0] + 1
                    and e not in events[:, 0] - 1
            ):
                events = np.append(events, [[e, 0, 2 ** i]], axis=0)

    # sort only along samples(column 0)
    events = events[events[:, 0].argsort()]

    # apply latency correction
    events[:, 0] = [
        ts + np.round(adjust_timeline_by_msec * 10 ** -3 * raw.info["sfreq"])
        for ts in events[:, 0]
    ]

    ids = np.unique(events[:, 2])
    print("unique ID's found: ", ids)

    if np.size(events) > 0:
        meeg.save_events(events)
    else:
        print("No events found")


def get_load_cell_events_regression_baseline(
        meeg,
        min_duration,
        shortest_event,
        adjust_timeline_by_msec,
        diff_window,
        min_ev_distance,
        max_ev_distance,
        len_baseline,
        baseline_limit,
        regression_degree,
        trig_channel,
        n_jobs,
):
    """The events are extracted from the load-cell signal using a rolling-difference, baseline and regression."""
    # Load Raw and extract the load-cell-trigger-channel
    raw = meeg.load_raw()
    if trig_channel not in raw.ch_names:
        print(f"Channel {trig_channel} not found in {meeg.name}")
        return

    eeg_raw = raw.copy().pick(trig_channel)
    eeg_series = eeg_raw.to_data_frame()[trig_channel]

    # Difference of Rolling Mean on both sides of each value
    rolling_left = eeg_series.rolling(diff_window, min_periods=1).mean()
    rolling_right = eeg_series.iloc[::-1].rolling(diff_window, min_periods=1).mean()
    rolling_diff = rolling_left - rolling_right

    # Find peaks of the Rolling-Difference
    rd_peaks, _ = find_peaks(
        abs(rolling_diff), height=np.std(rolling_diff), distance=min_ev_distance
    )

    try:
        # Find the other events encoded by the binary channel
        find_6ch_binary_events(
            meeg, min_duration, shortest_event, adjust_timeline_by_msec
        )
        events = meeg.load_events()
    except ValueError:
        events = np.asarray([[0, 0, 0]])

    events_meta_dict = dict()
    events_meta_pd = pd.DataFrame([])

    # Iterate through the peaks found in the rolling difference
    for ev_idx, pk in enumerate(rd_peaks):
        if ev_idx != len(rd_peaks) - 1:
            sys.stderr.write(f"\rProgress: {int((ev_idx + 1) / len(rd_peaks) * 100)} %")
        else:
            sys.stderr.write(
                f"\rProgress: {int((ev_idx + 1) / len(rd_peaks) * 100)} %\n"
            )

        # Get closest peak to determine down or up

        # Get first trigger if up follows in under min_ev_distance
        if ev_idx == 0:
            if rd_peaks[1] - pk < max_ev_distance:
                direction = "down"
            else:
                continue

        # Get last trigger if down was before under min_ev_distance
        elif ev_idx == len(rd_peaks) - 1:
            if pk - rd_peaks[ev_idx - 1] < max_ev_distance:
                direction = "up"
            else:
                continue

        # Get other peaks
        elif rd_peaks[ev_idx + 1] - pk < max_ev_distance:
            direction = "down"

        elif pk - rd_peaks[ev_idx - 1] < max_ev_distance:
            direction = "up"

        else:
            continue

        # Get Trigger-Time by finding the first samples going from peak crossing the baseline
        # (from baseline_limit with length=len_baseline)
        pre_baseline_mean = np.asarray(
            eeg_series[pk - (len_baseline + baseline_limit): pk - baseline_limit + 1]
        ).mean()
        post_baseline_mean = np.asarray(
            eeg_series[pk + baseline_limit: pk + baseline_limit + len_baseline + 1]
        ).mean()
        pre_peak_data = np.flip(np.asarray(eeg_series[pk - min_ev_distance: pk + 1]))
        post_peak_data = np.asarray(eeg_series[pk: pk + min_ev_distance + 1])
        if pre_baseline_mean > post_baseline_mean:
            first_idx = pk - (pre_peak_data > pre_baseline_mean).argmax()
            last_idx = pk + (post_peak_data < post_baseline_mean).argmax()
        else:
            first_idx = pk - (pre_peak_data < pre_baseline_mean).argmax()
            last_idx = pk + (post_peak_data > post_baseline_mean).argmax()

        # Do the regression for the data spanned by first-time and last-time
        y = eeg_series[first_idx:last_idx]
        x = PolynomialFeatures(
            degree=regression_degree, include_bias=False
        ).fit_transform(np.arange(len(y)).reshape(-1, 1))
        model = LinearRegression(n_jobs=n_jobs).fit(x, y)

        score = model.score(x, y)
        y_pred = model.predict(x)
        coef = model.coef_

        peak_time = pk + raw.first_samp
        first_time = first_idx + raw.first_samp
        last_time = last_idx + raw.first_samp

        # Store information about event in meta-dict
        events_meta_dict[ev_idx] = dict()
        events_meta_dict[ev_idx]["best_score"] = score
        events_meta_dict[ev_idx]["first_time"] = first_time
        events_meta_dict[ev_idx]["peak_time"] = peak_time
        events_meta_dict[ev_idx]["last_time"] = last_time
        events_meta_dict[ev_idx]["y_pred"] = y_pred
        events_meta_dict[ev_idx]["coef"] = coef
        events_meta_dict[ev_idx]["direction"] = direction

        # Event-ID-naming:
        #   4 = Down-First
        #   5 = Down-Middle
        #   6 = Down-Last
        #
        #   7 = Up-First
        #   8 = Up-Middle
        #   9 = Up-Last

        if direction == "down":
            event_id_first = 4
            event_id_middle = 5
            event_id_last = 6
        else:
            event_id_first = 7
            event_id_middle = 8
            event_id_last = 9

        for timep, evid in zip(
                [first_time, peak_time, last_time],
                [event_id_first, event_id_middle, event_id_last],
        ):
            meta_dict = {
                "time": timep,
                "score": score,
                "direction": direction,
                "id": evid,
                "coef": coef[0],
            }
            # coef_dict = {k: v for k, v in zip(list(ascii_lowercase)[:len(coef)], coef)}
            # meta_dict.update(coef_dict)
            new_meta = pd.DataFrame([meta_dict], columns=meta_dict.keys())
            events_meta_pd = pd.concat([events_meta_pd, new_meta], ignore_index=True)

        # add to events
        events = np.append(events, [[first_time, 0, event_id_first]], axis=0)
        events = np.append(events, [[peak_time, 0, event_id_middle]], axis=0)
        events = np.append(events, [[last_time, 0, event_id_last]], axis=0)

    # sort events by time (first column)
    events = events[events[:, 0].argsort()]

    # Remove duplicates
    while len(events[:, 0]) != len(np.unique(events[:, 0])):
        uniques, inverse, counts = np.unique(
            events[:, 0], return_inverse=True, return_counts=True
        )
        duplicates = uniques[np.nonzero(counts != 1)]

        for dpl in duplicates:
            events = np.delete(events, np.nonzero(events[:, 0] == dpl)[0][0], axis=0)
            print(f"Removed duplicate at {dpl}")

    print(f"Found {len(np.nonzero(events[:, 2] == 4)[0])} Events for Down-First")
    print(f"Found {len(np.nonzero(events[:, 2] == 5)[0])} Events for Down-Middle")
    print(f"Found {len(np.nonzero(events[:, 2] == 6)[0])} Events for Down-Last")
    print(f"Found {len(np.nonzero(events[:, 2] == 7)[0])} Events for Up-First")
    print(f"Found {len(np.nonzero(events[:, 2] == 8)[0])} Events for Up-Middle")
    print(f"Found {len(np.nonzero(events[:, 2] == 9)[0])} Events for Up-Last")

    # Save events
    meeg.save_events(events)

    # Save events-meta dictionary
    meeg.save_json("load_events_meta", events_meta_dict)

    # Save events-meta DataFrame
    file_name = "load_events_meta"
    file_path = join(meeg.save_dir, f"{meeg.name}_{meeg.p_preset}_{file_name}.csv")
    events_meta_pd["time"] = events_meta_pd["time"].astype(int)
    events_meta_pd.to_csv(file_path)

    # Save Trigger-Raw with correlation-signal for plotting
    reg_signal = np.asarray([])
    for idx, ev_idx in enumerate(events_meta_dict):
        first_time = events_meta_dict[ev_idx]["first_time"] - eeg_raw.first_samp
        best_y = events_meta_dict[ev_idx]["y_pred"]

        if idx == 0:
            # Fill the time before the first event
            reg_signal = np.concatenate(
                [reg_signal, np.full(first_time, best_y[0]), best_y]
            )
        else:
            # Get previous index even when it is missing
            n_minus = 1
            previous_idx = None
            while True:
                try:
                    events_meta_dict[ev_idx - n_minus]
                except KeyError:
                    n_minus += 1
                    if ev_idx - n_minus < 0:
                        break
                else:
                    previous_idx = ev_idx - n_minus
                    break
            if idx == len(events_meta_dict) - 1:
                # Fill the time before and after the last event
                first_fill_time = first_time - (
                        events_meta_dict[previous_idx]["last_time"] - eeg_raw.first_samp
                )
                last_fill_time = eeg_raw.n_times - (
                        events_meta_dict[ev_idx]["last_time"] - eeg_raw.first_samp
                )
                reg_signal = np.concatenate(
                    [
                        reg_signal,
                        np.full(first_fill_time, best_y[0]),
                        best_y,
                        np.full(last_fill_time, best_y[-1]),
                    ]
                )
            else:
                # Fill the time between events
                fill_time = first_time - (
                        events_meta_dict[previous_idx]["last_time"] - eeg_raw.first_samp
                )
                reg_signal = np.concatenate(
                    [reg_signal, np.full(fill_time, best_y[0]), best_y]
                )

    # Fit scalings back to eeg_raw
    reg_signal /= 1e6
    eeg_signal = eeg_raw.get_data()[0]
    reg_info = mne.create_info(
        ch_names=["reg_signal", "lc_signal"],
        ch_types=["eeg", "eeg"],
        sfreq=eeg_raw.info["sfreq"],
    )
    reg_raw = mne.io.RawArray([reg_signal, eeg_signal], reg_info)
    reg_raw_path = join(
        meeg.save_dir, f"{meeg.name}_{meeg.p_preset}_loadcell-regression-raw.fif"
    )
    reg_raw.save(reg_raw_path, overwrite=True)


def _get_load_cell_epochs(
        meeg,
        trig_plt_time,
        baseline_limit,
        trig_channel,
        apply_savgol=False,
):
    raw = meeg.load_raw()
    eeg_raw = raw.copy().pick(trig_channel)
    events = meeg.load_events()

    event_id = meeg.event_id
    trig_plt_tmin, trig_plt_tmax = trig_plt_time

    epochs_dict = dict()
    times = None

    # Exclude rating-trials
    for idx, trial in enumerate(
            [t for t in meeg.sel_trials if any([s in t for s in ["Down", "Up"]])]
    ):
        selected_ev_id = {key: value for key, value in event_id.items() if key == trial}
        # if 'Last' in trial:
        #     baseline = (round(baseline_limit/1000, 3), trig_plt_tmax)
        # else:
        #     baseline = (trig_plt_tmin, -round(baseline_limit / 1000, 3))

        eeg_epochs = mne.Epochs(
            eeg_raw,
            events,
            event_id=selected_ev_id,
            tmin=trig_plt_tmin,
            tmax=trig_plt_tmax,
            baseline=None,
            reject={"stim": 0.002},
        )
        times = eeg_epochs.times
        data = eeg_epochs.get_data()
        baseline_data = list()
        for ep in data:
            epd = ep[0]
            half_idx = int(len(epd) / 2) + 1
            if "Last" in trial:
                epd -= np.mean(epd[half_idx + baseline_limit:])
            else:
                epd -= np.mean(epd[: half_idx - baseline_limit])

            if np.mean(epd[half_idx + baseline_limit:]) < 0 and "Down" in trial:
                epd *= -1
            elif np.mean(epd[half_idx + baseline_limit:]) > 0 and "Up" in trial:
                epd *= -1

            if apply_savgol:
                epd = savgol_filter(epd, 201, 5)

            baseline_data.append(epd)

        epochs_dict[trial] = baseline_data

    return epochs_dict, times


##############################################################
# Plots
##############################################################
def _mean_of_different_lengths(data):
    arr = np.ma.empty((len(data), max([len(d) for d in data])))
    arr.mask = True
    for idx, d in enumerate(data):
        arr[idx, : len(d)] = d

    return np.mean(arr, axis=0), np.std(arr, axis=0)


def plot_ratings_combined(ct, rating_groups, group_colors, show_plots):
    """The Ratings of all groups are plotted together."""
    x = "Probanden"
    y = "Rating"
    hue = "Stimulation"
    fs = [figsize[0], figsize[1] * 2]
    fig, axes = plt.subplots(2, 1, figsize=fs)
    for ax_idx, (title, groups) in enumerate(rating_groups.items()):
        df = pd.DataFrame([], columns=[x, y])
        for group_name in groups:
            group = Group(group_name, ct)
            rating_dict = dict()
            for meeg in group.load_items(obj_type="MEEG", data_type=None):
                file_name = "ratings_meta"
                file_path = join(
                    meeg.save_dir, f"{meeg.name}_{meeg.p_preset}_{file_name}.csv"
                )
                ratings_pd = pd.read_csv(file_path, index_col=0)
                ratings = ratings_pd["rating"].values
                rating_dict[meeg.name] = ratings

            rating_dict = _merge_measurements(rating_dict, combine="combine")
            for idx, (meeg_name, ratings) in enumerate(rating_dict.items()):
                for rat in ratings:
                    df = pd.concat([df, pd.DataFrame({y: rat, x: idx + 1, hue: group_name}, index=[0])], axis=0,
                                   ignore_index=True, )

        sns.boxplot(data=df, x=x, y=y, hue=hue, palette=group_colors, ax=axes[ax_idx])
        axes[ax_idx].set_title(title)
        axes[ax_idx].label_outer()

    Group(ct.pr.name, ct).plot_save("ratings_combined")

    plt.tight_layout()

    if show_plots:
        plt.show()


def _merge_measurements(data_dict, combine="mean"):
    # Find pairs of measurments from Experiment 1
    new_data_dict = OrderedDict()
    keys = np.asarray(list(data_dict.keys()))
    reduced_keys = np.asarray([k[:-2] for k in data_dict.keys()])
    unique_keys = np.unique(reduced_keys)
    for uk in unique_keys:
        data_keys = keys[np.argwhere(reduced_keys == uk).flatten()]
        data_list = list()
        for dk in data_keys:
            data_list.append(data_dict[dk])
        if len(data_list) == 1:
            new_data_dict[dk] = data_dict[dk]
        else:
            # Take mean if data is of numpy-arrays (only optionpossible)
            if isinstance(data_list[0], np.ndarray):
                if combine == "mean":
                    res = np.mean(data_list, axis=0)
                else:
                    res = np.concatenate(data_list, axis=0)
                new_data_dict[uk] = res
            # Preserve dictionary with one level (e.g. channel-types or labels)
            elif isinstance(data_list[0], dict):
                new_data_dict[uk] = dict()
                datas = dict()
                for data in data_list:
                    for key, value in data.items():
                        if key in datas:
                            datas[key].append(value)
                        else:
                            datas[key] = [value]
                for key, value in datas.items():
                    if combine == "mean":
                        value = np.mean(value, axis=0)
                    new_data_dict[uk][key] = value
            # List is only for rating-combination (not mean)
            elif isinstance(data_list[0], list):
                if combine == "mean":
                    res = np.mean(data_list, axis=0)
                else:
                    res = np.concatenate(data_list, axis=0)
                res = list(res)
                new_data_dict[uk] = res
            else:
                new_data_dict[uk] = np.mean(data_list)

    return new_data_dict


def _get_n_subplots(n_items):
    n_subplots = np.ceil(np.sqrt(n_items)).astype(int)
    if n_items <= 2:
        nrows = 1
        ax_idxs = range(n_subplots)
    else:
        nrows = n_subplots
        ax_idxs = itertools.product(range(n_subplots), repeat=2)
    ncols = n_subplots

    return nrows, ncols, ax_idxs


ch_type_names = {"eeg": "EEG", "grad": "MEG"}


def plot_ratings_evoked_comparision(ct, ch_types, group_colors, show_plots, n_jobs):
    """Evokedsa are compared for lower and higher rating."""
    for ch_type in ch_types:
        nrows, ncols, ax_idxs = _get_n_subplots(len(ct.pr.sel_groups))
        fs = [figsize[0], figsize[1] * nrows]
        fig, ax = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            sharex=True,
            sharey=True,
            figsize=fs,
        )
        for group_name, ax_idx in zip(ct.pr.sel_groups, ax_idxs):
            group = Group(group_name, ct)
            gfps = dict()
            rating_queries = ["Lower_Ratings", "Higher_Ratings"]
            alphas = {"Lower_Ratings": 0.5, "Higher_Ratings": 1}
            colors = {rq: group_colors.get(group_name) for rq in rating_queries}
            # These trials need to be set in queries in the pipeline
            for query_trial in rating_queries:
                gfps[query_trial] = dict()
                for evokeds, meeg in group.load_items(
                        obj_type="MEEG", data_type="evoked"
                ):
                    try:
                        evoked = [ev for ev in evokeds if ev.comment == query_trial][0]
                        sfreq = evoked.info["sfreq"]
                    except IndexError:
                        raise RuntimeWarning(
                            f"No evoked found from {meeg.name} for {query_trial}"
                        )
                    else:
                        # Assuming times is the same for all measurements
                        times = evoked.times
                        gfp = calculate_gfp(evoked)
                        gfps[query_trial][meeg.name] = gfp
                if ct.pr.name == "Experiment1":
                    gfps[query_trial] = _merge_measurements(gfps[query_trial])
            group_data = list()
            for query_trial in gfps:
                query_list = list()
                for key, value in gfps[query_trial].items():
                    # Only use Gradiometer for now
                    query_list.append(value[ch_type])
                group_data.append(np.asarray(query_list))
            _plot_permutation_cluster_test(
                group_data,
                times,
                rating_queries,
                one_sample=True,
                show_plots=show_plots,
                n_jobs=n_jobs,
                unit="A/m" if ch_type == "grad" else "V",
                ax=ax[ax_idx],
                sfreq=sfreq,
                hpass=1,
                lpass=30,
                group_colors=colors,
                group_alphas=alphas,
            )
            ax[ax_idx].set_title(f"{group.name} - {ch_type_names[ch_type]}")
            ax[ax_idx].set_xlabel("Zeit (s)")
            if ch_type == "grad":
                ax[ax_idx].set_ylabel("Magnetfeldstärke (A/m)")
            else:
                ax[ax_idx].set_ylabel("elektrische Spannung (V)")
            ax[ax_idx].label_outer()
            ax[ax_idx].legend(loc="upper right", fontsize="small")
        fig.suptitle(ct.pr.name)
        Group(ct.pr.name, ct).plot_save("compare_ratings", trial=ch_type)


def plot_load_cell_group_ave(
        ct,
        trig_plt_time,
        baseline_limit,
        show_plots,
        apply_savgol,
        trig_channel,
        group_colors,
):
    """The Load-Cell-Signal is plotted for all groups and subjects."""
    groups = [g for g in ct.pr.sel_groups if "Laser" not in g]
    fs = [figsize[0], figsize[1] * len(groups)]
    fig, ax = plt.subplots(len(groups), 1, sharey=False, sharex=True, figsize=fs)
    if not isinstance(ax, np.ndarray):
        ax = [ax]

    for idx, group_name in enumerate(groups):
        group = Group(group_name, ct)
        for meeg_name in group.group_list:
            meeg = MEEG(meeg_name, group.ct)
            epochs_dict, times = _get_load_cell_epochs(
                meeg,
                trig_plt_time,
                baseline_limit,
                trig_channel,
                apply_savgol,
            )
            epo_data = list()
            for epd in epochs_dict["Down"]:
                epo_data.append(epd)
                ax[idx].plot(
                    times, epd, color=group_colors.get(group_name, "k"), alpha=0.5
                )
            epo_mean = np.mean(epo_data, axis=0)
            ax[idx].plot(
                times, epo_mean, color=group_colors.get(group_name, "k"), alpha=1
            )
            half_idx = int(len(epd) / 2) + 1
            ax[idx].plot(0, epd[half_idx], "xr")

        ax[idx].set_title(group_name)
        ax[idx].set_ylabel("Weight")
        if idx == len(ct.pr.sel_groups) - 1:
            ax[idx].set_xlabel("Time [s]")

    fig.suptitle("Load-Cell Data")
    Group(ct.pr.name, ct).plot_save("lc_trigger_all", matplotlib_figure=fig)

    if show_plots:
        fig.show()


def plot_gfp_group_stacked(ct, group_colors, ch_types, show_plots, save_plots):
    """The GFP of all groups is compared."""
    fs = [figsize[0], figsize[1] * len(ch_types)]
    fig, axes = plt.subplots(
        nrows=len(ch_types),
        ncols=1,
        sharex=True,
        sharey=False,
        figsize=fs,
    )
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    for ax_idx, ch_type in enumerate(ch_types):
        for group_name in ct.pr.sel_groups:
            group = Group(group_name, ct)
            gfps = list()
            for evokeds, meeg in group.load_items(data_type="evoked"):
                evoked = evokeds[0]
                # Assumes times is everywhere the same
                times = evoked.times
                gfp = calculate_gfp(evoked)[ch_type]
                # Apply bandpass filter 1-30 Hz
                gfp = mne.filter.filter_data(gfp, 1000, None, 30)
                gfps.append(gfp)
            axes[ax_idx].plot(
                times,
                np.mean(gfps, axis=0),
                label=group.name,
                color=group_colors.get(group_name, "k"),
            )
        axes[ax_idx].set_title(ch_type_names[ch_type])
        axes[ax_idx].set_xlabel("Time (s)")
        axes[ax_idx].set_ylabel(
            "elektrische Spannung (V)" if ch_type == "eeg" else "Magnetfeldstärke (A/m)"
        )
        axes[ax_idx].legend(loc="upper right", fontsize="small")
    plt.tight_layout()
    if show_plots:
        plt.show()
    Group("all", ct).plot_save("gfp_combined")


def plot_ltc_group_stacked(ct, group_colors, target_labels, show_plots, save_plots):
    """The label-time-course of all groups is compared."""
    nrows, ncols, ax_idxs = _get_n_subplots(len(target_labels))
    fs = [figsize[0], figsize[1] * nrows]
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex=True,
        sharey=True,
        figsize=fs,
    )
    ax_idxs = list(ax_idxs)
    for group_name in ct.pr.sel_groups:
        group = Group(group_name, ct)
        ltcs = group.load_ga_ltc()
        # Always take the first trial
        ltcs = ltcs[list(ltcs.keys())[0]]
        for ax_idx, label_name in zip(ax_idxs, target_labels):
            ltc = ltcs[label_name]
            # Apply bandpass filter 1-30 Hz
            ltc_data = mne.filter.filter_data(ltc[0], 1000, None, 30)
            times = ltc[1]
            axes[ax_idx].plot(
                times,
                ltc_data,
                label=group.name,
                color=group_colors.get(group_name, "k"),
            )
            axes[ax_idx].set_title(label_name)
    for ax in axes.flat:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("MNE Wert (A)")
        ax.legend(loc="upper right", fontsize="small")
        ax.label_outer()
    plt.tight_layout()
    if show_plots:
        plt.show()
    Group("all", ct).plot_save("ltc_combined")


##############################################################
# Plots (statistics)
##############################################################
def _get_threshold(p_value, n_observations, tail=0):
    degrees_of_freedom = n_observations - 1
    if tail == 0:
        threshold = scipy.stats.t.ppf(1 - p_value / 2, degrees_of_freedom)
    elif tail == 1:
        threshold = scipy.stats.t.ppf(1 - p_value, degrees_of_freedom)
    elif tail == -1:
        threshold = scipy.stats.t.ppf(p_value, degrees_of_freedom)
    else:
        raise ValueError("tail must be -1, 0 or 1")
    return threshold


def _plot_permutation_cluster_test(
        group_data,
        times,
        group_names,
        one_sample=False,
        show_plots=False,
        n_permutations=1000,
        tail=0,
        n_jobs=-1,
        unit="A/m",
        ax=None,
        sfreq=1000,
        hpass=None,
        lpass=30,
        group_colors={},
        group_alphas={},
):
    # Compute permutation cluster test
    if one_sample:
        if len(group_data) > 2:
            raise ValueError(
                "Only one or two group(s) allowed for one-sample test. (For two groups the difference is tested."
            )
        elif len(group_data) == 2:
            X = group_data[0] - group_data[1]
        else:
            X = group_data[0]
        perm_func = permutation_cluster_1samp_test
        threshold = _get_threshold(0.05, X.shape[0], tail=tail)
    else:
        perm_func = permutation_cluster_test
        X = group_data
        threshold = _get_threshold(0.05, X[0].shape[0], tail=tail)
    T_obs, clusters, cluster_p_values, H0 = perm_func(
        X,
        n_permutations=n_permutations,
        threshold=threshold,  # F-statistic with p-value=0.05
        tail=tail,
        n_jobs=n_jobs,
        out_type="mask",
        seed=8,
    )
    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    for data, group_name in zip(group_data, group_names):
        color = group_colors.get(group_name, None)
        alpha = group_alphas.get(group_name, 1)
        y = np.mean(data, axis=0)
        y = mne.filter.filter_data(y, sfreq, hpass, lpass)
        ax.plot(
            times,
            y,
            color=color,
            alpha=alpha,
            label=group_name,
        )

    s_counter = 0
    for i_c, c in enumerate(clusters):
        c = c[0]
        cpval = cluster_p_values[i_c]

        # Exclude clusters if before 0
        if times[c.start] < 0:
            continue

        if cpval <= 0.05:
            t_start = times[c.start]
            t_stop = times[c.stop - 1]
            ax.axvspan(
                t_start,
                t_stop,
                color="r",
                alpha=0.3,
                label=f"{s_counter}: p_val={cpval:.3f}, {t_start:.3f}-{t_stop:.3f} s",
            )
            s_counter += 1
        # else:
        #     ax.axvspan(
        #         times[c.start],
        #         times[c.stop - 1],
        #         color=(0.3, 0.3, 0.3),
        #         alpha=0.3,
        #     )
    if show_plots:
        plt.show()


def evoked_temporal_cluster(
        ct, ch_types, group_colors, compare_groups, cluster_trial, n_jobs, show_plots
):
    """A 1sample-permutation-cluster-test with clustering in time is performed between the evoked of two stimulus-groups."""
    from mne_pipeline_hd.pipeline.loading import Group
    from mne_pipeline_hd.functions.operations import calculate_gfp

    for group_names in compare_groups:
        fs = [figsize[0], figsize[1] * len(ch_types)]
        fig, axes = plt.subplots(
            nrows=len(ch_types),
            ncols=1,
            sharex=True,
            sharey=False,
            figsize=fs,
        )
        for ax_idx, ch_type in enumerate(ch_types):
            group_data = list()
            for group_name in group_names:
                group = Group(group_name, ct)
                trial = cluster_trial.get(group_name)
                datas = dict()
                for evokeds, meeg in group.load_items(
                        obj_type="MEEG", data_type="evoked"
                ):
                    try:
                        evoked = [ev for ev in evokeds if ev.comment == trial][0]
                    except IndexError:
                        print(f"No evoked for {trial} in {meeg.name}")
                    else:
                        times = evoked.times
                        gfp = calculate_gfp(evoked)[ch_type]
                        # Apply bandpass filter 1-30 Hz
                        gfp = mne.filter.filter_data(gfp, 1000, None, 30)
                        datas[meeg.name] = gfp
                if ct.pr.name == "Experiment1":
                    datas = _merge_measurements(datas)
                group_data.append(np.asarray(list(datas.values())))

            if isinstance(axes, np.ndarray):
                ax = axes[ax_idx]
            else:
                ax = axes

            unit = "V" if ch_type == "eeg" else "A/m"

            _plot_permutation_cluster_test(
                group_data,
                times,
                group_names,
                one_sample=True,
                show_plots=show_plots,
                n_jobs=n_jobs,
                group_colors=group_colors,
                ax=ax,
                unit=unit,
            )
            ax.legend(loc="upper right", fontsize="small")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(
                "elektrische Spannung (V)"
                if ch_type == "eeg"
                else "Magnetfeldstärke (A/m)"
            )
            ax.label_outer()
        plt.tight_layout()
        Group(f"{'-'.join(group_names)}", ct).plot_save("evoked_cluster_f_test")


def ltc_temporal_cluster(
        ct, compare_groups, group_colors, target_labels, cluster_trial, n_jobs, show_plots
):
    """A 1sample-permutation-cluster-test with clustering in time is performed between the label-time-courses of two stimulus-groups."""
    for group_names in compare_groups:
        label_X = list()
        for group_name in group_names:
            group = Group(group_name, ct)
            trial = cluster_trial[group_name]
            datas = {lb_name: dict() for lb_name in target_labels}
            for ltcs, meeg in group.load_items(obj_type="MEEG", data_type="ltc"):
                ltcs = ltcs[trial]
                for label_name, ltc in ltcs.items():
                    # Assumes times is everywhere the same
                    times = ltc[1]
                    # Apply bandpass filter 1-30 Hz
                    ltc_data = mne.filter.filter_data(ltc[0], 1000, None, 30)
                    datas[label_name][meeg.name] = ltc_data
            if ct.pr.name == "Experiment1":
                for label_name, data in datas.items():
                    datas[label_name] = _merge_measurements(data)
            label_X.append(datas)

        nrows, ncols, ax_idxs = _get_n_subplots(len(target_labels))
        fs = [figsize[0], figsize[1] * nrows]
        fig, ax = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            sharex=True,
            sharey=True,
            figsize=fs,
        )
        for label_name, ax_idx in zip(target_labels, ax_idxs):
            group_data = list()
            for data in label_X:
                label_data = data[label_name]
                group_data.append(np.asarray(list(label_data.values())))

            _plot_permutation_cluster_test(
                group_data,
                times,
                group_names,
                one_sample=True,
                ax=ax[ax_idx],
                unit="A",
                n_jobs=n_jobs,
                show_plots=show_plots,
                group_colors=group_colors,
            )
            ax[ax_idx].set_title(label_name)
            ax[ax_idx].legend(loc="upper right", fontsize="small")
            ax[ax_idx].set_xlabel("Time (s)")
            ax[ax_idx].set_ylabel("MNE Wert (A)")
            ax[ax_idx].label_outer()

        group.plot_save(
            "ltc_cluster_f_test",
            trial=" vs ".join(group_names),
        )


def label_power(
        meeg,
        tfr_freqs,
        inverse_method,
        target_labels,
        tfr_baseline,
        tfr_baseline_mode,
        n_jobs,
):
    """The power inside the given labels is computed and saved."""
    inv_op = meeg.load_inverse_operator()

    labels = meeg.fsmri.get_labels(target_labels)
    n_cycles = tfr_freqs / 3.0

    for trial, epoch in meeg.get_trial_epochs():
        for lix, label in enumerate(labels):
            print("Computing power of")
            power, itc = source_induced_power(
                epoch,
                inv_op,
                tfr_freqs,
                label,
                baseline=tfr_baseline,
                baseline_mode=tfr_baseline_mode,
                n_cycles=n_cycles,
                n_jobs=n_jobs,
                method=inverse_method,
                pick_ori=None,
            )

            # Average over sources
            power = np.mean(power, axis=0)
            power_save_path = join(
                meeg.save_dir, f"{meeg.name}-{meeg.p_preset}-{trial}-{label.name}.npy"
            )
            np.save(power_save_path, power)


def plot_label_power(ct, tfr_freqs, target_labels, cluster_trial, show_plots, n_jobs):
    """Plot the label power and compute permutation-cluster statistics against 0."""
    tfr_dict = dict()
    for group_name in ct.pr.sel_groups:
        tfr_dict[group_name] = dict()
        group = Group(group_name, ct)
        times = MEEG(group.group_list[0], ct).load_evokeds()[0].times
        trial = cluster_trial[group_name]
        for label_name in target_labels:
            tfr_dict[group_name][label_name] = list()
            for meeg in group.load_items(obj_type="MEEG", data_type=None):
                power_save_path = join(
                    meeg.save_dir,
                    f"{meeg.name}-{meeg.p_preset}-{trial}-{label_name}.npy",
                )
                power = np.load(power_save_path)
                tfr_dict[group_name][label_name].append(power)
    for label_name in target_labels:
        nrows, ncols, ax_idxs = _get_n_subplots(len(ct.pr.sel_groups))
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
        fig.suptitle(label_name)
        for group_name, ax_idx in zip(ct.pr.sel_groups, ax_idxs):
            tfr_data = tfr_dict[group_name][label_name]
            tfr_averaged = np.mean(tfr_data, axis=0)
            # Exclude edge-artefacts
            vmax = np.std(np.abs(tfr_averaged)) * 2
            vmin = -vmax
            im = ax[ax_idx].imshow(
                tfr_averaged,
                cmap=colormaps["rainbow"],
                extent=[times[0], times[-1], tfr_freqs[0], tfr_freqs[-1]],
                aspect="auto",
                origin="lower",
                vmin=vmin,
                vmax=vmax,
            )

            ax[ax_idx].set_title(group_name)
            ax[ax_idx].set_xlabel("Time (ms)")
            ax[ax_idx].set_ylabel("Frequency (Hz)")
            ax[ax_idx].label_outer()
            plt.colorbar(im, ax=ax[ax_idx])
        plt.tight_layout()
        Group(label_name, ct).plot_save(
            f"label_power_{label_name}", matplotlib_figure=fig
        )
    if show_plots:
        plt.show()


def label_power_cond_permclust(
        ct,
        label_pw_groups,
        tfr_freqs,
        target_labels,
        cluster_trial,
        n_jobs,
        show_plots,
):
    """The power inside the given labels is compared between groups with a 1sample-permutation-cluster-test with clustering in time and frequency."""
    """As in Compute power and phase lock in label of the source space."""
    for group_names in label_pw_groups:
        p_accept = 0.05
        tail = 1
        if len(group_names) > 2:
            raise ValueError("Only two groups allowed for comparison")
        nrows, ncols, ax_idxs = _get_n_subplots(len(target_labels))
        fs = [figsize[0], figsize[1] * nrows]
        fig, ax = plt.subplots(
            nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=fs
        )
        ims = list()
        diff_avgs = list()
        for label, ax_idx in zip(target_labels, ax_idxs):
            X = list()
            for group_name in group_names:
                group = Group(group_name, ct)
                group_powers = dict()
                trial = cluster_trial.get(group_name, None)
                assert trial is not None
                times = MEEG(group.group_list[0], ct).load_evokeds()[0].times
                for meeg_name in group.group_list:
                    power_save_path = join(
                        group.pr.data_path,
                        meeg_name,
                        f"{meeg_name}-{group.p_preset}-{trial}-{label}.npy",
                    )
                    power = np.load(power_save_path)
                    group_powers[meeg_name] = power
                group_powers_merged = _merge_measurements(group_powers)
                powers_array = np.asarray(list(group_powers_merged.values()))
                X.append(powers_array)
            # Also allow one sample (testing against 0)
            X = [np.transpose(x, (0, 2, 1)) for x in X]
            if len(X) == 2:
                X = X[0] - X[1]
            else:
                X = X[0]

            threshold = _get_threshold(p_accept, X.shape[0], tail=tail)
            T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
                X,
                n_permutations=1000,
                threshold=threshold,
                n_jobs=n_jobs,
                tail=tail,
                adjacency=None,
                seed=8,
            )
            good_clusted_inds = np.where(cluster_p_values < p_accept)[0]
            print(f"Found {len(good_clusted_inds)} significant clusters")

            T_obs_plot = np.nan * np.ones_like(T_obs)
            for c, p_val in zip(clusters, cluster_p_values):
                if p_val <= p_accept:
                    T_obs_plot[c] = T_obs[c]

            diff_avg = np.mean(X, axis=0)
            time_idx = np.nonzero(np.round(times, 3) == 0.43)
            tfr_idx = np.nonzero(np.round(tfr_freqs) == 50)
            gpower_mean = diff_avg[time_idx, tfr_idx]
            gpower_std = np.std(X, axis=0)[time_idx, tfr_idx]
            print(f"Label {label}: mean={gpower_mean}, std={gpower_std}")

            diff_avgs.append(diff_avg)
            im = ax[ax_idx].imshow(
                diff_avg.T,
                cmap=colormaps["rainbow"],
                extent=[times[0], times[-1], tfr_freqs[0], tfr_freqs[-1]],
                aspect="auto",
                origin="lower",
            )
            ims.append(im)
            ax[ax_idx].imshow(
                T_obs_plot.T,
                cmap=colormaps["Greys"],
                extent=[times[0], times[-1], tfr_freqs[0], tfr_freqs[-1]],
                aspect="auto",
                origin="lower",
            )
            fig.colorbar(im, ax=ax[ax_idx])
            if ax_idx[0] == nrows - 1:
                ax[ax_idx].set_xlabel("Time (ms)")
            if ax_idx[1] == 0:
                ax[ax_idx].set_ylabel("Frequency (Hz)")
            ax[ax_idx].set_title(label)
        # Set vmin and vmax according to all averages for min/max
        vmin = np.min(np.concatenate(diff_avgs))
        vmax = np.max(np.concatenate(diff_avgs))
        for im in ims:
            im.set(clim=(vmin, vmax))

        plt_title = "Difference " + "-".join(group_names)
        fig.suptitle(
            f"{plt_title} ({len(good_clusted_inds)} cluster for p < {p_accept}, tail={tail})"
        )
        plt.tight_layout()
        Group("all", ct).plot_save(
            f"label_power_permclust_{plt_title}", matplotlib_figure=fig
        )

        if show_plots:
            fig.show()


def _connectivity_geodesic_dist(A, B):
    """Prepares the connectivity matrix and computes the geodesic distance."""
    # Copy lower triangle to upper triangle
    A += np.rot90(np.fliplr(A))
    B += np.rot90(np.fliplr(B))

    # Fill diagonal with ones
    np.fill_diagonal(A, 1)
    np.fill_diagonal(B, 1)

    # Check if positive definite
    if not np.all(np.linalg.eigvals(A) > 0):
        raise ValueError("A is not positive definite")
    if not np.all(np.linalg.eigvals(B) > 0):
        raise ValueError("B is not positive definite")
    eigenvals, _ = np.linalg.eig(np.linalg.inv(A).dot(B))
    result = np.sqrt(np.sum(np.log(eigenvals) ** 2))

    return result


def connectivity_geodesic_statistics(
        ct,
        compare_groups,
        cluster_trial,
        show_plots,
        save_plots,
        con_fmin,
        con_fmax,
):
    """This computes the geodesic distance between connectivity matrices of two groups,
    calculates a 1sample-t-test and plots the results."""

    if not isinstance(con_fmin, list):
        con_fmin = [con_fmin]
    if not isinstance(con_fmax, list):
        con_fmax = [con_fmax]

    freq_pairs = list(zip(con_fmin, con_fmax))
    x = "Vergleichsgruppen"
    y = "Geodätische Distanz"
    fs = (figsize[0], figsize[1] * len(freq_pairs))
    fig, axes = plt.subplots(
        nrows=len(freq_pairs), sharex=True, sharey=True, figsize=fs
    )
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    for freq_idx, freq in enumerate(freq_pairs):
        df = pd.DataFrame([], columns=[x, y])
        for group_names in compare_groups:
            group_key = " - ".join(group_names)
            if len(group_names) != 2:
                raise ValueError("Group-Names of 'compare_groups' can only be two.")
            data = list()
            for group_name in group_names:
                group = Group(group_name, ct)
                trial = cluster_trial[group_name]
                group_data = dict()
                for con, meeg in group.load_items(obj_type="MEEG", data_type="src_con"):
                    con = con[trial]["wpli"].get_data("dense")[:, :, freq_idx]
                    group_data[meeg.name] = con
                group_data = _merge_measurements(group_data)
                data.append(group_data.values())

            for sub_data1, sub_data2 in zip(data[0], data[1]):
                try:
                    dist = _connectivity_geodesic_dist(sub_data1, sub_data2)
                except ValueError as exc:
                    print(exc)
                    print(f"Connectivity-data will be excluded")
                else:
                    df = pd.concat([df, pd.DataFrame({y: dist, x: group_key}, index=[0])], axis=0, ignore_index=True)
        pairs = [i for i in itertools.combinations(np.unique(df[x]), 2)]
        sns_ax = sns.boxplot(data=df, x=x, y=y, ax=axes[freq_idx])
        axes[freq_idx].set_title(f"{freq[0]}-{freq[1]} Hz")
        axes[freq_idx].label_outer()
        annotator = Annotator(sns_ax, pairs, data=df, x=x, y=y)
        annotator.configure(test="t-test_paired", comparisons_correction="Bonferroni",
                            text_format="star", show_test_name=True, hide_non_significant=True)
        annotator.apply_and_annotate()

    plt.tight_layout()
    if show_plots:
        fig.show()
    if save_plots:
        Group(ct.pr.name, ct).plot_save("connectivity_geodesic_distances")


def export_ltcs(ltc_target_dir, cluster_trial, ct):
    for group_name in ct.pr.sel_groups:
        group_dir = join(ltc_target_dir, group_name)
        if not isdir(group_dir):
            mkdir(group_dir)
        group = Group(group_name, ct)
        trial = cluster_trial.get(group_name, None)
        for ltcs, meeg in group.load_items(obj_type="MEEG", data_type="ltc"):
            ltcs = ltcs[trial]
            for label_name, ltc in ltcs.items():
                ltc_save_path = join(
                    group_dir,
                    f"{meeg.name}+{label_name}.npy",
                )
                np.save(ltc_save_path, ltc)
                print(f"{meeg.name}: Saved {ltc_save_path}")


def _significant_formatter(value):
    value_str = f"{value:.3f}"
    if value <= 0.05:
        value_str = "\\textbf{" + value_str + "}"
    return value_str


def con_t_test(
        compare_groups, con_fmin, con_fmax, con_compare_labels, cluster_trial, ct
):
    if not isinstance(con_fmin, list):
        con_fmin = [con_fmin]
    if not isinstance(con_fmax, list):
        con_fmax = [con_fmax]
    for group_names in compare_groups:
        results_df = pd.DataFrame(
            index=[f"{lb1} -> {lb2}" for lb1, lb2 in con_compare_labels],
            columns=[f"{fmin}-{fmax}" for fmin, fmax in zip(con_fmin, con_fmax)],
        )
        for fidx, (fmin, fmax) in enumerate(zip(con_fmin, con_fmax)):
            data = dict()
            for label1, label2 in con_compare_labels:
                label_key = f"{label1} -> {label2}"
                data[label_key] = dict()
                for group_name in group_names:
                    if group_name not in data[label_key]:
                        data[label_key][group_name] = dict()
                    group = Group(group_name, ct)
                    trial = cluster_trial[group_name]
                    for con, meeg in group.load_items(
                            obj_type="MEEG", data_type="src_con"
                    ):
                        con = con[trial]["wpli"]
                        assert len(con.freqs) == len(con_fmin)
                        label1_idx = con.names.index(label1)
                        label2_idx = con.names.index(label2)
                        con_data = con.get_data("dense")[:, :, fidx]
                        con_data += np.rot90(np.fliplr(con_data))
                        data[label_key][group_name][meeg.name] = con_data[
                            label1_idx, label2_idx
                        ]

            for lb_idx, label_key in enumerate(data):
                X = list()
                for group_name in data[label_key]:
                    merged = _merge_measurements(data[label_key][group_name])
                    X.append(list(merged.values()))
                X = np.asarray(X)
                result = ttest_1samp(X[0] - X[1], 0, alternative="greater")
                results_df.iloc[lb_idx, fidx] = result.pvalue
                result_str = f"{' vs. '.join(group_names)}, {label_key}, {fmin}-{fmax} Hz: t={result.statistic}, p={result.pvalue}, conf_int={result.confidence_interval()}\n"
                print(result_str)
        # Apply Bonferroni-Correction
        reject, pval_corrected = bonferroni_correction(
            np.asarray(results_df), alpha=0.05
        )
        results_df.iloc[:, :] = pval_corrected
        latex_table = results_df.to_latex(
            formatters=[_significant_formatter for i in range(len(results_df.columns))],
            caption=f"Ergebnisse des T-Tests für abhängige Stichproben für den Unterschied zwischen {' und '.join(group_names)} in den Konnektivitäten aus {ct.pr.name}."
                    f"Die Ergebnisse sind Bonferroni-korrigiert.",
            label=f"tab:con_t_test_{'-'.join(group_names)}",
        )
        with open(
                join(
                    ct.pr.save_dir_averages,
                    f"{ct.pr.name}_{' vs. '.join(group_names)}_con_t_statistics.tex",
                ),
                "w",
                encoding="utf-8",
        ) as f:
            f.write(latex_table)


def add_velo_meta(meeg):
    events = meeg.load_events()

    file_name = "velo_meta"
    file_path = join(meeg.save_dir, f"{meeg.name}_{meeg.p_preset}_{file_name}.csv")
    velo_meta_pd = pd.DataFrame([], columns=["time", "id", "velo"], dtype=int)

    target_times = events[np.nonzero(events[:, 2] == 4)][:, 0]

    for velo, velo_id in zip((28, 56), (1, 2)):
        time_points = events[np.nonzero(events[:, 2] == velo_id)][:, 0]
        for tp in time_points:
            diffs = np.abs(target_times - tp)
            print(f"For {tp} Min-Diff={min(diffs)}")
            time_result = target_times[np.argmin(diffs)]
            time_dict = {
                "time": time_result,
                "id": 4,
                "velo": velo,
            }
            meta_series = pd.Series(time_dict)
            velo_meta_pd = pd.concat(
                [velo_meta_pd, meta_series.to_frame().T],
                axis=0,
                ignore_index=True,
            )

    velo_meta_pd = velo_meta_pd.sort_values("time", ascending=True, ignore_index=True)
    velo_meta_pd.to_csv(file_path)

    epochs = meeg.load_epochs()
    _add_events_meta(epochs, velo_meta_pd, "velo")
    meeg.save_epochs(epochs)


def plot_velo_evoked(group, show_plots):
    evs = list()
    for evokeds, meeg in group.load_items(data_type="evoked"):
        evo = evokeds[0]
        evo.comment = f"{meeg.name[-2:]} mm/s"
        evs.append(evo)
    evs = mne.equalize_channels(evs)
    fig = mne.viz.plot_compare_evokeds(
        evs, title="Vergleich vertikale Geschwindigkeiten", show=show_plots
    )
    group.plot_save("velo_comparision", matplotlib_figure=fig)

def _get_group(meeg_name, groups):
    for group_name, group in groups.items():
        if meeg_name in group:
            return group_name
    return None

def combine_meegs_rating(meeg, combine_groups):
    group_names = dict()
    for group_name in combine_groups:
        group_names[group_name] = Group(group_name, meeg.ct).group_list
    if _get_group(meeg.name, group_names) is not None:
        first_pattern = r"_(\w{0,7})([ab_]*)"
        pattern = r"([p,l]{2}\d{1,2}a?)" + first_pattern
        match = re.match(pattern, meeg.name)
        sub_name = match.group(1)
        new_high_name = f"{sub_name}_combined_high"
        new_low_name = f"{sub_name}_combined_low"
        if new_high_name in meeg.pr.all_meeg and new_low_name in meeg.pr.all_meeg:
            print(f"{meeg.name} already combined")
            return
        all_epochs = [meeg.load_epochs()]
        stims = [_get_group(meeg.name, group_names)]
        for other_meeg in [m for m in meeg.pr.all_meeg if m!=meeg.name and _get_group(m, group_names) is not None]:
            match = re.match(sub_name + first_pattern, other_meeg)
            if match:
                all_epochs.append(MEEG(other_meeg, meeg.ct).load_epochs())
                stims.append(_get_group(other_meeg, group_names))
                print(f"Found {other_meeg} for {meeg.name}")

        new_id = 1
        all_epochs = mne.channels.equalize_channels(all_epochs)
        epochs = all_epochs[0]
        new_data = epochs.get_data()
        new_events = epochs.events.copy()
        new_events[:, 2] = new_id
        new_metadata = epochs.metadata.copy()
        new_metadata["Stimulus"] = stims[0]
        new_metadata["id"] = new_id

        for epo, stim in zip(all_epochs[1:], stims[1:]):
            new_data = np.concatenate((new_data, epo.get_data()), axis=0)
            e2_events = epo.events.copy()
            ev_offset = new_events[-1, 0] + 10
            e2_events[:, 0] += ev_offset
            e2_events[:, 2] = new_id
            new_events = np.concatenate((new_events, e2_events), axis=0)
            meta2 = epo.metadata.copy()
            meta2["time"] += ev_offset
            meta2["Stimulus"] = stim
            meta2["id"] = new_id
            new_metadata = pd.concat([new_metadata, meta2], axis=0, ignore_index=True)

        combined_epochs = mne.EpochsArray(new_data, epochs.info, new_events, tmin=epochs.tmin,
                                          event_id={"Stimulation": 1},
                                          baseline=epochs.baseline, metadata=new_metadata)
        epochs_high = combined_epochs[combined_epochs.metadata.reset_index().sort_values(by="rating").index[len(combined_epochs)//2:]]
        epochs_low = combined_epochs[combined_epochs.metadata.reset_index().sort_values(by="rating").index[:len(combined_epochs)//2]]
        for name, epoch, group_name in zip([new_high_name, new_low_name], [epochs_high, epochs_low], ["Hohes Rating", "Niedriges Rating"]):
            new_meeg = meeg.pr.add_meeg(name)
            meeg.pr.meeg_to_fsmri[name] = meeg.fsmri.name
            meeg.pr.sel_event_id[name] = meeg.sel_trials
            meeg.pr.meeg_event_id[name] = meeg.event_id
            new_meeg.save_epochs(epoch)
            print(f"Combined {len(all_epochs)} measurements to {name}")
            if group_name not in meeg.pr.all_groups:
                meeg.pr.all_groups[group_name] = [name]
            else:
                meeg.pr.all_groups[group_name].append(name)
    meeg.pr.save()


def plot_rating_share(ct, combine_groups, show_plots):
    fs = [figsize[0], figsize[1] * 2]
    fig, ax = plt.subplots(2, 1, figsize=fs)
    for ax_idx, rat_group in enumerate(["Hohes Rating", "Niedriges Rating"]):
        group = Group(rat_group, ct)
        bottom = np.zeros(len(group.group_list))
        vals = {g: list() for g in combine_groups}
        for epochs, meeg in group.load_items(data_type="epochs"):
            val_dict = epochs.metadata.value_counts("Stimulus").to_dict()
            for k in combine_groups:
                if k in val_dict:
                    vals[k].append(val_dict[k])
                else:
                    vals[k].append(0)
        for k, v in vals.items():
            p = ax[ax_idx].bar([str(i) for i in range(1, len(group.group_list) + 1)], v, bottom=bottom, label=k)
            bottom += v

            ax[ax_idx].bar_label(p, label_type="center")
        ax[ax_idx].set_title(rat_group)
        ax[ax_idx].set_ylabel("Anzahl der Stimuli")
        ax[ax_idx].set_xlabel("Proband")
        ax[ax_idx].legend()

    plt.tight_layout()

    if show_plots:
        plt.show()

    group.plot_save("rating_share", matplotlib_figure=fig)