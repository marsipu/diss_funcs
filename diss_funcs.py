import sys
from collections import OrderedDict
from os.path import join

import mne
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mne.stats import permutation_cluster_test, permutation_cluster_1samp_test
from mne_pipeline_hd.functions.operations import calculate_gfp, find_6ch_binary_events
from mne_pipeline_hd.pipeline.loading import Group, MEEG
from scipy.signal import find_peaks, savgol_filter
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


##############################################################
# Preparation
##############################################################


def combine_labels(fsmri, label_combinations):
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


##############################################################
# Ratings
##############################################################
def get_ratings(meeg, target_event_id):
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

    rating_meta_pd.to_csv(file_path)


def _add_events_meta(epochs, meta_pd):
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
                pd.Series({"time": miss_time, "id": miss_id, "rating": np.nan})
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
    epochs = meeg.load_epochs()
    file_name = "ratings_meta"
    file_path = join(meeg.save_dir, f"{meeg.name}_{meeg.p_preset}_{file_name}.csv")
    ratings_pd = pd.read_csv(file_path, index_col=0)

    _add_events_meta(epochs, ratings_pd)
    meeg.save_epochs(epochs)


def remove_metadata(meeg):
    epochs = meeg.load_epochs()
    epochs.metadata = None
    meeg.save_epochs(epochs)


##############################################################
# Load-Cell
##############################################################
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
            eeg_series[pk - (len_baseline + baseline_limit) : pk - baseline_limit + 1]
        ).mean()
        post_baseline_mean = np.asarray(
            eeg_series[pk + baseline_limit : pk + baseline_limit + len_baseline + 1]
        ).mean()
        pre_peak_data = np.flip(np.asarray(eeg_series[pk - min_ev_distance : pk + 1]))
        post_peak_data = np.asarray(eeg_series[pk : pk + min_ev_distance + 1])
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

    for idx, trial in enumerate(meeg.sel_trials):
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
        )
        times = eeg_epochs.times
        data = eeg_epochs.get_data()
        baseline_data = list()
        for ep in data:
            epd = ep[0]
            half_idx = int(len(epd) / 2) + 1
            if "Last" in trial:
                epd -= np.mean(epd[half_idx + baseline_limit :])
            else:
                epd -= np.mean(epd[: half_idx - baseline_limit])

            if np.mean(epd[half_idx + baseline_limit :]) < 0 and "Down" in trial:
                epd *= -1
            elif np.mean(epd[half_idx + baseline_limit :]) > 0 and "Up" in trial:
                epd *= -1

            if apply_savgol:
                epd = savgol_filter(epd, 201, 5)

            baseline_data.append(epd)

        epochs_dict[trial] = baseline_data

    return epochs_dict, times


##############################################################
# Plots (descriptive)
##############################################################
def _mean_of_different_lengths(data):
    arr = np.ma.empty((len(data), max([len(d) for d in data])))
    arr.mask = True
    for idx, d in enumerate(data):
        arr[idx, : len(d)] = d

    return np.mean(arr, axis=0), np.std(arr, axis=0)


def plot_ratings_combined(ct, rating_groups, group_colors, show_plots):
    fig, ax = plt.subplots(len(rating_groups), 1, sharex=True, sharey=True)
    for idx, (group_title, group_names) in enumerate(rating_groups.items()):
        for group_name in group_names:
            group = Group(group_name, ct)
            group_ratings = list()
            for meeg in group.load_items(obj_type="MEEG", data_type=None):
                file_name = "ratings_meta"
                file_path = join(
                    meeg.save_dir, f"{meeg.name}_{meeg.p_preset}_{file_name}.csv"
                )
                ratings_pd = pd.read_csv(file_path, index_col=0)
                ratings = ratings_pd["rating"].values
                group_ratings.append(ratings)
            group_mean, group_std = _mean_of_different_lengths(group_ratings)
            ax[idx].plot(
                group_mean, color=group_colors[group_name], alpha=1, label=group_name
            )
            ax[idx].fill_between(
                x=np.arange(len(group_mean)),
                y1=group_mean - group_std,
                y2=group_mean + group_std,
                alpha=0.5,
                color=group_colors[group_name],
            )
        ax[idx].set_xlabel("Epochs")
        ax[idx].set_ylabel("Rating")
        ax[idx].legend()
        ax[idx].set_title(group_title)
        # Hide inner labels
        ax[idx].label_outer()
    Group("all", ct).plot_save("ratings_combined")

    if show_plots:
        fig.show()


def _merge_measurements(data_dict):
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
        # Take mean if data is of numpy-arrays
        if isinstance(data_list[0], np.ndarray):
            new_data_dict[uk] = np.mean(data_list, axis=0)
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
                new_data_dict[uk][key] = np.mean(value, axis=0)

    return new_data_dict


def plot_ratings_evoked_comparision(group, show_plots, n_jobs):
    gfp_lr = OrderedDict()
    gfp_hr = OrderedDict()

    for epochs, meeg in group.load_items(obj_type="MEEG", data_type="epochs"):
        # Assuming times is the same for all measurements
        times = epochs.times
        try:
            ratings_mean = np.mean(epochs.metadata["rating"])
        except (KeyError, TypeError):
            print(f"{meeg.name} could not be included due to reasons")
        else:
            # Calculate GFP diff lower and higher ratings
            gfp_lr[meeg.name] = calculate_gfp(
                epochs[f"rating < {ratings_mean}"].average()
            )
            gfp_hr[meeg.name] = calculate_gfp(
                epochs[f"rating > {ratings_mean}"].average()
            )
    ch_types = gfp_lr[list(gfp_lr.keys())[0]]
    gfp_lr = _merge_measurements(gfp_lr)
    gfp_hr = _merge_measurements(gfp_hr)
    for ch_type in ch_types:
        lr_data = list()
        for key, value in gfp_lr.items():
            lr_data.append(value[ch_type])
        hr_data = list()
        for key, value in gfp_hr.items():
            hr_data.append(value[ch_type])
        X = [np.asarray(lr_data), np.asarray(hr_data)]
        _plot_permutation_cluster_test(
            X,
            times,
            ["GFP: Rating < Durchschnitt", "GFP: Rating > Durchschnitt"],
            show_plots,
            n_jobs=n_jobs,
            threshold=None,
        )
        plt.title(f"{group.name}: Ratings")
        plt.xlabel("Zeit (ms)")
        plt.ylabel("F-Werte")
        group.plot_save("compare_ratings", trial=ch_type)


def plot_load_cell_group_ave(
    ct,
    trig_plt_time,
    baseline_limit,
    show_plots,
    apply_savgol,
    trig_channel,
    group_colors,
):
    fig, ax = plt.subplots(len(ct.pr.sel_groups), 1, sharey=True, sharex=True)
    if not isinstance(ax, np.ndarray):
        ax = [ax]

    for idx, group_name in enumerate(ct.pr.sel_groups):
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

    plt.subplots_adjust(hspace=0.2)
    fig.suptitle("Load-Cell Data")
    Group("all", ct).plot_save("lc_trigger_all", matplotlib_figure=fig)

    if show_plots:
        fig.show()


def plot_gfp_stacked(group, show_plots, save_plots):
    fig, ax = plt.subplots()
    for evokeds, meeg in group.load_items(data_type="evoked"):
        evoked = evokeds[0]
        times = evoked.times
        gfp = calculate_gfp(evoked)
        ax.plot(times, gfp["grad"], label=meeg.name)
    plt.xlabel("Time (s)")
    plt.ylabel("GFP")
    plt.title(f"GFP compared across velocities for {group.name}")
    plt.legend()
    if show_plots:
        plt.show()
    if save_plots:
        group.plot_save("gfp_stacked")


def plot_gfp_group_stacked(ct, ch_types, show_plots, save_plots):
    fig, ax = plt.subplots(nrows=len(ch_types), sharex=True)
    for ch_idx, ch_type in enumerate(ch_types):
        for group_name in ct.pr.sel_groups:
            group = Group(group_name, ct)
            gfps = list()
            for evokeds, meeg in group.load_items(data_type="evoked"):
                evoked = evokeds[0]
                # Assumes times is everywhere the same
                times = evoked.times
                gfp = calculate_gfp(evoked)[ch_type]
                gfps.append(gfp)
            if len(ch_types) == 1:
                ax.plot(times, np.mean(gfps, axis=0), label=group.name)
            else:
                ax[ch_idx].plot(times, np.mean(gfps, axis=0), label=group.name)
    plt.xlabel("Time (s)")
    plt.ylabel("GFP")
    plt.legend()
    plt.tight_layout()
    if show_plots:
        plt.show()
    if save_plots:
        plt.savefig(
            join(
                ct.pr.save_dir_averages,
                f"gfp_group_stacked_{ch_type}.png",
            ),
            dpi=600,
        )


def plot_ltc_group_stacked(ct, target_labels, show_plots, save_plots):
    fig, ax = plt.subplots(nrows=len(target_labels), sharex=True, sharey=True)
    for group_name in ct.pr.sel_groups:
        group = Group(group_name, ct)
        ltcs = group.load_ga_ltc()
        # Always take the first trial
        ltcs = ltcs[list(ltcs.keys())[0]]
        for lb_idx, label_name in enumerate(target_labels):
            ltc = ltcs[label_name]
            ax[lb_idx].plot(ltc[1], ltc[0], label=group.name)
    plt.xlabel("Time (s)")
    plt.ylabel("GFP")
    plt.legend()
    plt.tight_layout()
    if show_plots:
        plt.show()
    if save_plots:
        plt.savefig(
            join(
                ct.pr.save_dir_averages,
                f"ltc_group_stacked.png",
            ),
            dpi=600,
        )


##############################################################
# Plots (statistics)
##############################################################
def _plot_permutation_cluster_test(
    X,
    times,
    group_names,
    one_sample=False,
    show_plots=True,
    n_permutations=1000,
    threshold=None,
    tail=0,
    n_jobs=-1,
):
    # Compute permutation cluster test
    if one_sample:
        perm_func = permutation_cluster_test
    else:
        perm_func = permutation_cluster_1samp_test
    T_obs, clusters, cluster_p_values, H0 = perm_func(
        X,
        n_permutations=n_permutations,
        threshold=threshold,  # F-statistic with p-value=0.05
        tail=tail,
        n_jobs=n_jobs,
        out_type="mask",
        seed=8,
    )

    plt.figure(figsize=(8, 4))
    for group_idx, group_name in enumerate(group_names):
        plt.plot(
            times,
            np.mean(X[group_idx], axis=0),
            label=group_name,
        )

    for i_c, c in enumerate(clusters):
        c = c[0]
        cpval = cluster_p_values[i_c]
        if cpval <= 0.05:
            plt.axvspan(
                times[c.start],
                times[c.stop - 1],
                color="r",
                alpha=0.3,
                label=f"p_val = {cpval:.3f}",
            )
        # else:
        #     plt.axvspan(
        #         times[c.start],
        #         times[c.stop - 1],
        #         color=(0.3, 0.3, 0.3),
        #         alpha=0.3,
        #     )
    plt.legend()
    if show_plots:
        plt.show()


def evoked_temporal_cluster(ct, compare_groups, cluster_trial, n_jobs, show_plots):
    import numpy as np
    from mne_pipeline_hd.pipeline.loading import Group
    from mne_pipeline_hd.functions.operations import calculate_gfp

    for group_names in compare_groups:
        X = list()
        for group_name in group_names:
            group = Group(group_name, ct)
            trial = cluster_trial.get(group_name)
            group_data = list()
            for evokeds, meeg in group.load_items(obj_type="MEEG", data_type="evoked"):
                try:
                    evoked = [ev for ev in evokeds if ev.comment == trial][0]
                except IndexError:
                    print(f"No evoked for {trial} in {meeg.name}")
                else:
                    times = evoked.times
                    gfp = calculate_gfp(evoked)["grad"]
                    # Apply bandpass filter 1-30 Hz
                    gfp = mne.filter.filter_data(gfp, 1000, 1, 30)
                    group_data.append(gfp)
            X.append(np.asarray(group_data))

        _plot_permutation_cluster_test(X, times, group_names, show_plots, n_jobs=n_jobs)

        group.plot_save(
            "evoked_cluster_f_test",
            trial=" vs ".join(group_names),
        )


def ltc_temporal_cluster(
    ct, compare_groups, target_labels, cluster_trial, n_jobs, show_plots
):
    for group_names in compare_groups:
        label_X = list()
        for group_name in group_names:
            group = Group(group_name, ct)
            trial = cluster_trial[group_name]
            group_data = {lb_name: list() for lb_name in target_labels}
            for ltcs, meeg in group.load_items(obj_type="MEEG", data_type="ltc"):
                ltcs = ltcs[trial]
                for label_name, ltc in ltcs.items():
                    # Assumes times is everywhere the same
                    times = ltc[1]
                    # Apply bandpass filter 1-30 Hz
                    ltc_data = mne.filter.filter_data(ltc[0], 1000, 1, 30)
                    group_data[label_name].append(ltc_data)
            label_X.append(group_data)

        for label_name in target_labels:
            X = list()
            for data in label_X:
                label_data = data[label_name]
                X.append(np.asarray(label_data))

            _plot_permutation_cluster_test(X, times, group_names, n_jobs, show_plots)

            group.plot_save(
                "ltc_cluster_f_test",
                subfolder=label_name,
                trial=" vs ".join(group_names),
            )
