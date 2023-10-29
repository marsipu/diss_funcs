from os.path import join

from matplotlib import pyplot as plt
from mne_pipeline_hd.functions.operations import calculate_gfp
from mne_pipeline_hd.pipeline.loading import Group


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
