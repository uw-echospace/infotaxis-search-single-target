
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def get_cell_numbers(r):
    return 1 + 3*r*(r+1)


def get_beam_search_space_ratio(cr, br):
    grid_in_beam = get_cell_numbers(br)
    grid_in_space = get_cell_numbers(cr)
    return grid_in_beam / grid_in_space


def consolidate_success_num_pings(df):
    # Filter df to include only runs that are successful in either criterion
    # so num_pings_pmax_th and num_pings_pmax_diff cannot be -1
    df_filt = df[(df["num_pings_pmax_th"] != -1) | (df["num_pings_pmax_diff"] != -1)].copy()

    # Use the success result from the smaller of the 2 num_pings
    df_filt.loc[:, "success"] = df_filt.apply(
        lambda row: row["success_pmax_th"]
            if (row["num_pings_pmax_th"] != -1) and (row["num_pings_pmax_th"] <= row["num_pings_pmax_diff"])
            else row["success_pmax_diff"],
        axis=1
    )
    df_filt.loc[:, "num_pings_final"] = df_filt.apply(
        lambda row: row["num_pings_pmax_th"] 
            if (row["num_pings_pmax_th"] != -1) and (row["num_pings_pmax_th"] <= row["num_pings_pmax_diff"])
            else row["num_pings_pmax_diff"],
        axis=1
    )
    df_filt.loc[:, "pmax_final"] = df_filt.apply(
        lambda row: row["pmax_pmax_th"] 
            if (row["num_pings_pmax_th"] != -1) and (row["num_pings_pmax_th"] <= row["num_pings_pmax_diff"])
            else row["pmax_pmax_diff"],
        axis=1
    )
    df_filt.loc[:, "h_actual_final"] = df_filt.apply(
        lambda row: row["h_actual_pmax_th"] 
            if (row["num_pings_pmax_th"] != -1) and (row["num_pings_pmax_th"] <= row["num_pings_pmax_diff"])
            else row["h_actual_pmax_diff"],
        axis=1
    )
    df_filt.loc[:, "path_length_cumsum_final"] = df_filt.apply(
        lambda row: row["path_length_cumsum_pmax_th"] 
            if (row["num_pings_pmax_th"] != -1) and (row["num_pings_pmax_th"] <= row["num_pings_pmax_diff"])
            else row["path_length_cumsum_pmax_diff"],
        axis=1
    )
    return df_filt


def get_stats(x, y, print_stats=True):
    stats_dict = dict()
    stats_dict["two-sided"] = stats.mannwhitneyu(x, y, alternative="two-sided")
    stats_dict["less"] = stats.mannwhitneyu(x, y, alternative="less")
    stats_dict["greater"] = stats.mannwhitneyu(x, y, alternative="greater")

    if print_stats:
        print(f"Samples: left: {len(x):4d}    right: {len(y):4d}")
        print(f"Mean:    left: {np.mean(x):.3f}  right: {np.mean(y):.3f}")
        print(f"Median:  left: {np.median(x):.3f}  right: {np.median(y):.3f}")
        print("")
        print("MWU test")
        print(f" - different: {stats_dict["two-sided"].pvalue:.5f}")
        print(f" - less:      {stats_dict["less"].pvalue:.5f}")
        print(f" - greater:   {stats_dict["greater"].pvalue:.5f}")
        print("--------------------------------------------")

    return stats_dict


def bar_violinplot_success(
    cr, br, pm_all,
    df_x_all, df_y_all,
    x_str, y_str,
    comparison_type,  # simple, agent_higher, agent_lower
    stats_type,  # less, greater, two-sided
    colors, colors_dark,
    color_idx_left, color_idx_right,
    title_str_additional="",
    alpha = 0.65, print_stats=True
):

    # Set up fig
    fig, axes = plt.subplots(
        2, len(pm_all), figsize=(6, 4.5), sharey="row",
        gridspec_kw={'height_ratios': [1, 3], 'hspace': 0.05, "wspace": 0}  # Top row is 1/3 height of bottom
    )

    ping_bins_max = None
    for idx, (pm, df_x, df_y) in enumerate(zip(pm_all, df_x_all, df_y_all)):
        
        # Consolidate success and num_pings: 
        # take the smaller num_pings between pmax_th and pmax_diff criteria
        df_x_filter = consolidate_success_num_pings(df_x)
        df_y_filter = consolidate_success_num_pings(df_y)

        # Only cound successful runs
        x = df_x_filter[df_x_filter["success"]]["num_pings_final"]
        y = df_y_filter[df_y_filter["success"]]["num_pings_final"]


        # Print statistics
        stats_dict = get_stats(x, y, print_stats=print_stats)

        # Plotting success rate
        ax_s = axes[0, idx]
        bar_x = ax_s.bar(-0.5, len(x)/1000*100, width=0.4, color=colors[color_idx_left], alpha=alpha)
        bar_y = ax_s.bar(0.5, len(y)/1000*100, width=0.4, color=colors[color_idx_right], alpha=alpha)
        ax_s.text(-0.5, len(x)/1000*100+5, (f"({len(x)})"), ha="center", va="bottom", fontsize=9)
        ax_s.text(0.5, len(y)/1000*100+5, (f"({len(y)})"), ha="center", va="bottom", fontsize=9)
        ax_s.set_xlim(-1, 1)
        ax_s.set_ylim(0, 140)
        ax_s.set_title(f"{pm}")


        # Plotting number of pings to success
        ping_bins = np.arange(0, np.ceil(np.hstack((x, y)).max())+10, 5)
        if ping_bins_max is None:
            ping_bins_max = ping_bins
        else:
            ping_bins_max = ping_bins if ping_bins_max[-1] < ping_bins[-1] else ping_bins_max

        ax_n = axes[1, idx]

        x_hist, x_bins = np.histogram(x, bins=ping_bins, density=True)
        y_hist, y_bins = np.histogram(y, bins=ping_bins, density=True)
        x_median = np.median(x)
        y_median = np.median(y)

        ax_n.barh(x_bins[:-1], -x_hist, height=np.diff(x_bins), left=0, align="edge", edgecolor="w", color=colors[color_idx_left], alpha=alpha)
        ax_n.barh(y_bins[:-1], y_hist, height=np.diff(x_bins), left=0, align="edge", edgecolor="w", color=colors[color_idx_right], alpha=alpha)        
        ax_s.set_xticks([])
        ax_s.set_xticklabels("")

        # Get x-axis limits
        xlim = ax_n.get_xlim()
        xlim = np.max([max(xlim), -min(xlim)])
        xlim = xlim + 0.1*np.max(xlim)
        ax_n.set_xlim(-xlim, xlim)
        ax_n.axhline(x_median, xmin=0.0, xmax=0.5, color=colors_dark[color_idx_left], solid_capstyle="butt", lw=2.5)
        ax_n.axhline(y_median, xmin=0.51, xmax=1, color=colors_dark[color_idx_right], solid_capstyle="butt", lw=2.5)

        # label sample number
        ax_n.set_xticks([])
        ax_n.set_xticklabels("")

        # label p-value for alternative="less", "greater", or "two-sided"
        ax_n.text(
            0.975, 0.93, f"p={stats_dict[stats_type].pvalue:.0e}",
            ha="right", va="center",
            transform=ax_n.transAxes,
            fontsize=11, 
        )

        ax_n.tick_params(direction='in', which='both')
        if ping_bins_max[-1] > 600:
            ax_n.set_yticks(ping_bins_max[::32])
        elif ping_bins_max[-1] > 200:
            ax_n.set_yticks(ping_bins_max[::8])
        else:
            ax_n.set_yticks(ping_bins_max[::4])

    # make title line
    if comparison_type == "simple":
        axes[0, 0].text(
            -2.75, 1.04, r"$P_M=P_{FA}$=",
            ha="right", va="bottom",
            transform=ax_s.transAxes,
            fontsize=12,
        )
    elif comparison_type == "pm_sweep":
        axes[0, 0].text(
            -2.75, 1.04, r"$P_M$=",
            ha="right", va="bottom",
            transform=ax_s.transAxes,
            fontsize=12,
        )
    elif comparison_type == "pfa_sweep":
        axes[0, 0].text(
            -2.75, 1.04, r"$P_{FA}$=",
            ha="right", va="bottom",
            transform=ax_s.transAxes,
            fontsize=12,
        )
    elif comparison_type == "agent_higher":
        axes[0, 0].text(
            -2.75, 1.02, r"$P_M^A=P_{FA}^A$=",
            ha="right", va="bottom",
            transform=ax_s.transAxes,
            fontsize=12,
        )
    elif comparison_type == "agent_lower":
        axes[0, 0].text(
            -2.75, 1.02, r"$P_M^T=P_{FA}^T$=",
            ha="right", va="bottom",
            transform=ax_s.transAxes,
            fontsize=12,
        )
    else:
        raise ValueError(f"Invalid comparison type: {comparison_type}")

    alpha_grid = get_beam_search_space_ratio(cr, br)
    axes[0, 0].set_ylabel("Success\nrate (%)", fontsize=12)    
    axes[1, 0].set_ylabel("Number of pings\nto success", fontsize=12)
    
    # Make specific title str here because problem of passing in raw strings for latex
    if comparison_type == "pm_sweep":
        title_str = (
            fr"$\mathbfit{{\alpha}}$={alpha_grid:.3f}  "
            fr"($\mathbfit{{r_A}}$={cr}, $\mathbfit{{r_B}}$={br}); "
            r"$\mathbfit{{P_{FA}}}$=0"
        )
    elif comparison_type == "pfa_sweep":
        title_str = (
            fr"$\mathbfit{{\alpha}}$={alpha_grid:.3f}  "
            fr"($\mathbfit{{r_A}}$={cr}, $\mathbfit{{r_B}}$={br}); "
            r"$\mathbfit{{P_M}}$=0"
        )
    else:
        title_str = (
            fr"$\mathbfit{{\alpha}}$={alpha_grid:.3f}  "
            fr"($\mathbfit{{r_A}}$={cr}, $\mathbfit{{r_B}}$={br})"
        )

    if title_str_additional:
        title_str += f";  {title_str_additional}"
    fig.text(0.5, 0.98, title_str, ha="center", va="center", fontsize=12, fontweight="bold")
    # fig.text(0.5, 0.97, fr"Space radius={cr}, footprint radius={br} ($\alpha$={alpha_grid:.3f})",
    #     ha="center", va="center", fontsize=12)

    # Add single horizontal legend below all panels
    fig.legend(
        handles=[bar_x, bar_y],
        labels=[x_str, y_str],
        loc='lower center',
        ncol=2,
        bbox_to_anchor=(0.5, 0.005),
        # frameon=False,
        fontsize=11
    )    

    # plt.tight_layout(pad=0)
    plt.show()

    return fig


def bar_violinplot_success_beam_dep(
    cr, br, pm_all,
    df_x_all, df_y_all,
    x_str, y_str,
    stats_type,  # less, greater, two-sided
    colors, colors_dark,
    color_idx_left, color_idx_right,
    title_str_additional="",
    alpha = 0.65, print_stats=True,
    crowded_bars=False,
):

    # Set up fig
    fig, axes = plt.subplots(
        2, len(pm_all), figsize=(6, 4.5), sharey="row",
        gridspec_kw={'height_ratios': [1, 3], 'hspace': 0.05, "wspace": 0}  # Top row is 1/3 height of bottom
    )

    ping_bins_max = None
    for idx, (pm, df_x, df_y) in enumerate(zip(pm_all, df_x_all, df_y_all)):

        # Consolidate success and num_pings: 
        # take the smaller num_pings between pmax_th and pmax_diff criteria
        df_x_filter = consolidate_success_num_pings(df_x)
        df_y_filter = consolidate_success_num_pings(df_y)

        # Only cound successful runs
        x = df_x_filter[df_x_filter["success"]]["num_pings_final"]
        y = df_y_filter[df_y_filter["success"]]["num_pings_final"]


        # Print statistics
        stats_dict = get_stats(x, y, print_stats=print_stats)

        # Plotting success rate
        ax_s = axes[0, idx]
        bar_x = ax_s.bar(-0.5, len(x)/1000*100, width=0.4, color=colors[color_idx_left], alpha=alpha)
        bar_y = ax_s.bar(0.5, len(y)/1000*100, width=0.4, color=colors[color_idx_right], alpha=alpha)
        ax_s.text(-0.5, len(x)/1000*100+5, (f"({len(x)})"), ha="center", va="bottom", fontsize=9)
        ax_s.text(0.5, len(y)/1000*100+5, (f"({len(y)})"), ha="center", va="bottom", fontsize=9)
        ax_s.set_xlim(-1, 1)
        ax_s.set_ylim(0, 140)
        ax_s.set_title(f"{pm}")


        # Plotting number of pings to success
        ping_bins = np.arange(0, np.ceil(np.hstack((x, y)).max())+10, 5)
        if ping_bins_max is None:
            ping_bins_max = ping_bins
        else:
            ping_bins_max = ping_bins if ping_bins_max[-1] < ping_bins[-1] else ping_bins_max
    
        ax_n = axes[1, idx]

        x_hist, x_bins = np.histogram(x, bins=ping_bins, density=True)
        y_hist, y_bins = np.histogram(y, bins=ping_bins, density=True)
        x_hist = x_hist / x_hist.max() if x_hist.max() > 0 else x_hist  # normalize to max distribution value
        y_hist = y_hist / y_hist.max() if y_hist.max() > 0 else y_hist
        x_median = np.median(x)
        y_median = np.median(y)

        if crowded_bars:
            ax_n.barh(x_bins[:-1], -x_hist, height=np.diff(x_bins), left=0, align="edge", edgecolor=colors[color_idx_left], color=colors[color_idx_left], alpha=alpha-0.3)
            ax_n.barh(y_bins[:-1], y_hist, height=np.diff(x_bins), left=0, align="edge", edgecolor=colors[color_idx_right], color=colors[color_idx_right], alpha=alpha-0.3)
        else:
            ax_n.barh(x_bins[:-1], -x_hist, height=np.diff(x_bins), left=0, align="edge", edgecolor="w", color=colors[color_idx_left], alpha=alpha)
            ax_n.barh(y_bins[:-1], y_hist, height=np.diff(x_bins), left=0, align="edge", edgecolor="w", color=colors[color_idx_right], alpha=alpha)
        ax_s.set_xticks([])
        ax_s.set_xticklabels("")

        # Get x-axis limits
        xlim = ax_n.get_xlim()
        xlim = np.max([max(xlim), -min(xlim)])
        xlim = xlim + 0.1*np.max(xlim)
        ax_n.set_xlim(-xlim, xlim)
        ax_n.axhline(x_median, xmin=0.0, xmax=0.5, color=colors_dark[color_idx_left], solid_capstyle="butt", lw=1.5)
        ax_n.axhline(y_median, xmin=0.51, xmax=1, color=colors_dark[color_idx_right], solid_capstyle="butt", lw=1.5)
        if crowded_bars:
            ax_n.axvline(0, color="w", lw=1.5)

        # label sample number
        ax_n.set_xticks([])
        ax_n.set_xticklabels("")

        # label p-value for alternative="less", "greater", or "two-sided"
        ax_n.text(
            0.975, 0.93, f"p={stats_dict[stats_type].pvalue:.0e}",
            ha="right", va="center",
            transform=ax_n.transAxes,
            fontsize=11, 
        )

        ax_n.tick_params(direction='in', which='both')
        if ping_bins_max[-1] > 600:
            ax_n.set_yticks(ping_bins_max[::32])
        elif ping_bins_max[-1] > 200:
            ax_n.set_yticks(ping_bins_max[::8])
        else:
            ax_n.set_yticks(ping_bins_max[::4])

    axes[0, 0].text(
        -2.75, 1.03, r"$P_M=P_{FA}$=",  # for 4 panels
        ha="right", va="bottom",
        transform=ax_s.transAxes,
        fontsize=12,
    )

    alpha_grid = get_beam_search_space_ratio(cr, br)
    axes[0, 0].set_ylabel("Success\nrate (%)", fontsize=12)    
    axes[1, 0].set_ylabel("Number of pings\nto success", fontsize=12)
    title_str = (
        fr"$\mathbfit{{\alpha}}$={alpha_grid:.3f}  "
        fr"($\mathbfit{{r_A}}$={cr}, $\mathbfit{{r_B}}$={br})"
    )
    if title_str_additional:
        title_str += f";  {title_str_additional}"
    fig.text(0.5, 0.97, title_str, ha="center", va="center", fontsize=12, fontweight="bold")

    # Add single horizontal legend below all panels
    fig.legend(
        handles=[bar_x, bar_y],
        labels=[x_str, y_str],
        loc='lower center',
        ncol=2,
        bbox_to_anchor=(0.5, 0.005),
        # frameon=False,
        fontsize=11
    )    

    # plt.tight_layout(pad=0)
    plt.show()

    return fig


def bar_violinplot_last_pmax(
    cr, br, pm_all,
    df_x_all, df_y_all,
    x_str, y_str,
    comparison_type,  # simple, agent_higher, agent_lower
    stats_type,  # less, greater, two-sided
    colors, colors_dark,
    color_idx_left, color_idx_right,
    bar_step = 0.1,
    alpha = 0.65, print_stats=True
):

    # Set up fig
    fig, axes = plt.subplots(
        2, len(pm_all), figsize=(6, 4.5), sharey="row",
        gridspec_kw={'height_ratios': [1, 3], 'hspace': 0.05, "wspace": 0}  # Top row is 1/3 height of bottom
    )


    for idx, (pm, df_x, df_y) in enumerate(zip(pm_all, df_x_all, df_y_all)):
        
        # Consolidate success and num_pings: 
        # take the smaller num_pings between pmax_th and pmax_diff criteria
        df_x_filter = consolidate_success_num_pings(df_x)
        df_y_filter = consolidate_success_num_pings(df_y)

        # Only cound successful runs
        x = df_x_filter[df_x_filter["success"]]["pmax_final"]
        y = df_y_filter[df_y_filter["success"]]["pmax_final"]

        # Print statistics
        stats_dict = get_stats(x, y, print_stats=print_stats)

        # Plotting success rate
        ax_s = axes[0, idx]
        bar_x = ax_s.bar(-0.5, len(x)/1000*100, width=0.4, color=colors[color_idx_left], alpha=alpha)
        bar_y = ax_s.bar(0.5, len(y)/1000*100, width=0.4, color=colors[color_idx_right], alpha=alpha)
        ax_s.text(-0.5, len(x)/1000*100+5, (f"({len(x)})"), ha="center", va="bottom", fontsize=9)
        ax_s.text(0.5, len(y)/1000*100+5, (f"({len(y)})"), ha="center", va="bottom", fontsize=9)
        ax_s.set_xlim(-1, 1)
        ax_s.set_ylim(0, 140)
        ax_s.set_title(f"{pm}")


        # Plotting last pmax distribution
        pmax_bins = np.arange(0, 1+bar_step, bar_step)

        ax_n = axes[1, idx]

        x_hist, x_bins = np.histogram(x, bins=pmax_bins, density=True)
        y_hist, y_bins = np.histogram(y, bins=pmax_bins, density=True)
        x_median = np.median(x)
        y_median = np.median(y)

        ax_n.barh(x_bins[:-1], -x_hist, height=np.diff(x_bins), left=0, align="edge", edgecolor="w", color=colors[color_idx_left], alpha=alpha)
        ax_n.barh(y_bins[:-1], y_hist, height=np.diff(x_bins), left=0, align="edge", edgecolor="w", color=colors[color_idx_right], alpha=alpha)        
        ax_s.set_xticks([])
        ax_s.set_xticklabels("")

        # Get x-axis limits
        xlim = ax_n.get_xlim()
        xlim = np.max([max(xlim), -min(xlim)])
        xlim = xlim + 0.1*np.max(xlim)
        ax_n.set_xlim(-xlim, xlim)
        ax_n.axhline(x_median, xmin=0.0, xmax=0.5, color=colors_dark[color_idx_left], solid_capstyle="butt", lw=2.5)
        ax_n.axhline(y_median, xmin=0.51, xmax=1, color=colors_dark[color_idx_right], solid_capstyle="butt", lw=2.5)

        # label sample number
        ax_n.set_xticks([])
        ax_n.set_xticklabels("")

        # label p-value for alternative="less", "greater", or "two-sided"
        ax_n.text(
            0.975, 0.07, f"p={stats_dict[stats_type].pvalue:.0e}",
            ha="right", va="center",
            transform=ax_n.transAxes,
            fontsize=11, 
        )

        ax_n.tick_params(direction='in', which='both')

    # make title line
    if comparison_type == "simple":
        axes[0, 0].text(
            -2.75, 1.04, r"$P_M=P_{FA}$=",
            ha="right", va="bottom",
            transform=ax_s.transAxes,
            fontsize=12,
        )
    elif comparison_type == "agent_higher":
        axes[0, 0].text(
            -2.75, 1.02, r"$P_M^A=P_{FA}^A$=",
            ha="right", va="bottom",
            transform=ax_s.transAxes,
            fontsize=12,
        )
    elif comparison_type == "agent_lower":
        axes[0, 0].text(
            -2.75, 1.02, r"$P_M^T=P_{FA}^T$=",
            ha="right", va="bottom",
            transform=ax_s.transAxes,
            fontsize=12,
        )
    else:
        raise ValueError(f"Invalid comparison type: {comparison_type}")

    alpha_grid = get_beam_search_space_ratio(cr, br)
    axes[0, 0].set_ylabel("Success\nrate (%)", fontsize=12)    
    axes[1, 0].set_ylabel(r"$P_{{max}}$ at the end of search", fontsize=12)
    fig.text(
        0.5, 0.98,
        fr"$\mathbfit{{\alpha}}$={alpha_grid:.3f}  "
        fr"($\mathbfit{{r_A}}$={cr}, $\mathbfit{{r_B}}$={br})",
        ha="center", va="center", fontsize=12, fontweight="bold")

    # Add single horizontal legend below all panels
    fig.legend(
        handles=[bar_x, bar_y],
        labels=[x_str, y_str],
        loc='lower center',
        ncol=2,
        bbox_to_anchor=(0.5, 0.005),
        # frameon=False,
        fontsize=11
    )    

    # plt.tight_layout(pad=0)
    plt.show()

    return fig


def bar_violinplot_last_h_actual(
    cr, br, pm_all,
    df_x_all, df_y_all,
    x_str, y_str,
    comparison_type,  # simple, agent_higher, agent_lower
    stats_type,  # less, greater, two-sided
    colors, colors_dark,
    color_idx_left, color_idx_right,
    crowded_bars=False,
    bar_step = 0.1,
    alpha = 0.65, print_stats=True
):

    # Set up fig
    fig, axes = plt.subplots(
        2, len(pm_all), figsize=(6, 4.5), sharey="row",
        gridspec_kw={'height_ratios': [1, 3], 'hspace': 0.05, "wspace": 0}  # Top row is 1/3 height of bottom
    )

    h_actual_bins_max = None
    for idx, (pm, df_x, df_y) in enumerate(zip(pm_all, df_x_all, df_y_all)):
        
        # Consolidate success and num_pings: 
        # take the smaller num_pings between pmax_th and pmax_diff criteria
        df_x_filter = consolidate_success_num_pings(df_x)
        df_y_filter = consolidate_success_num_pings(df_y)

        # Only cound successful runs
        x = df_x_filter[df_x_filter["success"]]["h_actual_final"]
        y = df_y_filter[df_y_filter["success"]]["h_actual_final"]

        # Print statistics
        stats_dict = get_stats(x, y, print_stats=print_stats)

        # Plotting success rate
        ax_s = axes[0, idx]
        bar_x = ax_s.bar(-0.5, len(x)/1000*100, width=0.4, color=colors[color_idx_left], alpha=alpha)
        bar_y = ax_s.bar(0.5, len(y)/1000*100, width=0.4, color=colors[color_idx_right], alpha=alpha)
        ax_s.text(-0.5, len(x)/1000*100+5, (f"({len(x)})"), ha="center", va="bottom", fontsize=9)
        ax_s.text(0.5, len(y)/1000*100+5, (f"({len(y)})"), ha="center", va="bottom", fontsize=9)
        ax_s.set_xlim(-1, 1)
        ax_s.set_ylim(0, 140)
        ax_s.set_title(f"{pm}")


        # Plotting last h_actual distribution
        h_actual_bins = np.arange(0, np.ceil(np.hstack((x, y)).max()/bar_step)*bar_step+bar_step, bar_step)
        if h_actual_bins_max is None:
            h_actual_bins_max = h_actual_bins
        else:
            h_actual_bins_max = h_actual_bins if h_actual_bins_max[-1] < h_actual_bins[-1] else h_actual_bins_max


        ax_n = axes[1, idx]

        x_hist, x_bins = np.histogram(x, bins=h_actual_bins, density=True)
        y_hist, y_bins = np.histogram(y, bins=h_actual_bins, density=True)
        x_median = np.median(x)
        y_median = np.median(y)

        if crowded_bars:
            ax_n.barh(x_bins[:-1], -x_hist, height=np.diff(x_bins), left=0, align="edge", edgecolor=colors[color_idx_left], color=colors[color_idx_left], alpha=alpha-0.3)
            ax_n.barh(y_bins[:-1], y_hist, height=np.diff(x_bins), left=0, align="edge", edgecolor=colors[color_idx_right], color=colors[color_idx_right], alpha=alpha-0.3)
        else:
            ax_n.barh(x_bins[:-1], -x_hist, height=np.diff(x_bins), left=0, align="edge", edgecolor="w", color=colors[color_idx_left], alpha=alpha)
            ax_n.barh(y_bins[:-1], y_hist, height=np.diff(x_bins), left=0, align="edge", edgecolor="w", color=colors[color_idx_right], alpha=alpha)
        ax_s.set_xticks([])
        ax_s.set_xticklabels("")

        # Get x-axis limits
        xlim = ax_n.get_xlim()
        xlim = np.max([max(xlim), -min(xlim)])
        xlim = xlim + 0.1*np.max(xlim)
        ax_n.set_xlim(-xlim, xlim)
        if crowded_bars:
            ax_n.axhline(x_median, xmin=0.0, xmax=0.5, color=colors_dark[color_idx_left], solid_capstyle="butt", lw=1.5)
            ax_n.axhline(y_median, xmin=0.51, xmax=1, color=colors_dark[color_idx_right], solid_capstyle="butt", lw=1.5)
            ax_n.axvline(0, color="w", lw=1.5)
        else:
            ax_n.axhline(x_median, xmin=0.0, xmax=0.5, color=colors_dark[color_idx_left], solid_capstyle="butt", lw=2.5)
            ax_n.axhline(y_median, xmin=0.51, xmax=1, color=colors_dark[color_idx_right], solid_capstyle="butt", lw=2.5)

        # label sample number
        ax_n.set_xticks([])
        ax_n.set_xticklabels("")

        # label p-value for alternative="less", "greater", or "two-sided"
        ax_n.text(
            0.975, 0.93, f"p={stats_dict[stats_type].pvalue:.0e}",
            ha="right", va="center",
            transform=ax_n.transAxes,
            fontsize=11, 
        )

        ax_n.tick_params(direction='in', which='both')

    # make title line
    if comparison_type == "simple":
        axes[0, 0].text(
            -2.75, 1.04, r"$P_M=P_{FA}$=",
            ha="right", va="bottom",
            transform=ax_s.transAxes,
            fontsize=12,
        )
    elif comparison_type == "agent_higher":
        axes[0, 0].text(
            -2.75, 1.02, r"$P_M^A=P_{FA}^A$=",
            ha="right", va="bottom",
            transform=ax_s.transAxes,
            fontsize=12,
        )
    elif comparison_type == "agent_lower":
        axes[0, 0].text(
            -2.75, 1.02, r"$P_M^T=P_{FA}^T$=",
            ha="right", va="bottom",
            transform=ax_s.transAxes,
            fontsize=12,
        )
    else:
        raise ValueError(f"Invalid comparison type: {comparison_type}")
    alpha_grid = get_beam_search_space_ratio(cr, br)
    axes[0, 0].set_ylabel("Success\nrate (%)", fontsize=12)    
    axes[1, 0].set_ylabel("Entropy at the end of search", fontsize=12)
    fig.text(
        0.5, 0.98,
        fr"$\mathbfit{{\alpha}}$={alpha_grid:.3f}  "
        fr"($\mathbfit{{r_A}}$={cr}, $\mathbfit{{r_B}}$={br})",
        ha="center", va="center", fontsize=12, fontweight="bold")

    # Add single horizontal legend below all panels
    fig.legend(
        handles=[bar_x, bar_y],
        labels=[x_str, y_str],
        loc='lower center',
        ncol=2,
        bbox_to_anchor=(0.5, 0.005),
        # frameon=False,
        fontsize=11
    )    

    # plt.tight_layout(pad=0)
    plt.show()

    return fig


def bar_violinplot_path_length(
    cr, br, pm_all,
    df_x_all, df_y_all,
    x_str, y_str,
    comparison_type,  # simple, agent_higher, agent_lower
    stats_type,  # less, greater, two-sided
    colors, colors_dark,
    color_idx_left, color_idx_right,
    title_str_additional="",
    alpha = 0.65, print_stats=True
):

    # Set up fig
    fig, axes = plt.subplots(
        2, len(pm_all), figsize=(6, 4.5), sharey="row",
        gridspec_kw={'height_ratios': [1, 3], 'hspace': 0.05, "wspace": 0}  # Top row is 1/3 height of bottom
    )

    ping_bins_max = None
    x_all = []
    y_all = []
    for idx, (pm, df_x, df_y) in enumerate(zip(pm_all, df_x_all, df_y_all)):
        
        # Consolidate success and num_pings: 
        # take the smaller num_pings between pmax_th and pmax_diff criteria
        df_x_filter = consolidate_success_num_pings(df_x)
        df_y_filter = consolidate_success_num_pings(df_y)

        # Only cound successful runs
        x = df_x_filter[df_x_filter["success"]]["path_length_cumsum_final"]
        y = df_y_filter[df_y_filter["success"]]["path_length_cumsum_final"]

        # Get ping_bins_max to determine if plotted as crowded bars
        ping_bins = np.arange(0, np.ceil(np.hstack((x, y)).max())+10, 5)
        if ping_bins_max is None:
            ping_bins_max = ping_bins
        else:
            ping_bins_max = ping_bins if ping_bins_max[-1] < ping_bins[-1] else ping_bins_max

        x_all.append(x)
        y_all.append(y)

    for idx, (pm, x, y) in enumerate(zip(pm_all, x_all, y_all)):

        # Print statistics
        stats_dict = get_stats(x, y, print_stats=print_stats)

        # Plotting success rate
        ax_s = axes[0, idx]
        bar_x = ax_s.bar(-0.5, len(x)/1000*100, width=0.4, color=colors[color_idx_left], alpha=alpha)
        bar_y = ax_s.bar(0.5, len(y)/1000*100, width=0.4, color=colors[color_idx_right], alpha=alpha)
        ax_s.text(-0.5, len(x)/1000*100+5, (f"({len(x)})"), ha="center", va="bottom", fontsize=9)
        ax_s.text(0.5, len(y)/1000*100+5, (f"({len(y)})"), ha="center", va="bottom", fontsize=9)
        ax_s.set_xlim(-1, 1)
        ax_s.set_ylim(0, 140)
        ax_s.set_title(f"{pm}")

        # Determine if to plot as crowded bars
        crowded_bars = ping_bins_max[-1] > 400

        # Plotting number of pings to success
        ax_n = axes[1, idx]

        x_hist, x_bins = np.histogram(x, bins=ping_bins, density=True)
        y_hist, y_bins = np.histogram(y, bins=ping_bins, density=True)
        x_median = np.median(x)
        y_median = np.median(y)

        if crowded_bars:
            ax_n.barh(x_bins[:-1], -x_hist, height=np.diff(x_bins), left=0, align="edge", edgecolor=colors[color_idx_left], color=colors[color_idx_left], alpha=alpha-0.3)
            ax_n.barh(y_bins[:-1], y_hist, height=np.diff(x_bins), left=0, align="edge", edgecolor=colors[color_idx_right], color=colors[color_idx_right], alpha=alpha-0.3)
        else:
            ax_n.barh(x_bins[:-1], -x_hist, height=np.diff(x_bins), left=0, align="edge", edgecolor="w", color=colors[color_idx_left], alpha=alpha)
            ax_n.barh(y_bins[:-1], y_hist, height=np.diff(x_bins), left=0, align="edge", edgecolor="w", color=colors[color_idx_right], alpha=alpha)

        ax_s.set_xticks([])
        ax_s.set_xticklabels("")

        # Get x-axis limits
        xlim = ax_n.get_xlim()
        xlim = np.max([max(xlim), -min(xlim)])
        xlim = xlim + 0.1*np.max(xlim)
        ax_n.set_xlim(-xlim, xlim)
        if crowded_bars:
            ax_n.axhline(x_median, xmin=0.0, xmax=0.5, color=colors_dark[color_idx_left], solid_capstyle="butt", lw=2)
            ax_n.axhline(y_median, xmin=0.51, xmax=1, color=colors_dark[color_idx_right], solid_capstyle="butt", lw=2)
            ax_n.axvline(0, color="w", lw=1.5)
        else:
            ax_n.axhline(x_median, xmin=0.0, xmax=0.5, color=colors_dark[color_idx_left], solid_capstyle="butt", lw=2.5)
            ax_n.axhline(y_median, xmin=0.51, xmax=1, color=colors_dark[color_idx_right], solid_capstyle="butt", lw=2.5)

        # label sample number
        ax_n.set_xticks([])
        ax_n.set_xticklabels("")

        # label p-value for alternative="less", "greater", or "two-sided"
        ax_n.text(
            0.975, 0.93, f"p={stats_dict[stats_type].pvalue:.0e}",
            ha="right", va="center",
            transform=ax_n.transAxes,
            fontsize=11, 
        )

        ax_n.tick_params(direction='in', which='both')
        if ping_bins_max[-1] > 600:
            ax_n.set_yticks(ping_bins_max[::32])
            ax_n.set_ylim(-20, ping_bins_max[-1]+20)
        elif ping_bins_max[-1] > 200:
            ax_n.set_yticks(ping_bins_max[::8])
            ax_n.set_ylim(-10, ping_bins_max[-1]+10)
        else:
            ax_n.set_yticks(ping_bins_max[::4])
            ax_n.set_ylim(-5, ping_bins_max[-1]+5)

    # make title line
    if comparison_type == "simple":
        axes[0, 0].text(
            -2.75, 1.04, r"$P_M=P_{FA}$=",
            ha="right", va="bottom",
            transform=ax_s.transAxes,
            fontsize=12,
        )
    elif comparison_type == "agent_higher":
        axes[0, 0].text(
            -2.75, 1.02, r"$P_M^A=P_{FA}^A$=",
            ha="right", va="bottom",
            transform=ax_s.transAxes,
            fontsize=12,
        )
    elif comparison_type == "agent_lower":
        axes[0, 0].text(
            -2.75, 1.02, r"$P_M^T=P_{FA}^T$=",
            ha="right", va="bottom",
            transform=ax_s.transAxes,
            fontsize=12,
        )
    else:
        raise ValueError(f"Invalid comparison type: {comparison_type}")

    alpha_grid = get_beam_search_space_ratio(cr, br)
    axes[0, 0].set_ylabel("Success\nrate (%)", fontsize=12)    
    axes[1, 0].set_ylabel("Cumulative path length\nat the end of search", fontsize=12)
    title_str = (
        fr"$\mathbfit{{\alpha}}$={alpha_grid:.3f}  "
        fr"($\mathbfit{{r_A}}$={cr}, $\mathbfit{{r_B}}$={br})"
    )
    if title_str_additional:
        title_str += f";  {title_str_additional}"
    fig.text(0.5, 0.98, title_str, ha="center", va="center", fontsize=12, fontweight="bold")
    # fig.text(0.5, 0.97, fr"Space radius={cr}, footprint radius={br} ($\alpha$={alpha_grid:.3f})",
    #     ha="center", va="center", fontsize=12)

    # Add single horizontal legend below all panels
    fig.legend(
        handles=[bar_x, bar_y],
        labels=[x_str, y_str],
        loc='lower center',
        ncol=2,
        bbox_to_anchor=(0.5, 0.005),
        # frameon=False,
        fontsize=11
    )    

    # plt.tight_layout(pad=0)
    plt.show()

    return fig


def bar_violinplot_success_beam_move_res(
    cr, br, bmr,
    pm_all,
    df_x_all, df_y_all,
    x_str, y_str,
    comparison_type,  # simple, agent_higher, agent_lower
    stats_type,  # less, greater, two-sided
    colors, colors_dark,
    color_idx_left, color_idx_right,
    title_str_additional="",
    alpha = 0.65, print_stats=True
):

    # Set up fig
    fig, axes = plt.subplots(
        2, len(pm_all), figsize=(6, 4.5), sharey="row",
        gridspec_kw={'height_ratios': [1, 3], 'hspace': 0.05, "wspace": 0}  # Top row is 1/3 height of bottom
    )

    ping_bins_max = None
    for idx, (pm, df_x, df_y) in enumerate(zip(pm_all, df_x_all, df_y_all)):
        
        # Consolidate success and num_pings: 
        # take the smaller num_pings between pmax_th and pmax_diff criteria
        df_x_filter = consolidate_success_num_pings(df_x)
        df_y_filter = consolidate_success_num_pings(df_y)

        # Only cound successful runs
        x = df_x_filter[df_x_filter["success"]]["num_pings_final"]
        y = df_y_filter[df_y_filter["success"]]["num_pings_final"]


        # Print statistics
        stats_dict = get_stats(x, y, print_stats=print_stats)

        # Plotting success rate
        ax_s = axes[0, idx]
        bar_x = ax_s.bar(-0.5, len(x)/1000*100, width=0.4, color=colors[color_idx_left], alpha=alpha)
        bar_y = ax_s.bar(0.5, len(y)/1000*100, width=0.4, color=colors[color_idx_right], alpha=alpha)
        ax_s.text(-0.5, len(x)/1000*100+5, (f"({len(x)})"), ha="center", va="bottom", fontsize=9)
        ax_s.text(0.5, len(y)/1000*100+5, (f"({len(y)})"), ha="center", va="bottom", fontsize=9)
        ax_s.set_xlim(-1, 1)
        ax_s.set_ylim(0, 140)
        ax_s.set_title(f"{pm}")


        # Plotting number of pings to success
        ping_bins = np.arange(0, np.ceil(np.hstack((x, y)).max())+10, 5)
        if ping_bins_max is None:
            ping_bins_max = ping_bins
        else:
            ping_bins_max = ping_bins if ping_bins_max[-1] < ping_bins[-1] else ping_bins_max

        ax_n = axes[1, idx]

        x_hist, x_bins = np.histogram(x, bins=ping_bins, density=True)
        y_hist, y_bins = np.histogram(y, bins=ping_bins, density=True)
        x_median = np.median(x)
        y_median = np.median(y)

        ax_n.barh(x_bins[:-1], -x_hist, height=np.diff(x_bins), left=0, align="edge", edgecolor="w", color=colors[color_idx_left], alpha=alpha)
        ax_n.barh(y_bins[:-1], y_hist, height=np.diff(x_bins), left=0, align="edge", edgecolor="w", color=colors[color_idx_right], alpha=alpha)        
        ax_s.set_xticks([])
        ax_s.set_xticklabels("")

        # Get x-axis limits
        xlim = ax_n.get_xlim()
        xlim = np.max([max(xlim), -min(xlim)])
        xlim = xlim + 0.1*np.max(xlim)
        ax_n.set_xlim(-xlim, xlim)
        ax_n.axhline(x_median, xmin=0.0, xmax=0.5, color=colors_dark[color_idx_left], solid_capstyle="butt", lw=2.5)
        ax_n.axhline(y_median, xmin=0.51, xmax=1, color=colors_dark[color_idx_right], solid_capstyle="butt", lw=2.5)

        # label sample number
        ax_n.set_xticks([])
        ax_n.set_xticklabels("")

        # label p-value for alternative="less", "greater", or "two-sided"
        ax_n.text(
            0.975, 0.93, f"p={stats_dict[stats_type].pvalue:.0e}",
            ha="right", va="center",
            transform=ax_n.transAxes,
            fontsize=11, 
        )

        ax_n.tick_params(direction='in', which='both')
        if ping_bins_max[-1] > 600:
            ax_n.set_yticks(ping_bins_max[::32])
        elif ping_bins_max[-1] > 200:
            ax_n.set_yticks(ping_bins_max[::8])
        else:
            ax_n.set_yticks(ping_bins_max[::4])

    # make title line
    if comparison_type == "simple":
        axes[0, 0].text(
            -2.75, 1.04, r"$P_M=P_{FA}$=",
            ha="right", va="bottom",
            transform=ax_s.transAxes,
            fontsize=12,
        )
    elif comparison_type == "agent_higher":
        axes[0, 0].text(
            -2.75, 1.02, r"$P_M^A=P_{FA}^A$=",
            ha="right", va="bottom",
            transform=ax_s.transAxes,
            fontsize=12,
        )
    elif comparison_type == "agent_lower":
        axes[0, 0].text(
            -2.75, 1.02, r"$P_M^T=P_{FA}^T$=",
            ha="right", va="bottom",
            transform=ax_s.transAxes,
            fontsize=12,
        )
    else:
        raise ValueError(f"Invalid comparison type: {comparison_type}")

    alpha_grid = get_beam_search_space_ratio(cr, br)
    axes[0, 0].set_ylabel("Success\nrate (%)", fontsize=12)    
    axes[1, 0].set_ylabel("Number of pings\nto success", fontsize=12)
    title_str = (
        fr"$\mathbfit{{\alpha}}$={alpha_grid:.3f}  "
        fr"($\mathbfit{{r_A}}$={cr}, $\mathbfit{{r_B}}$={br}, $\mathbfit{{r_M}}$={bmr})"
    )
    if title_str_additional:
        title_str += f";  {title_str_additional}"
    fig.text(0.5, 0.98, title_str, ha="center", va="center", fontsize=12, fontweight="bold")
    # fig.text(0.5, 0.97, fr"Space radius={cr}, footprint radius={br} ($\alpha$={alpha_grid:.3f})",
    #     ha="center", va="center", fontsize=12)

    # Add single horizontal legend below all panels
    fig.legend(
        handles=[bar_x, bar_y],
        labels=[x_str, y_str],
        loc='lower center',
        ncol=2,
        bbox_to_anchor=(0.5, 0.005),
        # frameon=False,
        fontsize=11
    )    

    # plt.tight_layout(pad=0)
    plt.show()

    return fig