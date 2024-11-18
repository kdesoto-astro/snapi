"""File holding classifier and classification results base classes."""
import numpy as np

from ..formatter import Formatter
from ..base_classes import Base

class ClassifierResult(Base):
    """Stores information about a trained classifier's output on a dataset.
    Mostly a pd.DataFrame + metadata.
    """
    def __init__(self, df) -> None:
        super().__init__()
        self._df = df
        
    def generate_roc_curve(self, fig, ax, ax1, formatter=Formatter()):
        """Generate a combined ROC curve of all SN classes.
        """
        legend_lines = []
        
        for ref_label in np.unique(self.true_labels):
            t, fpr, tpr, tpr_err = self._roc_curve_w_uncertainties(ref_label)
            idx_50 = np.argmin((t - 0.5) ** 2)
            (legend_line,) = ax1.step(fpr, tpr, label=ref_label, c=formatter.edge_color, where='post')
            ax1.fill_between(
                fpr, tpr-tpr_err, tpr+tpr_err,
                color=colors[ref_class], step='post', alpha=0.2
            )
            ax2.step(fpr, tpr, label=ref_label, c=formatter.edge_color, where='post')
            legend_lines.append(legend_line)
            ax2.fill_between(
                fpr, tpr-tpr_err, tpr+tpr_err,
                color=colors[ref_class], step='post', alpha=0.2
            )
            ax2.scatter(
                (fpr[idx_50] + fpr[idx_50 + 1]) / 2, tpr[idx_50],
                color=formatter.edge_color, s=100, marker="d", zorder=1000
            )
            formatter.rotate_colors()
            formatter.rotate_markers()

        ax1.plot(
            [0, 1], [0,1],
            c="#BBBBBB", linestyle='dotted'
        )

        for ax_i in double_axes:
            ratio = 1.5
            x_left, x_right = ax_i.get_xlim()
            y_low, y_high = ax_i.get_ylim()

            ax_i.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)
            ax_i.yaxis.set_minor_locator(AutoMinorLocator())
            ax_i.xaxis.set_minor_locator(AutoMinorLocator())
            ax_i.set_xlabel("False Positive Rate")

        legend_keys = list(labels_to_classes.keys())
        fig.legend(legend_lines, legend_keys, loc="lower center", ncol=3)
        
        formatter.reset_colors()
        formatter.reset_markers()
        
        return fig, ax

        
    def _roc_curve_w_uncertainties(self, ref_label):
        """Incorporate K-fold uncertainties."""
        threshholds = np.linspace(0, 1, num=1000)
        all_tpr = []
        all_fpr = []
        all_thresh = []

        df_copy = self._df.copy()
        
        for t in threshholds:
            _, df_sums = self._calculate_binary_threshold_vals(df_copy, ref_label, t)
            all_tpr.extend(df_sums['purity'].to_numpy().tolist())
            all_fpr.extend(df_sums['purity'].to_numpy().tolist())
            all_thresh.extend([t,] * len(df_sums))
            
        fpr_bin_edges = np.unique(histedges_equalN(all_fpr, 30))
        fpr_centers = (fpr_bin_edges[1:] + fpr_bin_edges[:-1]) / 2
        tpr, _, _ = binned_statistic(all_fpr, all_tpr, statistic='mean', bins=fpr_bin_edges)
        tpr_err, _, _ = binned_statistic(all_fpr, all_tpr, statistic='std', bins=fpr_bin_edges)
        bin_widths = fpr_bin_edges[1:] - fpr_bin_edges[:-1] 
        auc = np.sum(tpr*bin_widths)
        t, _, _ = binned_statistic(all_fpr, all_thresh, statistic='mean', bins=fpr_bin_edges)

        return (
            np.array([*t, t[-1]]),
            np.asarray(fpr_bin_edges),
            np.array([*tpr, tpr[-1]]),
            np.array([*tpr_err, tpr_err[-1]]),
        )

    def plot_f1_curve(self, ax, ref_label, formatter=Formatter()):
        """Plot calibration curve."""
        ax.set_ylabel(r"F$_1$")

        thresholds = np.linspace(0, 1, 1000)
        f1_mu = []
        f1_sig = []

        for t in thresholds:
            y_pred = y_score > t
            intersect = (y_pred & y_true).astype(int)
            intersect_other = (~y_pred & ~y_true).astype(int)
            f1s = []
            for f in np.unique(folds):
                f_idx = folds == f
                if sum(y_pred[f_idx].astype(int)) == 0:
                    precision = 1.0
                else:
                    precision = sum(intersect[f_idx]) / sum(y_pred[f_idx].astype(int))
                recall = sum(intersect[f_idx]) / sum(y_true[f_idx].astype(int))
                Ia_f1 = 2 * precision * recall / (precision + recall)

                if sum((~y_pred[f_idx]).astype(int)) == 0:
                    prec2 = 1.0
                else:
                    prec2 = sum(intersect_other[f_idx]) / sum((~y_pred[f_idx]).astype(int))
                recall2 = sum(intersect_other[f_idx]) / sum((~y_true[f_idx]).astype(int))
                other_f1 = 2 * prec2 * recall2 / (prec2 + recall2)
                f1s.append((Ia_f1 + other_f1)/2)
            f1_mu.append(np.nanmean(f1s))
            f1_sig.append(np.nanstd(f1s))

        f1_mu = np.asarray(f1_mu)
        f1_sig = np.asarray(f1_sig)
        ax.plot(thresholds, f1_mu, c=formatter.edge_color)
        ax.fill_between(thresholds, f1_mu-f1_sig, f1_mu+f1_sig, alpha=0.2, color=formatter.face_color)

        formatter.rotate_colors()
        
        # retrieve optimal F1 score
        best_idx = np.argmax(f1_mu)
        best_t = thresholds[best_idx]
        ax.axvline(x=best_t, linestyle='dotted', color=formatter.edge_color)
        x_left, x_right = ax.get_xlim()
        y_low, y_high = ax.get_ylim()

        ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xlabel("Confidence Threshold")
        
        formatter.reset_colors()
        formatter.reset_markers()

        return ax, best_t
    
    def plot_calibration_curve(self, fig, ax, formatter=Formatter()):
        """Plot calibration curve."""
        
        ax.set_ylabel("True Fraction")
        plt.locator_params(axis="x", nbins=3)

        legend_lines = []
        for ref_label in np.unique(self.true_classes):
            t, f, ferr = self._calc_calibration_curve(ref_label)

            (legend_line,) = ax.step(
                t, f, label=ref_label, c=colors[ref_class], where='post'
            )
            ax.fill_between(t, f-ferr, f+ferr, alpha=0.2, color=colors[ref_label], step='post')
            legend_lines.append(legend_line)

        ax.plot([0,1], [0,1], linestyle='dotted', color='k', linewidth=1)
        x_left, x_right = ax.get_xlim()
        y_low, y_high = ax.get_ylim()

        ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)))
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xlabel("Confidence")

        legend_keys = list(labels_to_classes.keys())
        fig.legend(legend_lines, legend_keys, loc="lower center", ncol=3)
        
        return fig, ax
    
    def _calc_calibration_curve(self, ref_label):
        """Return confidence vs. fraction of true positives."""
        df_copy = self._df.copy()
        thresh_bins = histedges_equalN(self._df[ref_label], 50)
        thresh_bins[-1] = 1.0
        f_means = []
        f_errs = []
        
        for i in range(len(thresh_bins)-1):
            _, df_sums = self._calculate_binary_threshold_vals(df_copy, ref_label, thresh_bins[i], thresh_bins[i+1])
            f_means.append(df_sums['purity'].mean())
            f_errs.append(df_sums['purity'].std())

        return thresh_bins, np.array(*f_means, 1.0), np.array(*f_errs, 0.0)
    
    def _calculate_binary_threshold_vals(self, df_copy, ref_label, t_min=-np.inf, t_max=np.inf):
        """Calculate binary y_true and y_pred derivatives
        that appear in lots of plotting functions.
        """
        df_copy['y_true'] = (df_copy['true_class'] == ref_label).astype(int)
        df_copy['y_score'] = df_copy[ref_label]
        df_copy['y_pred'] = (df_copy['y_score'] >= t_min) & (df_copy['y_score'] < t_max)
        df_copy['true_pos'] = df_copy['y_pred'] * df_copy['y_true']
        df_copy['false_pos'] = df_copy['y_pred'] * (1 - df_copy['y_true'])
        
        # aggregates across folds
        df_sums = df_copy[['y_true','y_pred','true_pos','false_pos']].groupby(df_copy['fold']).sum()
        df_sums['purity'] = df['true_pos'] / df_copy['y_pred']
        df_sums['fpr'] = df['false_pos'] / (1 - df_copy['y_true'])
        df_sums['tpr'] = df['true_pos'] / df_copy['y_true']
        return df_copy, df_sums
    
    def _calc_precision_recall(self, ref_label):
        """Calculate purity recall values at
        multiple threshholds for plot. Assumes y_true only
        contains 0s and 1s (target), and y_score are
        probabilities of being class 1.
        """
        df_copy = self._df.copy()
        
        threshholds = np.linspace(0, 1, num=1000)
        all_precs = []
        all_recalls = []
        all_thresh = []
        
        for t in threshholds:
            df_copy, df_sums = self._calculate_binary_threshold_vals(df_copy, ref_label, t)
            all_precs.extend(df_sums['purity'].to_numpy().tolist())
            all_recalls.extend(df_sums['completeness'].to_numpy().tolist())
            all_thresh.extend([t,] * len(df_sums))

        recall_bin_edges = np.unique(histedges_equalN(all_recalls, 30))
        p, _, _ = binned_statistic(all_recalls, all_precs, statistic='mean', bins=recall_bin_edges)
        perr, _, _ = binned_statistic(all_recalls, all_precs, statistic='std', bins=recall_bin_edges)
        t, _, _ = binned_statistic(all_recalls, all_thresh, statistic='mean', bins=recall_bin_edges)

        p = np.append(p, sum(df_copy['y_true'])/len(df_copy['y_true'])) # end of curve
        perr = np.append(perr, 0.0)
        return np.asarray(t), np.asarray(recall_bin_edges), np.asarray(p), np.asarray(perr)


    def rebin_prec_recall(self, t, r, rerr, p, perr):
        """Turn completeness to the independent variable,
        and bin precision accordingly.
        """
        r_bin_centers = r
        r_bin_edges = (r[1:] + r[:-1]) / 2.0
        # add endpoints
        r_bin_edges = np.insert(r_bin_edges, 0, 2*r_bin_centers[0] - r_bin_edges[0])
        r_bin_edges = np.append(r_bin_edges, 2*r_bin_centers[-1] - r_bin_edges[-1])

        # find purity idxs that fall into each bin
        pmin_updated = []
        pmax_updated = []
        for i, b_center in enumerate(r_bin_centers):
            contained_idxs = (
                r - rerr < r_bin_edges[i+1]
            ) & (
                r + rerr >= r_bin_edges[i]
            )
            if len(p[contained_idxs]) == 0:
                pmin_updated.append(p[i])
                pmax_updated.append(p[i])
            else:
                pmin_updated.append(
                    p[contained_idxs][0] - perr[contained_idxs][0]
                )
                pmax_updated.append(
                    p[contained_idxs][-1] + perr[contained_idxs][-1]
                )
        pmin_updated.append(pmin_updated[-1])
        pmax_updated.append(pmax_updated[-1])
        p_copy = np.append(p, p[-1])
        return np.asarray(r_bin_edges), np.asarray(p_copy), np.asarray(pmin_updated), np.asarray(pmax_updated)
    
    def plot_precision_recall(self, fig, ax, plot_fleet=True):
        """Show how adjusting binary threshholds impact
        purity and completeness values."""
        ax1.set_ylabel("Purity")
        plt.locator_params(axis="x", nbins=3)

        legend_lines = []

        for ref_label in np.unique(self._df['true_class']):
            y_true = np.where(self._df['true_class'] == ref_label, 1, 0)
            y_score = probs[:, ref_label]

            prevalence = sum(y_true) / len(y_true)
            t, r, p, perr = calc_precision_recall(y_true, y_score, folds)

            idx_50 = np.argmin((t - 0.5) ** 2)
            (legend_line,) = ax1.step(
                r, p, label=ref_label, c=colors[ref_class], where='post'
            )
            ax.fill_between(r, p-perr, p+perr, alpha=0.2, color=colors[ref_class], step='post')
            legend_lines.append(legend_line)
            ax.scatter(
                (r[idx_50]+r[idx_50+1])/2, p[idx_50],
                color=colors[ref_class], s=100, marker="d", zorder=1000
            )
            # print AUPR value
            aupr = np.sum((r[1:] - r[:-1]) * p[:-1])
            aupr_min = np.sum((r[1:] - r[:-1]) * (p-perr)[:-1])
            aupr_max = np.sum((r[1:] - r[:-1]) * (p+perr)[:-1])
            print(ref_label, aupr, aupr_min, aupr_max)
        """
        if plot_fleet:
            fleet_fn = os.path.abspath(
                os.path.join(os.path.dirname(__file__), '../../..', 'data', 'SLSN_late-time.txt')
            )
            prevalence = 187 / 4780
            fleet_df = pd.read_csv(fleet_fn, sep='\s+')
            p_fleet = fleet_df['Purity'].to_numpy()
            perr_fleet = fleet_df['PurityStd'].to_numpy()
            r_fleet = fleet_df['Completeness'].to_numpy()
            rerr_fleet = fleet_df['CompletenessStd'].to_numpy()
            t_fleet = fleet_df['P(SLSN-I)'].to_numpy()

            rbin_fleet, pbin_fleet, pmin_fleet, pmax_fleet = rebin_prec_recall(
                t_fleet, r_fleet, rerr_fleet, p_fleet, perr_fleet
            )
            idx_50 = np.argmin((t_fleet - 0.5) ** 2)

            pscaled_fleet = (pbin_fleet - prevalence) / (1 - prevalence)
            pmin_scaled = (pmin_fleet - prevalence) / (1 - prevalence)
            pmax_scaled = (pmax_fleet - prevalence) / (1 - prevalence)

            (legend_line,) = ax1.step(
                rbin_fleet, pbin_fleet, label='FLEET (SLSN-I)', c=colors[5], where='post'
            )
            ax1.fill_between(
                rbin_fleet, pmin_fleet, pmax_fleet, alpha=0.2, color=colors[5], step='post'
            )
            legend_lines.append(legend_line)
            ax1.scatter(
                r_fleet[idx_50], p_fleet[idx_50],
                color=colors[5], s=100, marker="d", zorder=1000
            )

            aupr = np.sum((rbin_fleet[:-1] - rbin_fleet[1:]) * pbin_fleet[:-1])
            aupr_min = np.sum((rbin_fleet[:-1] - rbin_fleet[1:]) * pmin_fleet[:-1])
            aupr_max = np.sum((rbin_fleet[:-1] - rbin_fleet[1:]) * pmax_fleet[:-1])
            print("FLEET", aupr, aupr_min, aupr_max)
            
        """
        x_left, x_right = ax.get_xlim()
        y_low, y_high = ax.get_ylim()

        ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)))

        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xlabel("Completeness")

        #legend_lines.append(legend_line)
        legend_keys = [*list(labels_to_classes.keys()), "FLEET (SLSN-I)"]
        fig.legend(legend_lines, legend_keys, loc="lower center", ncol=3)
        return fig, ax