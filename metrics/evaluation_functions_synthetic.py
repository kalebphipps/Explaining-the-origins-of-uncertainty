import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from captum.attr import IntegratedGradients, Saliency, FeatureAblation, FeaturePermutation, ShapleyValueSampling


def create_prediction_plot(start, num, fh, target, mu, logvar, scaler):
    upper99 = scaler.inverse_transform(mu + 2.58 * logvar.exp().sqrt())
    lower99 = scaler.inverse_transform(mu - 2.58 * logvar.exp().sqrt())
    upper80 = scaler.inverse_transform(mu + 1.28 * logvar.exp().sqrt())
    lower80 = scaler.inverse_transform(mu - 1.28 * logvar.exp().sqrt())
    upper60 = scaler.inverse_transform(mu + 0.84 * logvar.exp().sqrt())
    lower60 = scaler.inverse_transform(mu - 0.84 * logvar.exp().sqrt())
    target = scaler.inverse_transform(target)

    u99 = upper99[start, :]
    l99 = lower99[start, :]
    u80 = upper80[start, :]
    l80 = lower80[start, :]
    u60 = upper60[start, :]
    l60 = lower60[start, :]
    t = target[start, :]

    for i in range(num):
        u99 = np.concatenate([u99, upper99[start + (i + 1) * fh, :]])
        l99 = np.concatenate([l99, lower99[start + (i + 1) * fh, :]])
        u80 = np.concatenate([u80, upper80[start + (i + 1) * fh, :]])
        l80 = np.concatenate([l80, lower80[start + (i + 1) * fh, :]])
        u60 = np.concatenate([u60, upper60[start + (i + 1) * fh, :]])
        l60 = np.concatenate([l60, lower60[start + (i + 1) * fh, :]])
        t = np.concatenate([t, target[start + (i + 1) * fh, :]])

    x_axis = np.arange(start=0, stop=fh * (num + 1), step=1)
    fig, ax = plt.subplots(figsize=(3, 1), dpi=600)
    ax.plot(x_axis, t, label='True')
    ax.fill_between(x_axis, u99, l99, label="99% PI", alpha=0.2)
    ax.fill_between(x_axis, u80, l80, label="80% PI", alpha=0.3)
    ax.fill_between(x_axis, u60, l60, label="60% PI", alpha=0.4)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=[0.5, 1.62], frameon=False)
    plt.close(fig)
    return fig


def create_prediction_plot_quantile(start, num, fh, target, quantile_dict, scaler):
    upper99 = scaler.inverse_transform(quantile_dict[0.99])
    lower99 = scaler.inverse_transform(quantile_dict[0.01])
    target = scaler.inverse_transform(target)

    u99 = upper99[start, :]
    l99 = lower99[start, :]
    t = target[start, :]

    for i in range(num):
        u99 = np.concatenate([u99, upper99[start + (i + 1) * fh, :]])
        l99 = np.concatenate([l99, lower99[start + (i + 1) * fh, :]])
        t = np.concatenate([t, target[start + (i + 1) * fh, :]])

    x_axis = np.arange(start=0, stop=fh * (num + 1), step=1)
    fig, ax = plt.subplots(figsize=(3, 1), dpi=600)
    ax.plot(x_axis, t, label='True')
    ax.fill_between(x_axis, u99, l99, label="99% PI", alpha=0.2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    fig.legend(ncol=2, loc="upper center", bbox_to_anchor=[0.5, 1.2], frameon=False)
    plt.close(fig)
    return fig


def gaussian_explaination_plot_sample(model, sample, fh, history, hl, scaler):
    explainer_dict = dict({"IG": IntegratedGradients(model, multiply_by_inputs=False),
                           "Saliency": Saliency(model),
                           "FA": FeatureAblation(model),
                           "FP": FeaturePermutation(model),
                           "SVS": ShapleyValueSampling(model)})

    x_axis = np.arange(0, hl, 5)
    x_labels = [f"t-{hl - n}" for n in x_axis]
    y_axis = np.arange(0, fh, 4)
    y_label = y_axis + 1
    scaled_history = scaler.inverse_transform(history)

    save_dict = dict()
    for key, item in explainer_dict.items():
        mu_dict = dict()
        logvar_dict = dict()
        explainer = explainer_dict[key]
        for i in range(fh):
            mu_dict[i] = explainer.attribute(inputs=history[sample:sample + 2], target=(i, 0))
            logvar_dict[i] = explainer.attribute(inputs=history[sample:sample + 2], target=(i, 1))

        mu_stack = torch.stack([mu_dict[i][0] for i in range(fh)])
        logvar_stack = torch.stack([logvar_dict[i][0] for i in range(fh)])
        fig, ax = plt.subplots(3, 1, figsize=(6, 4.5), dpi=600)
        ax[0].plot(scaled_history[sample, :])
        ax[0].set_ylabel("Value")
        ax[0].set_title("History Input")
        ax[0].set_xticks(x_axis, labels=x_labels)
        im1 = ax[1].imshow(mu_stack, cmap='Blues', aspect='auto')
        ax[1].set_yticks(y_axis, labels=y_label)
        ax[1].set_title("Attribution for Mean")
        ax[1].set_xticks(x_axis, labels=x_labels)
        ax[1].set_xlim(ax[0].get_xlim())
        im2 = ax[2].imshow(logvar_stack, cmap='Blues', aspect='auto')
        ax[2].set_yticks(y_axis, labels=y_label)
        ax[2].set_xlabel("Time  (Historical Values)")
        ax[2].set_title("Attribution for Variance")
        ax[2].set_xticks(x_axis, labels=x_labels)
        ax[2].set_xlim(ax[0].get_xlim())
        fig.subplots_adjust(right=0.85)
        cbar1_ax = fig.add_axes([1.00, 0.44, 0.03, 0.16])
        cbar2_ax = fig.add_axes([1.00, 0.13, 0.03, 0.16])
        fig.colorbar(im1, cax=cbar1_ax)
        fig.colorbar(im2, cax=cbar2_ax)
        fig.text(0.02, 0.35, 'Forecast Horizon (h)', va='center', rotation='vertical')
        fig.tight_layout()
        save_dict[key] = fig
        plt.close(fig)
    return save_dict


def quantile_explain_plot_sample(model_dict, sample, fh, history, hl, scaler):
    explainer_dict01 = dict({"IG": IntegratedGradients(model_dict[0.01], multiply_by_inputs=False),
                             "Saliency": Saliency(model_dict[0.01]),
                             "FA": FeatureAblation(model_dict[0.01]),
                             "FP": FeaturePermutation(model_dict[0.01]),
                             "SVS": ShapleyValueSampling(model_dict[0.01])})
    explainer_dict5 = dict({"IG": IntegratedGradients(model_dict[0.5], multiply_by_inputs=False),
                            "Saliency": Saliency(model_dict[0.5]),
                            "FA": FeatureAblation(model_dict[0.5]),
                            "FP": FeaturePermutation(model_dict[0.5]),
                            "SVS": ShapleyValueSampling(model_dict[0.5])})
    explainer_dict99 = dict({"IG": IntegratedGradients(model_dict[0.99], multiply_by_inputs=False),
                             "Saliency": Saliency(model_dict[0.99]),
                             "FA": FeatureAblation(model_dict[0.99]),
                             "FP": FeaturePermutation(model_dict[0.99]),
                             "SVS": ShapleyValueSampling(model_dict[0.99])})

    x_axis = np.arange(0, hl, 5)
    x_labels = [f"t-{hl - n}" for n in x_axis]
    y_axis = np.arange(0, fh, 4)
    y_label = y_axis + 1
    scaled_history = scaler.inverse_transform(history)

    save_dict = dict()
    for key, item in explainer_dict01.items():
        quant_dict01 = dict()
        quant_dict5 = dict()
        quant_dict99 = dict()
        explainer01 = explainer_dict01[key]
        explainer5 = explainer_dict5[key]
        explainer99 = explainer_dict99[key]
        for i in range(fh):
            quant_dict01[i] = explainer01.attribute(inputs=history[sample:sample + 2], target=i)
            quant_dict5[i] = explainer5.attribute(inputs=history[sample:sample + 2], target=i)
            quant_dict99[i] = explainer99.attribute(inputs=history[sample:sample + 2], target=i)
        quantile01_stack = torch.stack([quant_dict01[i][0] for i in range(fh)])
        quantile5_stack = torch.stack([quant_dict5[i][0] for i in range(fh)])
        quantile99_stack = torch.stack([quant_dict99[i][0] for i in range(fh)])
        fig, ax = plt.subplots(4, 1, figsize=(6, 6), dpi=600)
        ax[0].plot(scaled_history[sample, :])
        ax[0].set_ylabel("Value")
        ax[0].set_title("History Input")
        ax[0].set_xticks(x_axis, labels=x_labels)
        im1 = ax[1].imshow(quantile01_stack, cmap='Blues', aspect='auto')
        ax[1].set_yticks(y_axis, labels=y_label)
        ax[1].set_title("Attribution for 0.01 Quantile")
        ax[1].set_xticks(x_axis, labels=x_labels)
        ax[1].set_xlim(ax[0].get_xlim())
        im2 = ax[2].imshow(quantile5_stack, cmap='Blues', aspect='auto')
        ax[2].set_yticks(y_axis, labels=y_label)
        ax[2].set_title("Attribution for 0.5 Quantile")
        ax[2].set_xticks(x_axis, labels=x_labels)
        ax[2].set_xlim(ax[0].get_xlim())
        im3 = ax[3].imshow(quantile99_stack, cmap='Blues', aspect='auto')
        ax[3].set_yticks(y_axis, labels=y_label)
        ax[3].set_xlabel("Time  (Historical Values)")
        ax[3].set_title("Attribution for 0.99 Quantile")
        ax[3].set_xticks(x_axis, labels=x_labels)
        ax[3].set_xlim(ax[0].get_xlim())
        fig.subplots_adjust(right=0.85)
        cbar1_ax = fig.add_axes([1.00, 0.58, 0.03, 0.11])
        cbar2_ax = fig.add_axes([1.00, 0.34, 0.03, 0.11])
        cbar3_ax = fig.add_axes([1.00, 0.11, 0.03, 0.11])
        fig.colorbar(im1, cax=cbar1_ax)
        fig.colorbar(im2, cax=cbar2_ax)
        fig.colorbar(im3, cax=cbar3_ax)
        fig.text(0.02, 0.4, 'Forecast Horizon (h)', va='center', rotation='vertical')
        fig.tight_layout()
        save_dict[key] = fig
        plt.close(fig)
    return save_dict


def return_scaled_absolute_tensor(the_tensor):
    return (torch.abs(the_tensor) - torch.min(torch.abs(the_tensor))) / (
                torch.max(torch.abs(the_tensor)) - torch.min(torch.abs(the_tensor)))


def compare_mean_scaled_attributions(model, indexes_to_compare, fh, history):
    explainer_dict = dict({"IG": IntegratedGradients(model, multiply_by_inputs=False),
                           "Saliency": Saliency(model),
                           "FA": FeatureAblation(model),
                           "FP": FeaturePermutation(model),
                           "SVS": ShapleyValueSampling(model)})

    save_dict = dict()
    diff_mu_df = pd.DataFrame(columns=["IG", "Saliency", "FP", "FA", "SVS"],
                              index=["IG", "Saliency", "FP", "FA", "SVS"])
    diff_logvar_df = pd.DataFrame(columns=["IG", "Saliency", "FP", "FA", "SVS"],
                                  index=["IG", "Saliency", "FP", "FA", "SVS"])

    num_samples = len(indexes_to_compare)
    for key, item in explainer_dict.items():
        mu_dict = dict()
        logvar_dict = dict()
        explainer = explainer_dict[key]
        for i in range(fh):
            mu_dict[i] = explainer.attribute(inputs=history[indexes_to_compare], target=(i, 0))
            logvar_dict[i] = explainer.attribute(inputs=history[indexes_to_compare], target=(i, 1))
        scaled_mu = mu_dict.copy()
        scaled_logvar = logvar_dict.copy()
        for i in range(fh):
            for j in range(num_samples):
                scaled_mu[i][j] = return_scaled_absolute_tensor(mu_dict[i][j])
                scaled_logvar[i][j] = return_scaled_absolute_tensor(logvar_dict[i][j])
        save_dict[key] = dict({'mu': scaled_mu, "logvar": scaled_logvar})

    for c in diff_mu_df.columns:
        for i in diff_mu_df.index:
            diff_mu_df.loc[i, c] = torch.mean(
                torch.abs(torch.concatenate([save_dict[i]['mu'][n] - save_dict[c]['mu'][n] for n in range(fh)])))
            diff_logvar_df.loc[i, c] = torch.mean(torch.abs(
                torch.concatenate([save_dict[i]['logvar'][n] - save_dict[c]['logvar'][n] for n in range(fh)])))
    diff_mu_df = diff_mu_df.astype(np.float64)
    diff_logvar_df = diff_logvar_df.astype(np.float64)
    diff_combined_df = (diff_mu_df + diff_logvar_df) / 2

    fig1, ax1 = plt.subplots(figsize=(3, 2), dpi=600)
    sns.heatmap(diff_mu_df, cmap='Blues',
                annot=True,
                ax=ax1)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(3, 2), dpi=600)
    sns.heatmap(diff_logvar_df, cmap='Blues',
                annot=True,
                ax=ax2)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(3, 2), dpi=600)
    sns.heatmap(diff_combined_df, cmap='Blues',
                annot=True,
                ax=ax3)
    plt.close(fig3)

    fig_dict = dict({"mu": fig1, "logvar": fig2, "combined": fig3})
    difference_dict = dict({"mu": diff_mu_df, "logvar": diff_logvar_df, "combined": diff_combined_df})
    return save_dict, difference_dict, fig_dict


def plot_scaled_attribute_sample(scaled_explain, sample, fh, history, hl, scaler):
    x_axis = np.arange(0, hl, 5)
    x_labels = [f"t-{hl - n}" for n in x_axis]
    y_axis = np.arange(0, fh, 4)
    y_label = y_axis + 1
    scaled_history = scaler.inverse_transform(history)

    save_dict = dict()
    for key, item in scaled_explain.items():
        mu_stack = torch.stack([scaled_explain[key]['mu'][i][sample] for i in range(fh)])
        logvar_stack = torch.stack([scaled_explain[key]['logvar'][i][sample] for i in range(fh)])
        fig, ax = plt.subplots(3, 1, figsize=(6, 4.5), dpi=600)
        ax[0].plot(scaled_history[sample, :])
        ax[0].set_ylabel("Value")
        ax[0].set_title("History Input")
        ax[0].set_xticks(x_axis, labels=x_labels)
        im1 = ax[1].imshow(mu_stack, cmap='Blues', aspect='auto')
        ax[1].set_yticks(y_axis, labels=y_label)
        ax[1].set_title("Attribution for Mean")
        ax[1].set_xticks(x_axis, labels=x_labels)
        ax[1].set_xlim(ax[0].get_xlim())
        im2 = ax[2].imshow(logvar_stack, cmap='Blues', aspect='auto')
        ax[2].set_yticks(y_axis, labels=y_label)
        ax[2].set_xlabel("Time  (Historical Values)")
        ax[2].set_title("Attribution for Variance")
        ax[2].set_xticks(x_axis, labels=x_labels)
        ax[2].set_xlim(ax[0].get_xlim())
        fig.subplots_adjust(right=0.85)
        cbar1_ax = fig.add_axes([1.00, 0.44, 0.03, 0.16])
        cbar2_ax = fig.add_axes([1.00, 0.13, 0.03, 0.16])
        fig.colorbar(im1, cax=cbar1_ax)
        fig.colorbar(im2, cax=cbar2_ax)
        fig.text(0.02, 0.35, 'Forecast Horizon (h)', va='center', rotation='vertical')
        fig.tight_layout()
        save_dict[key] = fig
        plt.close(fig)
    return save_dict


def create_stacked_explainations(explainer_dict, indexes, fh, history):
    save_dict = dict()

    for key, item in explainer_dict.items():
        explainer = explainer_dict[key]
        mu_dict = dict()
        logvar_dict = dict()
        for h in range(fh):
            mu_dict[h] = explainer.attribute(inputs=(history[indexes]), target=(h, 0))
            logvar_dict[h] = explainer.attribute(inputs=(history[indexes]), target=(h, 1))
        stacked_mu = torch.stack([mu_dict[h] for h in range(fh)])
        stacked_logvar = torch.stack([logvar_dict[h] for h in range(fh)])
        save_dict[key] = dict({"mu": stacked_mu, "logvar": stacked_logvar})
    return save_dict


def plot_mean_specific_time(stacked_explainer_dict, indexes, history, step, start, scaler):
    save_dict = dict()
    x_axis = np.arange(0, 40, 5)
    x_labels = [f"t-{40 - n}" for n in x_axis]
    y_axis = np.arange(0, 20, 4)
    y_label = y_axis + 1
    scaled_history = scaler.inverse_transform(history)
    scaled_history = scaled_history[indexes]
    length_of_test = len(scaled_history)
    mean_his = np.mean(np.stack([scaled_history[n] for n in np.arange(start, length_of_test, step)]), axis=0)
    for key, item in stacked_explainer_dict.items():
        stacked_mu = torch.mean(torch.abs(
            torch.stack([stacked_explainer_dict[key]['mu'][:, s, :] for s in np.arange(start, length_of_test, step)])),
                                dim=0)
        stacked_logvar = torch.mean(torch.abs(torch.stack(
            [stacked_explainer_dict[key]['logvar'][:, s, :] for s in np.arange(start, length_of_test, step)])), dim=0)
        fig, ax = plt.subplots(3, 1, figsize=(6, 4.5), dpi=600)
        ax[0].plot(mean_his)
        ax[0].set_ylabel("Value")
        ax[0].set_title("Mean History Input")
        ax[0].set_xticks(x_axis, labels=x_labels)
        im1 = ax[1].imshow(stacked_mu, cmap='Blues', aspect='auto')
        ax[1].set_yticks(y_axis, labels=y_label)
        ax[1].set_title("Average Attribution for Mean")
        ax[1].set_xticks(x_axis, labels=x_labels)
        ax[1].set_xlim(ax[0].get_xlim())
        im2 = ax[2].imshow(stacked_logvar, cmap='Blues', aspect='auto')
        ax[2].set_yticks(y_axis, labels=y_label)
        ax[2].set_xlabel("Time (Historical Values)")
        ax[2].set_title("Average Attribution for Variance")
        ax[2].set_xticks(x_axis, labels=x_labels)
        ax[2].set_xlim(ax[0].get_xlim())
        fig.subplots_adjust(right=0.85)
        cbar1_ax = fig.add_axes([1.00, 0.44, 0.03, 0.16])
        cbar2_ax = fig.add_axes([1.00, 0.13, 0.03, 0.16])
        fig.colorbar(im1, cax=cbar1_ax)
        fig.colorbar(im2, cax=cbar2_ax)
        fig.text(0.02, 0.35, 'Forecast Horizon (h)', va='center', rotation='vertical')
        fig.tight_layout()
        save_dict[key] = fig
        plt.close(fig)
    return save_dict
