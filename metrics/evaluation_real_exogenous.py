import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
import torch
from captum.attr import Saliency


def create_prediction_plot(start, num, fh, target, mu, logvar, scaler, name):
    upper99 = scaler.inverse_transform(mu + 2.58 * logvar.exp().sqrt())
    lower99 = scaler.inverse_transform(mu - 2.58 * logvar.exp().sqrt())
    upper80 = scaler.inverse_transform(mu + 1.28 * logvar.exp().sqrt())
    lower80 = scaler.inverse_transform(mu - 1.28 * logvar.exp().sqrt())
    upper60 = scaler.inverse_transform(mu + 0.84 * logvar.exp().sqrt())
    lower60 = scaler.inverse_transform(mu - 0.84 * logvar.exp().sqrt())
    lower60[lower60 < 0] = 0
    lower80[lower80 < 0] = 0
    lower99[lower99 < 0] = 0
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

    x_axis = np.arange(start=0, stop=int(fh) * (num + 1), step=1)
    fig, ax = plt.subplots(figsize=(6, 3), dpi=600)
    ax.plot(x_axis, t, label='True')
    ax.fill_between(x_axis, u99, l99, label="99% PI", alpha=0.2)
    ax.fill_between(x_axis, u80, l80, label="80% PI", alpha=0.3)
    ax.fill_between(x_axis, u60, l60, label="60% PI", alpha=0.4)
    ax.set_xlabel("Test Index")
    ax.set_ylabel("Load")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=[0.5, 1.14], frameon=False)
    tikzplotlib.save(name)
    plt.close(fig)
    return fig


def create_stacked_explainations(model, indexes, fh, history, feature1, feature2, feature3=None):
    explainer = Saliency(model)
    mu_dict = dict()
    logvar_dict = dict()
    for h in range(int(fh)):
        if feature3 is not None:
            mu_dict[h] = explainer.attribute(
                inputs=(history[indexes], feature1[indexes], feature2[indexes], feature3[indexes]), target=(h, 0))
            logvar_dict[h] = explainer.attribute(
                inputs=(history[indexes], feature1[indexes], feature2[indexes], feature3[indexes]), target=(h, 1))
        else:
            mu_dict[h] = explainer.attribute(inputs=(history[indexes], feature1[indexes], feature2[indexes]),
                                             target=(h, 0))
            logvar_dict[h] = explainer.attribute(inputs=(history[indexes], feature1[indexes], feature2[indexes]),
                                                 target=(h, 1))
    stacked_history_mu = torch.stack([mu_dict[h][0] for h in range(int(fh))])
    stacked_history_logvar = torch.stack([logvar_dict[h][0] for h in range(int(fh))])
    stacked_feature1_mu = torch.stack([mu_dict[h][1] for h in range(int(fh))])
    stacked_feature1_logvar = torch.stack([logvar_dict[h][1] for h in range(int(fh))])
    stacked_feature2_mu = torch.stack([mu_dict[h][2] for h in range(int(fh))])
    stacked_feature2_logvar = torch.stack([logvar_dict[h][2] for h in range(int(fh))])
    if feature3 is not None:
        stacked_feature3_mu = torch.stack([mu_dict[h][3] for h in range(int(fh))])
        stacked_feature3_logvar = torch.stack([logvar_dict[h][3] for h in range(int(fh))])
        final_mu = dict({"history": stacked_history_mu, "feature1": stacked_feature1_mu,
                         "feature2": stacked_feature2_mu, "feature3": stacked_feature3_mu})
        final_logvar = dict({"history": stacked_history_logvar, "feature1": stacked_feature1_logvar,
                             "feature2": stacked_feature2_logvar, "feature3": stacked_feature3_logvar})
    else:
        final_mu = dict({"history": stacked_history_mu, "feature1": stacked_feature1_mu,
                         "feature2": stacked_feature2_mu})
        final_logvar = dict({"history": stacked_history_logvar, "feature1": stacked_feature1_logvar,
                             "feature2": stacked_feature2_logvar})
    save_dict = dict({"mu": final_mu, "logvar": final_logvar})
    return save_dict


def plot_mean_specific_time_history(stacked_explainer_dict, indexes, step, start, scaler, fh, hl, history, day_offset):
    x_axis = np.arange(0, int(hl), 15)
    x_labels = [f"t-{int(hl) - n}" for n in x_axis]
    y_axis = np.arange(0, int(fh), 4)
    y_label = y_axis + 1
    scaled_history = scaler.inverse_transform(history)
    scaled_history = scaled_history[indexes]
    length_of_test = len(scaled_history)
    mean_his = np.mean(
        np.stack([scaled_history[day_offset + n] for n in np.arange(start, length_of_test - day_offset, step)]), axis=0)
    stacked_mu = torch.mean(
        torch.abs(torch.stack([stacked_explainer_dict['mu']['history'][:, day_offset + s, :] for s in
                               np.arange(start, length_of_test - day_offset, step)])),
        dim=0)
    stacked_logvar = torch.mean(torch.abs(
        torch.stack([stacked_explainer_dict['logvar']['history'][:, day_offset + s, :] for s in
                     np.arange(start, length_of_test - day_offset, step)])),
        dim=0)
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
    plt.close(fig)
    return fig


def plot_mean_specific_time(stacked_explainer_dict, indexes, step, start, scaler, fh, feature, feature_key, day_offset):
    x_axis = np.arange(0, int(fh), 4)
    x_labels = [f"t+{int(n) + 1}" for n in x_axis]
    y_axis = np.arange(0, int(fh), 4)
    y_label = y_axis + 1
    scaled_feature = scaler.inverse_transform(feature)
    scaled_feature = scaled_feature[indexes]
    length_of_test = len(scaled_feature)
    mean_feature = np.mean(
        np.stack([scaled_feature[day_offset + n] for n in np.arange(start, length_of_test - day_offset, step)]), axis=0)
    stacked_mu = torch.mean(
        torch.abs(torch.stack([stacked_explainer_dict['mu'][feature_key][:, day_offset + s, :] for s in
                               np.arange(start, length_of_test - day_offset, step)])),
        dim=0)
    stacked_logvar = torch.mean(torch.abs(
        torch.stack([stacked_explainer_dict['logvar'][feature_key][:, day_offset + s, :] for s in
                     np.arange(start, length_of_test - day_offset, step)])),
        dim=0)
    fig, ax = plt.subplots(3, 1, figsize=(6, 4.5), dpi=600)
    ax[0].plot(mean_feature)
    ax[0].set_ylabel("Value")
    ax[0].set_title("Mean Feature Input")
    ax[0].set_xticks(x_axis, labels=x_labels)
    im1 = ax[1].imshow(stacked_mu, cmap='Blues', aspect='auto')
    ax[1].set_yticks(y_axis, labels=y_label)
    ax[1].set_title("Average Attribution for Mean")
    ax[1].set_xticks(x_axis, labels=x_labels)
    ax[1].set_xlim(ax[0].get_xlim())
    im2 = ax[2].imshow(stacked_logvar, cmap='Blues', aspect='auto')
    ax[2].set_yticks(y_axis, labels=y_label)
    ax[2].set_xlabel("Time (Forecast Horizon)")
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
    plt.close(fig)
    return fig


def calculate_mean_stacked_explainations(model, indexes, fh, history, feature1, feature2, feature3=None):
    explainer = Saliency(model)
    mu_dict = dict()
    logvar_dict = dict()
    for h in range(int(fh)):
        if feature3 is not None:
            mu_dict[h] = explainer.attribute(
                inputs=(history[indexes], feature1[indexes], feature2[indexes], feature3[indexes]), target=(h, 0))
            logvar_dict[h] = explainer.attribute(
                inputs=(history[indexes], feature1[indexes], feature2[indexes], feature3[indexes]), target=(h, 1))
        else:
            mu_dict[h] = explainer.attribute(inputs=(history[indexes], feature1[indexes], feature2[indexes]),
                                             target=(h, 0))
            logvar_dict[h] = explainer.attribute(inputs=(history[indexes], feature1[indexes], feature2[indexes]),
                                                 target=(h, 1))

    stacked_history_mu = dict()
    stacked_history_logvar = dict()
    stacked_feature1_mu = dict()
    stacked_feature1_logvar = dict()
    stacked_feature2_mu = dict()
    stacked_feature2_logvar = dict()
    stacked_feature3_mu = dict()
    stacked_feature3_logvar = dict()
    for h in range(int(fh)):
        stacked_history_mu[h] = torch.mean(torch.abs(torch.stack([mu_dict[h][0].T[i] for i in range(int(fh))])), dim=0)
        stacked_history_logvar[h] = torch.mean(torch.abs(torch.stack([logvar_dict[h][0].T[i] for i in range(int(fh))])),
                                               dim=0)
        stacked_feature1_mu[h] = torch.mean(torch.abs(torch.stack([mu_dict[h][1].T[i] for i in range(int(fh))])), dim=0)
        stacked_feature1_logvar[h] = torch.mean(
            torch.abs(torch.stack([logvar_dict[h][1].T[i] for i in range(int(fh))])), dim=0)
        stacked_feature2_mu[h] = torch.mean(torch.abs(torch.stack([mu_dict[h][2].T[i] for i in range(int(fh))])), dim=0)
        stacked_feature2_logvar[h] = torch.mean(
            torch.abs(torch.stack([logvar_dict[h][2].T[i] for i in range(int(fh))])), dim=0)
        if feature3 is not None:
            stacked_feature3_mu[h] = torch.mean(torch.abs(torch.stack([mu_dict[h][3].T[i] for i in range(int(fh))])),
                                                dim=0)
            stacked_feature3_logvar[h] = torch.mean(
                torch.abs(torch.stack([logvar_dict[h][3].T[i] for i in range(int(fh))])), dim=0)

    history_mu = torch.stack([stacked_history_mu[h] for h in range(int(fh))])
    history_logvar = torch.stack([stacked_history_logvar[h] for h in range(int(fh))])
    feature1_mu = torch.stack([stacked_feature1_mu[h] for h in range(int(fh))])
    feature1_logvar = torch.stack([stacked_feature1_logvar[h] for h in range(int(fh))])
    feature2_mu = torch.stack([stacked_feature2_mu[h] for h in range(int(fh))])
    feature2_logvar = torch.stack([stacked_feature2_logvar[h] for h in range(int(fh))])

    if feature3 is not None:
        feature3_mu = torch.stack([stacked_feature3_mu[h] for h in range(int(fh))])
        feature3_logvar = torch.stack([stacked_feature3_logvar[h] for h in range(int(fh))])
        final_mu = dict({"history": history_mu, "feature1": feature1_mu,
                         "feature2": feature2_mu, "feature3": feature3_mu})
        final_logvar = dict({"history": history_logvar, "feature1": feature1_logvar,
                             "feature2": feature2_logvar, "feature3": feature3_logvar})
    else:
        final_mu = dict({"history": history_mu, "feature1": feature1_mu,
                         "feature2": feature2_mu})
        final_logvar = dict({"history": history_logvar, "feature1": feature1_logvar,
                             "feature2": feature2_logvar})
    save_dict = dict({"mu": final_mu, "logvar": final_logvar})
    return save_dict


def plot_average_attribution_for_multiple_samples_solar(stacked, mean_or_logvar, fh, day_offset, end_point, vmin, vmax):
    fig, ax = plt.subplots(4, 1, figsize=(6, 8), dpi=600)
    y_axis = np.arange(0, int(fh), 4)
    y_label = y_axis + 1
    x_axis = np.arange(0, int(end_point) - int(day_offset), 5)
    x_labels = (x_axis + day_offset) % int(24)
    im1 = ax[0].imshow(stacked[mean_or_logvar]["history"][:, day_offset:end_point], cmap='Blues', aspect="auto",
                       vmin=vmin, vmax=vmax)
    fig.colorbar(im1, ax=ax[0])
    ax[0].set_yticks(y_axis, labels=y_label)
    ax[0].set_title("SMAA History")
    ax[0].set_xticks(x_axis, labels=x_labels)
    im2 = ax[1].imshow(stacked[mean_or_logvar]["feature1"][:, day_offset:end_point], cmap='Blues', aspect='auto',
                       vmin=vmin, vmax=vmax)
    fig.colorbar(im2, ax=ax[1])
    ax[1].set_yticks(y_axis, labels=y_label)
    ax[1].set_title("SMAA TCLW")
    ax[1].set_xticks(x_axis, labels=x_labels)
    ax[1].set_xlim(ax[0].get_xlim())
    im3 = ax[2].imshow(stacked[mean_or_logvar]["feature2"][:, day_offset:end_point], cmap='Blues', aspect='auto',
                       vmin=vmin, vmax=vmax)
    fig.colorbar(im3, ax=ax[2])
    ax[2].set_yticks(y_axis, labels=y_label)
    ax[2].set_title("SMAA TCC")
    ax[2].set_xticks(x_axis, labels=x_labels)
    ax[2].set_xlim(ax[0].get_xlim())
    im4 = ax[3].imshow(stacked[mean_or_logvar]["feature3"][:, day_offset:end_point], cmap='Blues', aspect='auto',
                       vmin=vmin, vmax=vmax)
    fig.colorbar(im4, ax=ax[3])
    ax[3].set_yticks(y_axis, labels=y_label)
    ax[3].set_xlabel("Hour in Day")
    ax[3].set_xticks(x_axis, labels=x_labels)
    ax[3].set_xlim(ax[0].get_xlim())
    ax[3].set_title("SMAA SSRD")
    fig.text(0.0, 0.5, 'Forecast Horizon (h)', va='center', rotation='vertical')
    fig.tight_layout()
    plt.close(fig)
    return fig
