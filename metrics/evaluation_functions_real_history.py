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


def gaussian_explaination_plot_sample(model, sample, fh, history, hl, scaler):
    explainer = Saliency(model)

    x_axis = np.arange(0, int(hl), 15)
    x_labels = [f"t-{int(hl) - n}" for n in x_axis]
    y_axis = np.arange(0, int(fh), 4)
    y_label = y_axis + 1
    scaled_history = scaler.inverse_transform(history)
    mu_dict = dict()
    logvar_dict = dict()
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
    ax[2].set_xlabel("Time (Historical Values)")
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
    plt.close(fig)
    return fig


def create_stacked_explainations(model, indexes, fh, history):
    explainer = Saliency(model)
    mu_dict = dict()
    logvar_dict = dict()
    for h in range(int(fh)):
        mu_dict[h] = explainer.attribute(inputs=(history[indexes]), target=(h, 0))
        logvar_dict[h] = explainer.attribute(inputs=(history[indexes]), target=(h, 1))
    stacked_mu = torch.stack([mu_dict[h] for h in range(int(fh))])
    stacked_logvar = torch.stack([logvar_dict[h] for h in range(int(fh))])
    save_dict = dict({"mu": stacked_mu, "logvar": stacked_logvar})
    return save_dict


def plot_mean_specific_time(stacked_explainer_dict, indexes, history, step, start, scaler, fh, hl):
    x_axis = np.arange(0, int(hl), 15)
    x_labels = [f"t-{int(hl) - n}" for n in x_axis]
    y_axis = np.arange(0, int(fh), 4)
    y_label = y_axis + 1
    scaled_history = scaler.inverse_transform(history)
    scaled_history = scaled_history[indexes]
    length_of_test = len(scaled_history)
    mean_his = np.mean(np.stack([scaled_history[n] for n in np.arange(start, length_of_test, step)]), axis=0)
    stacked_mu = torch.mean(
        torch.abs(torch.stack([stacked_explainer_dict['mu'][:, s, :] for s in np.arange(start, length_of_test, step)])),
        dim=0)
    stacked_logvar = torch.mean(torch.abs(
        torch.stack([stacked_explainer_dict['logvar'][:, s, :] for s in np.arange(start, length_of_test, step)])),
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
