import html

import torch

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML, display

from transformer_lens import HookedTransformer

def batch_color_interpolate(scores, max_color, zero_color, scores_min=None, scores_max=None):
    if scores_min is None: scores_min = scores.min()
    if scores_max is None: scores_max = scores.max()
    scores_normalized = (scores - scores_min) / (scores_max - scores_min)
    
    max_color_vec = np.array([int(max_color[1:3], 16), int(max_color[3:5], 16), int(max_color[5:7], 16)])
    zero_color_vec = np.array([int(zero_color[1:3], 16), int(zero_color[3:5], 16), int(zero_color[5:7], 16)])

    color_vecs = np.einsum('i, j -> ij', scores_normalized, max_color_vec) + np.einsum('i, j -> ij', 1-scores_normalized, zero_color_vec)
    color_strs = [f"#{int(x[0]):02x}{int(x[1]):02x}{int(x[2]):02x}" for x in color_vecs]
    return color_strs

def plot_pulledback_feature(U, feature, size=None):
    if size is None: size = (5, 3)

    with torch.no_grad():
        p = U.T @ feature

    p = p.cpu().numpy()

    colors = batch_color_interpolate(p, '#7f7fff', '#ff7f7f')

    fig, ax = plt.subplots()
    ax.plot(p, alpha=0.5)
    ax.scatter(range(len(p)), p, color=colors)
    fig.set_size_inches(size[0], size[1])
    plt.xlabel("Feature index")
    plt.ylabel("Connection strength")
    plt.show()

    return p

@torch.no_grad()
def get_pullback_features(U, feature, size=None, k=7):
    pulledback_feature = torch.from_numpy(plot_pulledback_feature(U, feature, size=size))

    most_pos = torch.topk(pulledback_feature, k=k)
    most_neg = torch.topk(-pulledback_feature, k=k)

    return zip(
        most_pos.values.numpy(),
        most_pos.indices.numpy(),
        -most_neg.values.numpy(),
        most_neg.indices.numpy(),
    )

def display_pullback_features(U, feature, k=7):
    logits = list(get_pullback_features(U, feature, k=k))

    table_html = """
<style>
    span.token {
        font-family: monospace;
        
        border-style: solid;
        border-width: 1px;
        border-color: #dddddd;
    }
</style>
<table>
    <thead>
        <tr>
            <th colspan=2 style='text-align:center'>Most-negative features</th>
            <th colspan=2 style='text-align:center'>Most-positive features</th>
        </tr>
    </thead>
    <tbody>
"""

    top_scores = np.array([x[0] for x in logits])
    bot_scores = np.array([x[2] for x in logits])
    scores_max = np.max(top_scores)
    scores_min = np.min(bot_scores)
    
    top_color_strs = batch_color_interpolate(top_scores, '#7f7fff', '#ffffff', scores_min=scores_min, scores_max=scores_max)
    bot_color_strs = batch_color_interpolate(-bot_scores, '#ff7f7f', '#ffffff', scores_min=scores_min, scores_max=scores_max)

    for i, (top_val, top_idx, bot_val, bot_idx) in enumerate(logits):
        row_html = f"""<tr>
    <td style='text-align:left'><span class='token' style='background-color: {bot_color_strs[i]}'>{bot_idx}</span></td>
    <td style='text-align:right'>{bot_val:.3f}</td>
    <td style='text-align:left'><span class='token' style='background-color: {top_color_strs[i]}'>{top_idx}</span></td>
    <td style='text-align:right'>+{top_val:.3f}</td>
</tr>"""
        table_html = table_html + row_html
    table_html = table_html + "</tbody></table>"

    display(HTML(table_html))

@torch.no_grad()
def get_feature_deembeddings(model: HookedTransformer, feature, k=7):
    pulledback_feature = model.W_E.float() @ feature

    most_pos = torch.topk(pulledback_feature, k=k)
    most_neg = torch.topk(-pulledback_feature, k=k)

    top_vals = most_pos.values.cpu().numpy()
    top_idxs = most_pos.indices.cpu().numpy()
    top_tokens = model.to_str_tokens(top_idxs)
    
    bot_vals = -most_neg.values.cpu().numpy()
    bot_idxs = most_neg.indices.cpu().numpy()
    bot_tokens = model.to_str_tokens(bot_idxs)

    return zip(top_vals, top_tokens, bot_vals, bot_tokens)


def display_feature_deembeddings(model, feature, k=7):
    deembeddings = get_feature_deembeddings(model, feature, k=k)
    deembeddings = list(deembeddings)

    table_html = """
<style>
    span.token {
        font-family: monospace;
        
        border-style: solid;
        border-width: 1px;
        border-color: #dddddd;
    }
</style>"""f"""
<table>
    <thead>
        <tr>
            <th colspan=2 style='text-align:center'>Most-negative de-embedding tokens</th>
            <th colspan=2 style='text-align:center'>Most-positive de-embedding tokens</th>
        </tr>
    </thead>
    <tbody>
"""

    top_scores = np.array([x[0] for x in deembeddings])
    bot_scores = np.array([x[2] for x in deembeddings])
    scores_max = np.max(top_scores)
    scores_min = np.min(bot_scores)
    
    top_color_strs = batch_color_interpolate(top_scores, '#7f7fff', '#ffffff', scores_min=scores_min, scores_max=scores_max)
    bot_color_strs = batch_color_interpolate(-bot_scores, '#ff7f7f', '#ffffff', scores_min=scores_min, scores_max=scores_max)

    for i, (top_val, top_token, bot_val, bot_token) in enumerate(deembeddings):
        row_html =\
f"""<tr>
    <td style='text-align:left'><span class='token' style='background-color: {bot_color_strs[i]}'>{html.escape(bot_token).replace(" ", "&nbsp;")}</span></td>
    <td style='text-align:right'>{bot_val:.3f}</td>
    <td style='text-align:left'><span class='token' style='background-color: {top_color_strs[i]}'>{html.escape(top_token).replace(" ", "&nbsp;")}</span></td>
    <td style='text-align:right'>+{top_val:.3f}</td>
</tr>"""
        table_html = table_html + row_html
    table_html = table_html + "</tbody></table>"
    display(HTML(table_html))