import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import patches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyArrowPatch
import matplotlib as mpl
from shapely.geometry import Polygon as SHPolygon
from shapely.geometry import LineString
from matplotlib.patches import Polygon
import os

def plot_polygons_with_labels(collected_samples, predicted_labels, class_colors=None, save_path=None,
                              n_classes_per_row=2,
                              show_labels=False
                              ):
    # Ground truth class label names
    class_label_names = ['L', 'Z', 'O', 'Y', 'I', 'T', 'E', 'C', 'F', 'H']

    # Generate unique ground truth labels and sort them
    unique_labels = sorted(collected_samples.keys())

    # Determine the maximum number of samples per class from the data
    samples_per_class = min(len(polygons) for polygons in collected_samples.values())

    # Assign random colors to each class if no colors are provided
    if class_colors is None:
        palette = sns.color_palette("colorblind", len(unique_labels))
        class_colors = {label: palette[i] for i, label in enumerate(unique_labels)}

    # Calculate grid dimensions dynamically
    n_cols = n_classes_per_row * samples_per_class  # Two classes per row, each with samples_per_class columns
    n_rows = 10 // n_classes_per_row  # Fixed number of rows since there are always 10 classes

    fig, axes = plt.subplots(n_rows + 1, n_cols, figsize=((n_cols) * 5, (n_rows) * 5))

    # Flatten the axes for easier indexing
    axes = axes.flatten()

    plot_idx = 0  # Index for placing each plot in the grid
    for idx, gt_label in enumerate(unique_labels):
        polygons = collected_samples[gt_label][
                   :samples_per_class]  # Limit to the determined number of samples per class
        # row_offset = (idx % n_classes_per_row) * samples_per_class  # Offset within the row for the second class
        row_offset = (idx % n_rows) * n_cols
        # base_row = (idx // n_classes_per_row) * n_cols  # Calculate base row index for each class pair
        base_row = (idx // n_rows)
        for col_idx, polygon in enumerate(polygons):
            # ax_idx = base_row + row_offset + col_idx
            ax_idx = row_offset + base_row
            ax = axes[ax_idx]
            polygon = polygon.numpy()  # Convert from tensor to numpy if necessary

            # Create a graph from the polygon
            G = nx.Graph()
            n_nodes = polygon.shape[0]
            G.add_nodes_from(range(n_nodes))
            G.add_edges_from([(i, i + 1) for i in range(n_nodes - 1)])
            G.add_edge(0, n_nodes - 1)  # Close the polygon

            # Get the color for the predicted label
            pred_label = predicted_labels[gt_label][col_idx]
            color = class_colors.get(pred_label, [0.5, 0.5, 0.5])  # Default to gray if label not in class_colors

            # Draw the graph edges
            nx.draw(G, pos={i: polygon[i] for i in range(n_nodes)}, ax=ax, with_labels=False,
                    edge_color="gray", node_size=0)

            # Fill the area inside the polygon
            ax.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=0.5, edgecolor="gray", linewidth=1.5)

            # Add the predicted label as text in the centroid
            if show_labels:
                centroid = polygon.mean(axis=0)
                ax.text(centroid[0], centroid[1], class_label_names[pred_label], color="black", fontsize=14,
                        ha="center",
                        va="center",
                        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.4"))

            # Set aspect ratio and remove axes
            ax.set_aspect('equal', 'box')
            ax.set_xticks([])
            ax.set_yticks([])

    # Remove unused axes
    for ax in axes[plot_idx:]:
        ax.axis('off')

    # Add a legend for the ground truth colors using patches
    handles = [patches.Patch(color=color, label=f'{class_label_names[label]}') for label, color in class_colors.items()]
    fig.legend(handles=handles, loc='lower center', frameon=True, fancybox=True, shadow=True,
               borderpad=1, title="Ground Truth Labels", title_fontsize=45, fontsize=36, ncols=n_cols)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.15)
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

def presentation_simplification_plot(original_vertices,gt_labels,ground_truth_footprint, show_labels=True,title=None,save_path=None):

    from shapely.geometry.polygon import Polygon as SHPolygon

    fig,ax = plt.subplots(figsize=(12, 10))
    # Set up seaborn style
    sns.set_theme(style="white")
    colors = sns.color_palette("pastel")

    if ax is None:
        # Create figure with seaborn aesthetics
        plt.figure(figsize=(12, 10))
        ax = plt.gca()

    if isinstance(original_vertices, SHPolygon):
        x_orig, y_orig = original_vertices.exterior.xy
    else:
        x_orig = [point.x for point in original_vertices]
        y_orig = [point.y for point in original_vertices]

    # Calculate the centroid (center point) of the original footprint to use as offset
    centroid_x = np.mean(x_orig)
    centroid_y = np.mean(y_orig)

    # Center all coordinates by subtracting the centroid
    x_orig_centered = x_orig - centroid_x
    y_orig_centered = y_orig - centroid_y

    # Plot vertices with gt labels
    remove_xs, remove_ys = x_orig_centered[:-1][gt_labels[:, 0] == 0], y_orig_centered[:-1][gt_labels[:, 0] == 0]
    keep_xs, keep_ys = x_orig_centered[:-1][gt_labels[:, 0] == 1], y_orig_centered[:-1][gt_labels[:, 0] == 1]
    move_xs, move_ys = x_orig_centered[:-1][gt_labels[:, 0] == 2], y_orig_centered[:-1][gt_labels[:, 0] == 2]
    ax.scatter(remove_xs, remove_ys, color=colors[4], s=500, edgecolor='black', zorder=10, label='Remove')
    ax.scatter(keep_xs, keep_ys, color=colors[2], s=500, edgecolor='black', zorder=10, label='Keep')
    ax.scatter(move_xs, move_ys, color=colors[3], s=500, edgecolor='black', zorder=10, label='Move')

    # Close the polygon if needed
    if (x_orig_centered[0] != x_orig_centered[-1]) or (y_orig_centered[0] != y_orig_centered[-1]):
        x_orig_centered = np.append(x_orig_centered, x_orig_centered[0])
        y_orig_centered = np.append(y_orig_centered, y_orig_centered[0])

    # Create and plot polygon with fill and no edge
    if show_labels:
        poly_orig = Polygon(np.column_stack([x_orig_centered, y_orig_centered]),
                            facecolor=colors[0],  # First pastel color
                            edgecolor='none',
                            alpha=0.5,
                            label='Original')
    else:
        poly_orig = Polygon(np.column_stack([x_orig_centered, y_orig_centered]),
                            facecolor=colors[0],  # First pastel color
                            edgecolor='none',
                            alpha=0.5)

    ax.add_patch(poly_orig)

    # Plot ground truth footprint with orange line
    if isinstance(ground_truth_footprint, SHPolygon):
        x_gt, y_gt = ground_truth_footprint.exterior.xy
    else:
        x_gt = [point.x for point in ground_truth_footprint]
        y_gt = [point.y for point in ground_truth_footprint]

    x_gt_centered = x_gt - centroid_x
    y_gt_centered = y_gt - centroid_y

    # Close the polygon if needed
    if (x_gt_centered[0] != x_gt_centered[-1]) or (y_gt_centered[0] != y_gt_centered[-1]):
        x_gt_centered = np.append(x_gt_centered, x_gt_centered[0])
        y_gt_centered = np.append(y_gt_centered, y_gt_centered[0])

    ax.plot(x_gt_centered, y_gt_centered, '--', color='black', linewidth=2.5, label='simplification')

    #plot displacement vectors
    legend_added = {'preMove': False, 'nextMove': False}  # Track legend entries

    for i in range(len(original_vertices)-1):
        if gt_labels[i, 0] == 2:
            prev_idx = (i - 1) % len(original_vertices)
            next_idx = (i + 1) % len(original_vertices)
            prev_vertex = [x_orig_centered[prev_idx], y_orig_centered[prev_idx]]
            next_vertex = [x_orig_centered[next_idx], y_orig_centered[next_idx]]
            vec_pre = [x_orig_centered[i] - prev_vertex[0], y_orig_centered[i] - prev_vertex[1]]
            vec_next = [next_vertex[0]-x_orig_centered[i], next_vertex[1]-y_orig_centered[i]]
            vec_pre_mod = LineString([prev_vertex, (x_orig_centered[i],y_orig_centered[i])]).length
            vec_next_mod = LineString([next_vertex, (x_orig_centered[i],y_orig_centered[i])]).length

            if vec_pre_mod > 0:
                pre_dir = (vec_pre[0]/vec_pre_mod, vec_pre[1]/vec_pre_mod)
            if vec_next_mod > 0:
                next_dir = (vec_next[0]/vec_next_mod, vec_next[1]/vec_next_mod)

            nextMove = (gt_labels[i,2] * next_dir[0], gt_labels[i,2] * next_dir[1])
            preMove = (gt_labels[i,1] * pre_dir[0], gt_labels[i,1] * pre_dir[1])

            # Plot preMove vector
            if not (preMove[0] == 0 and preMove[1] == 0):
                print(preMove)
                if not legend_added['preMove']:
                    ax.arrow(x_orig_centered[i], y_orig_centered[i],
                             preMove[0], preMove[1],
                             color='firebrick', width=0.07,
                             fc='firebrick', ec='firebrick', label='preMove',length_includes_head=True,
                             alpha=1)
                    legend_added['preMove'] = True
                else:
                    ax.arrow(x_orig_centered[i], y_orig_centered[i],
                             preMove[0], preMove[1],
                             color='firebrick', width=0.07,
                             fc='firebrick', ec='firebrick',length_includes_head=True,alpha=1)

            # Plot nextMove arrow
            if not (nextMove[0] == 0 and nextMove[1] == 0):
                print(nextMove)
                if not legend_added['nextMove']:
                    ax.arrow(x_orig_centered[i], y_orig_centered[i],
                             nextMove[0], nextMove[1],
                             color='fuchsia', width=0.07,
                             fc='fuchsia', ec='fuchsia', label='nextMove',length_includes_head=True,alpha=1)
                    legend_added['nextMove'] = True
                else:
                    ax.arrow(x_orig_centered[i], y_orig_centered[i],
                             nextMove[0], nextMove[1],
                             color='fuchsia', width=0.07,
                             fc='fuchsia', ec='fuchsia',length_includes_head=True,alpha=1)
    ax.set_aspect('equal')
    ax.set_axis_off()
    fig.tight_layout()
    fig.legend(loc='center', fontsize=20)

    if title:
        plt.title(title, fontsize=24)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_footprints_with_node_labels(original_vertices, gt_labels, pred_labels, save_path=None, title = None):
    """
    Plot original footprint, reconstructed footprint, and ground truth footprint.

    Args:
        original_vertices: Original vertex coordinates [N, 2] as numpy array
        pred_labels: Predicted labels (0:remove, 1:keep, 2:move) [N] as numpy array
        reconstructed_footprint: Reconstructed footprint vertices [M, 2] as numpy array
        ground_truth_footprint: Ground truth footprint vertices [K, 2] as numpy array
        save_path: Optional path to save the plot
    """
    # Define colors for labels
    label_colors = sns.color_palette('pastel', 3)
    sns.set_theme(style="white", palette="pastel")
    # Create figure
    fig, ax = plt.subplots(1,2, sharey=True, figsize=(12, 10))

    #close loop of original footprint for plotting
    #original_vertices.append[original_vertices[0]]

    # Create polygon for original footprint with fill
    x_orig = [point.x for point in original_vertices]
    y_orig = [point.y for point in original_vertices]

    # Calculate the centroid (center point) of the original footprint to use as offset
    centroid_x = np.mean(x_orig)
    centroid_y = np.mean(y_orig)

    # Center all coordinates by subtracting the centroid
    x_orig_centered = x_orig - centroid_x
    y_orig_centered = y_orig - centroid_y

    # Close the polygon if needed
    if (x_orig_centered[0] != x_orig_centered[-1]) or (y_orig_centered[0] != y_orig_centered[-1]):
        x_orig_centered = np.append(x_orig_centered, x_orig_centered[0])
        y_orig_centered = np.append(y_orig_centered, y_orig_centered[0])

    # Plot original footprint with colored vertices
    poly_orig = Polygon(np.column_stack([x_orig_centered, y_orig_centered]),
            facecolor='grey',  # First pastel color
            edgecolor='black',
            alpha=0.5)
    ax[0].add_patch(poly_orig)
    # Plot vertices with gt labels
    for i in range(len(x_orig_centered)-1):
        x = x_orig_centered[i]
        y = y_orig_centered[i]
        label = gt_labels[i]
        ax[0].scatter(x, y, color=label_colors[int(label)], s=200, edgecolor='black', zorder=10)

    poly_orig = Polygon(np.column_stack([x_orig_centered, y_orig_centered]),
                        facecolor='grey',  # First pastel color
                        edgecolor='black',
                        alpha=0.5)
    ax[1].add_patch(poly_orig)
    for i in range(len(x_orig_centered)-1):
        x = x_orig_centered[i]
        y = y_orig_centered[i]
        label = pred_labels[i]
        ax[1].scatter(x, y, color=label_colors[label], s=300, edgecolor='black', zorder=10)

    # Create custom legend elements for the vertex labels
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=label_colors[0], markersize=10, label='Remove'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=label_colors[1], markersize=10, label='Keep'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=label_colors[2], markersize=10, label='Move')
    ]

    ax[0].set_title('Ground Truth', fontsize=20)
    ax[1].plot(x_orig_centered, y_orig_centered, 'k-', alpha=0.5, linewidth=1.5)
    ax[1].set_title('Predicetd labels', fontsize=20)
    ax[0].set_aspect('equal')
    ax[0].set_axis_off()
    ax[1].set_aspect('equal')
    ax[1].set_axis_off()
    fig.tight_layout()
    fig.legend(handles=custom_lines, loc='center', fontsize=20)

    if title:
        plt.title(title, fontsize=24)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_reconstructed_footprints(original_vertices, reconstructed_footprint, ground_truth_footprint,
                                   save_path=None, title=None, ax = None, show_legend=True, show = True, show_labels = True, value = None):

    sns.set_theme(style="white")
    colors = sns.color_palette("pastel")

    if ax is None:
        plt.figure(figsize=(12, 10))
        ax = plt.gca()

    if isinstance(original_vertices,SHPolygon):
        x_orig,y_orig = original_vertices.exterior.xy
    else:
        x_orig = [point.x for point in original_vertices]
        y_orig = [point.y for point in original_vertices]

    centroid_x = np.mean(x_orig)
    centroid_y = np.mean(y_orig)

    x_orig_centered = x_orig - centroid_x
    y_orig_centered = y_orig - centroid_y

    # Close the polygon if needed
    if (x_orig_centered[0] != x_orig_centered[-1]) or (y_orig_centered[0] != y_orig_centered[-1]):
        x_orig_centered = np.append(x_orig_centered, x_orig_centered[0])
        y_orig_centered = np.append(y_orig_centered, y_orig_centered[0])

    if show_labels:
        poly_orig = Polygon(np.column_stack([x_orig_centered, y_orig_centered]),
                            facecolor=colors[0],
                            edgecolor='none',
                            alpha=0.5,
                            label='Original')
    else:
        poly_orig = Polygon(np.column_stack([x_orig_centered, y_orig_centered]),
                            facecolor=colors[0],
                            edgecolor='none',
                            alpha=0.5)

    ax.add_patch(poly_orig)

    # Plot ground truth footprint with orange line
    if isinstance(ground_truth_footprint, SHPolygon):
        x_gt, y_gt = ground_truth_footprint.exterior.xy
    else:
        x_gt = [point.x for point in ground_truth_footprint]
        y_gt = [point.y for point in ground_truth_footprint]

    x_gt_centered = x_gt - centroid_x
    y_gt_centered = y_gt - centroid_y

    # Close the polygon if needed
    if (x_gt_centered[0] != x_gt_centered[-1]) or (y_gt_centered[0] != y_gt_centered[-1]):
        x_gt_centered = np.append(x_gt_centered, x_gt_centered[0])
        y_gt_centered = np.append(y_gt_centered, y_gt_centered[0])

    if show_labels:
        ax.plot(x_gt_centered, y_gt_centered, '-', color=colors[1], linewidth=2.5, label='Ground Truth')
    else:
        ax.plot(x_gt_centered, y_gt_centered, '-', color=colors[1], linewidth=2.5)

    if isinstance(reconstructed_footprint, SHPolygon):
        x_recon, y_recon = reconstructed_footprint.exterior.xy
    else:
        x_recon = [point.x for point in reconstructed_footprint]
        y_recon = [point.y for point in reconstructed_footprint]

    # Center reconstructed footprint using the same centroid
    x_recon_centered = x_recon - centroid_x
    y_recon_centered = y_recon - centroid_y

    # Close the polygon if needed
    if (x_recon_centered[0] != x_recon_centered[-1]) or (y_recon_centered[0] != y_recon_centered[-1]):
        x_recon_centered = np.append(x_recon_centered, x_recon_centered[0])
        y_recon_centered = np.append(y_recon_centered, y_recon_centered[0])

    # Use matplotlib plot for reliable line display with seaborn styling
    if show_labels:
        ax.plot(x_recon_centered, y_recon_centered, '--', color='black', linewidth=2,
             label='Prediction')
    else:
        ax.plot(x_recon_centered, y_recon_centered, '--', color='black', linewidth=2)

    if show_legend:
        ax.legend(loc='best', frameon=True, fontsize = 20)

    if title:
        plt.title(title, fontsize=24)

    if value is not None:
        ax.text(0.01, 0.01, value, fontsize=14,
                verticalalignment='bottom', horizontalalignment='left',
                transform=ax.transAxes)

    # Keep equal aspect ratio
    ax.set_aspect('equal')
    ax.set_axis_off()

    # Add slight padding around the plot
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    elif show:
        plt.show()
        
def plot_iou_hd_quantiles(hd_df, iou_df, plot_dir, quantiles = [0.,0.25,0.5,0.75,1.0], plot_uppers = True):
    hd_lowers = [hd_df.loc[hd_df.hd_dist == hd_df.hd_dist.quantile(q, interpolation='lower')].iloc[0] for q in quantiles]
    hd_uppers = [hd_df.loc[hd_df.hd_dist == hd_df.hd_dist.quantile(q, interpolation='higher')].iloc[0] for q in quantiles]
    iou_lowers =  [iou_df.loc[iou_df.jaccard_coeff == iou_df.jaccard_coeff.quantile(q, interpolation='lower')].iloc[0] for q in quantiles]
    iou_uppers =  [iou_df.loc[iou_df.jaccard_coeff == iou_df.jaccard_coeff.quantile(q, interpolation='higher')].iloc[0] for q in quantiles]

    nrows = 2
    ncols = len(quantiles)
    fig,axs = plt.subplots(nrows,ncols, figsize=(12,8))
    sns.set_theme(style="white")
    for i in range(len(quantiles)-1):
        plot_reconstructed_footprints(hd_lowers[i].geometry, hd_lowers[i].reconstructed_pred, hd_lowers[i].reconstructed_gt, ax = axs[0,i], show_legend=False, show=False, value = np.round(hd_lowers[i].hd_dist, 3), show_labels=False)
        if plot_uppers:
            plot_reconstructed_footprints(hd_uppers[i].geometry, hd_uppers[i].reconstructed_pred, hd_uppers[i].reconstructed_gt, ax = axs[i,1], show_legend=False, show=False,value = np.round(hd_uppers[i].hd_dist,3), show_labels=False)
            plot_reconstructed_footprints(iou_lowers[i].geometry, iou_lowers[i].reconstructed_pred, iou_lowers[i].reconstructed_gt, ax = axs[i,2], show_legend=False, show=False,value = np.round(iou_lowers[i].jaccard_coeff, 3), show_labels=False)
            plot_reconstructed_footprints(iou_uppers[i].geometry, iou_uppers[i].reconstructed_pred, iou_uppers[i].reconstructed_gt, ax = axs[i,3], show_legend=False, show=False,value = np.round(iou_uppers[i].jaccard_coeff,3),show_labels=False)
        else:
            plot_reconstructed_footprints(iou_lowers[i].geometry, iou_lowers[i].reconstructed_pred, iou_lowers[i].reconstructed_gt, ax = axs[1,i], show_legend=False, show=False,value = np.round(iou_lowers[i].jaccard_coeff, 3),show_labels=False)

    plot_reconstructed_footprints(hd_lowers[-1].geometry, hd_lowers[-1].reconstructed_pred,
                                   hd_lowers[-1].reconstructed_gt, ax=axs[0,-1], show_legend=False, show=False,
                                   value=np.round(hd_lowers[-1].hd_dist, 3), show_labels=False)
    plot_reconstructed_footprints(iou_lowers[-1].geometry, iou_lowers[-1].reconstructed_pred,
                                   iou_lowers[-1].reconstructed_gt, ax=axs[1,-1], show_legend=False, show=False,
                                   value=np.round(iou_lowers[-1].jaccard_coeff, 3), show_labels=True)

    column_titles = ['1-Quantile','2-Quantile','3-Quantile','4-Quantile']
    for j, label in enumerate(column_titles):
        # Calculate x position for each column title (center of each column)
        x_pos = (j + 0.5) / 4  # For 4 columns: 0.125, 0.375, 0.625, 0.875
        fig.text(x_pos, 0.95, label, fontsize=18, ha='center', va='center')

    row_labels = ['HD', '1-IoU']
    for i, label in enumerate(row_labels):
        # Position for row label: x = 0.01 (1% from left), y = center of the row
        y_pos = (1 - (i + 0.5) / 2)  # Positions at 0.75 and 0.25 for two rows
        fig.text(0.02, y_pos, label, fontsize=20, rotation=90,
                 ha='center', va='center')
    fig.legend(loc='center', frameon=True, fontsize=16)
    fig.tight_layout()
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(os.path.join(plot_dir, "iou_hd_quantiles.png"), dpi=300, bbox_inches='tight')
    plt.show()

    def create_fig1():
        # ----------------------------
        # LaTeX Setup for matplotlib
        # ----------------------------
        mpl.rcParams['text.usetex'] = True
        mpl.rcParams['text.latex.preamble'] = r"""
        \usepackage[dvipsnames]{xcolor}
        \definecolor{lime}{rgb}{0.00,1.00,0.00}
        \definecolor{dodgerblue}{rgb}{0.12,0.56,1.00}
        \definecolor{cadmiumgreen}{rgb}{0.0, 0.42, 0.24}
        \definecolor{bleudefrance}{rgb}{0.19, 0.55, 0.91}
        """

        # ----------------------------
        # Common Settings & Node Positions
        # ----------------------------
        n_nodes = 6
        theta = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
        radius = 3
        # Position nodes evenly on a circle.
        pos = {i: (radius * np.cos(t), radius * np.sin(t)) for i, t in enumerate(theta)}
        node_size = 700
        node_radius_pt = np.sqrt(node_size) / 2  # ≈13.2 points
        # ----------------------------
        # Create Figure and Grid Layout (3 segments: Input, Middle, Output)
        # ----------------------------
        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.3)

        def draw_offset_labels(ax, pos, labels, fontsize=12, offset_x=0.3, offset_y=0.2):

            for i, text in labels.items():
                x, y = pos[i]
                # Choose alignment and a small offset in data coordinates
                dx = offset_x if x >= 0 else -offset_x
                dy = offset_y if y >= 0 else -offset_y

                if x > 0 and y > 0:
                    ha, va = 'left', 'bottom'
                elif x < 0 and y > 0:
                    ha, va = 'right', 'bottom'
                elif x < 0 and y < 0:
                    ha, va = 'right', 'top'
                elif x > 0 and y < 0:
                    ha, va = 'left', 'top'
                else:
                    ha, va = 'center', 'center'

                ax.text(x + dx, y + dy, text,
                        ha=ha, va=va, fontsize=fontsize)

        def draw_edges_with_shrink(ax, pos, edgelist,
                                   arrowstyle='-|>',
                                   mutation_scale=15,
                                   connectionstyle='arc3,rad=0.0',
                                   color='black',
                                   linewidth=1.5,
                                   shrink_pts=13.2):
            """
            Draw each edge (u,v) in edgelist as a FancyArrowPatch that stops
            `shrink_pts` away from each node center.
            """
            for u, v in edgelist:
                xyA = pos[u]
                xyB = pos[v]
                patch = FancyArrowPatch(xyA, xyB,
                                        arrowstyle=arrowstyle,
                                        mutation_scale=mutation_scale,
                                        connectionstyle=connectionstyle,
                                        color=color,
                                        linewidth=linewidth,
                                        shrinkA=shrink_pts,
                                        shrinkB=shrink_pts)
                ax.add_patch(patch)

        def draw_self_loop_edges(ax, pos, edgelist,
                                 arrowstyle='->',
                                 mutation_scale=15,
                                 color='black',
                                 linewidth=1.5,
                                 shrink_pts=13.2,
                                 rad=2):
            # grab the extents of your layout
            xs = [p[0] for p in pos.values()]
            ys = [p[1] for p in pos.values()]
            xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
            offset_x = 0.4
            offset_y = 0.8
            for u, v in edgelist:
                x, y = pos[u]
                # decide which side this node lives on:
                if abs(y - ymin) < 1e-6:  # bottom row
                    sign = 1
                    dx = offset_x
                    dy = 0
                elif abs(y - ymax) < 1e-6:  # top row
                    sign = -1
                    dx = offset_x
                    dy = 0
                elif abs(x - xmin) < 1e-6:  # left column (center‑left)
                    sign = -1
                    dx = 0
                    dy = offset_y
                else:  # right column (center‑right)
                    sign = 1
                    dx = 0
                    dy = offset_y
                loop = FancyArrowPatch(
                    (x - dx, y - dy), (x + dx, y + dy),
                    connectionstyle=f"arc3,rad={sign * rad}",
                    color=color,
                    shrinkA=shrink_pts,
                    shrinkB=shrink_pts,
                    mutation_scale=mutation_scale,
                    linewidth=linewidth,
                    arrowstyle=arrowstyle
                )
                ax.add_patch(loop)

        # ----------------------------
        # Segment 1: Input Graph (no node fill, outlined only)
        # ----------------------------
        ax_input = fig.add_subplot(gs[0, 0])
        G_input = nx.Graph()
        G_input.add_nodes_from(range(n_nodes))
        for i in range(n_nodes - 1):
            G_input.add_edge(i, i + 1)
        G_input.add_edge(0, n_nodes - 1)

        # Labels: (x^1, y^1), (x^2, y^2), ...
        labels_input = {i: r"$[\textbf x^{{{0}}}, \textbf y^{{{0}}}]$".format(i + 1) for i in range(n_nodes)}

        # 1) Draw the nodes:
        nx.draw_networkx_nodes(
            G_input, pos,
            node_color='none',
            edgecolors='black',
            node_size=node_size,
            ax=ax_input
        )
        # Draw nodes with no fill (node_color='none') and black outlines using edgecolors.
        draw_edges_with_shrink(ax_input, pos, list(G_input.edges()),
                               color='gray',
                               arrowstyle='-',
                               linewidth=2.,
                               mutation_scale=1,
                               shrink_pts=node_radius_pt)
        draw_offset_labels(ax_input, pos, labels_input, offset_x=0.4, offset_y=-0.4, fontsize=26)
        ax_input.set_title("Input Feature Channels", fontsize=28)
        ax_input.axis('off')

        # ----------------------------
        # Segment 2: Convolutional Layers (Directed Graph)
        # ----------------------------
        # Build directed graph with edges from predecessor, successor, and self-loop.
        G_conv = nx.DiGraph()
        G_conv.add_nodes_from(range(n_nodes))
        for i in range(n_nodes):
            pred = (i - 1) % n_nodes
            succ = (i + 1) % n_nodes
            G_conv.add_edge(pred, i)
            G_conv.add_edge(succ, i)
            G_conv.add_edge(i, i)  # Self-loop

        # labels_conv = {i: r"$(x^{{{0}}}, y^{{{0}}})$".format(i+1) for i in range(n_nodes)}

        # Create a subgrid for the middle segment with 3 rows:
        # Row 1: Green filter panel, Row 2: Blue filter panel, Row 3: Text panel (dots and Conv1D title)
        conv_gs = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0, 1], hspace=0.3)

        # Define arrow styles for the panels.
        arrow_kwargs = {
            'arrowstyle': '->',
            'mutation_scale': 15,
            'linewidth': 2,
            'shrink_pts': node_radius_pt
        }

        # Separate non-self-loop edges and self-loop edges.
        non_self_edges = [e for e in G_conv.edges() if e[0] != e[1]]
        # self_edges = list(nx.selfloop_edges(G_conv))
        self_edges = [(i, i) for i in range(n_nodes)]

        # ---- Row 1: Green Filter Panel ----
        ax_conv_green = fig.add_subplot(conv_gs[0, 0])
        nx.draw_networkx_nodes(G_conv, pos, node_color='none', edgecolors='black', node_size=node_size,
                               ax=ax_conv_green)
        draw_edges_with_shrink(ax_conv_green, pos, non_self_edges, connectionstyle='arc3,rad=0.2', color='lime',
                               **arrow_kwargs)
        draw_self_loop_edges(ax_conv_green, pos, self_edges, color='lime', rad=2., **arrow_kwargs)
        # draw_offset_labels(ax_conv_green, pos, labels_conv, offset_x = 0.4, offset_y = 1.3, fontsize=12)
        ax_conv_green.axis('off')

        # ---- Row 2: Blue Filter Panel ----
        ax_conv_blue = fig.add_subplot(conv_gs[1, 0])
        nx.draw_networkx_nodes(G_conv, pos, node_color='none', edgecolors='black', node_size=node_size, ax=ax_conv_blue)
        draw_edges_with_shrink(ax_conv_blue, pos, non_self_edges, connectionstyle='arc3,rad=0.2', color='dodgerblue',
                               **arrow_kwargs)
        draw_self_loop_edges(ax_conv_blue, pos, self_edges, color='dodgerblue', rad=2., **arrow_kwargs)
        # draw_offset_labels(ax_conv_blue, pos, labels_conv, fontsize=12)
        ax_conv_blue.axis('off')

        # ---- Row 3: Text Panel: Large dots and Conv1D Title ----
        ax_conv_text = fig.add_subplot(conv_gs[2, 0])
        ax_conv_text.axis('off')
        conv_text = r"\Huge{$\ldots$}" + '\n \n' + r"\Huge{Conv1d: 3, circular padding}"
        ax_conv_text.text(0.5, 0.5, conv_text, ha='center', va='bottom', fontsize=30)

        # ----------------------------
        # Segment 3: Output Graph (Feature Channels, no node fill)
        # ----------------------------
        ax_output = fig.add_subplot(gs[0, 2])
        G_output = nx.Graph()
        G_output.add_nodes_from(range(n_nodes))
        for i in range(n_nodes - 1):
            G_output.add_edge(i, i + 1)
        G_output.add_edge(0, n_nodes - 1)

        # Output labels: feature activations (with brackets),
        # first channel in lime, second in dodgerblue.
        labels_output = {}
        for i in range(n_nodes):
            labels_output[i] = (
                r"$[\textcolor{{cadmiumgreen}}{{\Huge \textbf h^{{{0}}}_1}}, \textcolor{{blue}}{{\Huge \textbf h^{{{0}}}_2}}, \dots]$".format(
                    i + 1)
            )

        # Draw nodes with no fill.
        nx.draw_networkx_nodes(G_output, pos, node_color='none', edgecolors='black', node_size=node_size, ax=ax_output)
        draw_edges_with_shrink(ax_output, pos, list(G_output.edges()),
                               color='gray',
                               arrowstyle='-',
                               mutation_scale=1,
                               linewidth=1.5,
                               shrink_pts=node_radius_pt)
        draw_offset_labels(ax_output, pos, labels_output, offset_x=-0.4, offset_y=-0.5, fontsize=30)
        ax_output.set_title("Output Feature Channels", fontsize=28)
        ax_output.axis('off')

        # ----------------------------
        # Figure-Level Arrows Between Segments (Corrected Direction)
        # ----------------------------
        ax_ann = fig.add_axes([0, 0, 1, 1], zorder=-1)
        ax_ann.set_axis_off()

        # Arrow from Input to Middle:
        # Tail at Input right border and head at Middle (Conv text panel) left border.
        ax_ann.annotate("",
                        xy=(0.40, 0.5),  # head: near left side of middle segment
                        xytext=(0.37, 0.5),  # tail: near right side of Input segment
                        xycoords='figure fraction',
                        arrowprops=dict(arrowstyle="->", lw=4, color='black'))
        # Arrow from Middle to Output:
        # Tail at Middle right border and head at Output left border.
        ax_ann.annotate("",
                        xy=(0.66, 0.5),  # head: near left side of Output segment
                        xytext=(0.63, 0.5),  # tail: near right side of Middle segment
                        xycoords='figure fraction',
                        arrowprops=dict(arrowstyle="->", lw=4, color='black'))
        if not os.path.exists('plots'):
            os.makedirs('plots')
        fig.savefig("plots/message_passing.eps")
        plt.show()