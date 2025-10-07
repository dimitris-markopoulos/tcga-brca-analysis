import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def plot_label_across_methods(label_name, y, embeddings_dict, palette, save_path):

    methods = list(embeddings_dict.keys())
    embeddings_ = list(embeddings_dict.values())
    label_values = y[label_name].values

    # --- Create grid (3x2 layout) ---
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=methods,
        specs=[[{}, {}],
               [{}, {}],
               [{}, None]],
        horizontal_spacing=0.08,
        vertical_spacing=0.10
    )

    # --- Add scatterplots per method ---
    for i, (name, X_emb) in enumerate(zip(methods, embeddings_)):
        row = i // 2 + 1
        col = i % 2 + 1
        df_plot = pd.DataFrame({
            "x": X_emb[:, 0],
            "y": X_emb[:, 1],
            label_name: label_values
        })

        # Add one scatter trace per unique label
        unique_vals = df_plot[label_name].unique()
        for val in unique_vals:
            sub = df_plot[df_plot[label_name] == val]
            color = palette.get(val, "#999999")  # fallback gray if unseen
            if len(sub) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=sub["x"],
                        y=sub["y"],
                        mode="markers",
                        marker=dict(size=5, color=color, opacity=0.85),
                        name=f"{label_name}: {val}",
                        showlegend=(i == 0)
                    ),
                    row=row, col=col
                )

    # --- Layout ---
    fig.update_layout(
        template="plotly_white",
        height=1000,
        width=900,
        title_text=f"Unsupervised Methods ({label_name})",
        title_x=0.5,
        font=dict(size=13),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.12,
            xanchor="center",
            x=0.5,
            title=None
        ),
        margin=dict(l=40, r=40, t=80, b=90)
    )

    # --- Export & show ---
    fig.write_html(save_path, include_plotlyjs="cdn")
    fig.show()
