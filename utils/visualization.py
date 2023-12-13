import plotly.graph_objects as go
import umap


def generate_umap(audio_embeddings, text_embeddings, text_labels, audio_ids):
    reducer = umap.UMAP(random_state=42)
    audio_embeddings_2d = reducer.fit_transform(audio_embeddings)
    text_embeddings_2d = reducer.transform(text_embeddings)

    audio_hover_over = [
        f"{audio_id}: {text}" for audio_id, text in zip(audio_ids, text_labels)
    ]

    trace_audio = go.Scatter(
        x=audio_embeddings_2d[:, 0],
        y=audio_embeddings_2d[:, 1],
        mode="markers",
        marker=dict(color="blue"),
        name="Audio Embeddings",
        text=audio_ids,  # Audio IDs for hover
        hoverinfo="text",
    )

    # Create scatter plot for text embeddings
    trace_text = go.Scatter(
        x=text_embeddings_2d[:, 0],
        y=text_embeddings_2d[:, 1],
        mode="markers",
        marker=dict(color="red"),
        name="Text Embeddings",
        text=audio_hover_over,  # Combined ID and text label for hover
        hoverinfo="text",
    )

    # Combine the plots
    fig = go.Figure(data=[trace_audio, trace_text])

    # Customize layout
    fig.update_layout(
        title="Audio and Text Embeddings",
        xaxis_title="Component 1",
        yaxis_title="Component 2",
    )

    return fig
