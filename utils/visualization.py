import plotly.graph_objects as go
import umap


def generate_umap(audio_embeddings, text_embeddings, text_labels, audio_ids):
    reducer = umap.UMAP(random_state=1337)
    audio_embeddings_2d = reducer.fit_transform(audio_embeddings)

    audio_hover_over = [
        f"{audio_id}: {text}" for audio_id, text in zip(audio_ids, text_labels)
    ]

    trace_audio = go.Scatter(
        x=audio_embeddings_2d[:, 0],
        y=audio_embeddings_2d[:, 1],
        mode="markers",
        marker=dict(color="blue"),
        name="Audio Embeddings",
        text=audio_hover_over,  # Audio IDs for hover
        hoverinfo="text",
    )

    if text_embeddings is not None:
        text_embeddings_2d = reducer.transform(text_embeddings)
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

    else:
        # Combine the plots
        fig = go.Figure(data=[trace_audio])
        # Customize layout
        fig.update_layout(
            title="Audio Embeddings",
            xaxis_title="Component 1",
            yaxis_title="Component 2",
        )

    return fig


def generate_vae_umap(latents, text_labels):
    reducer = umap.UMAP(random_state=1337)
    latents_flattened = latents.reshape(latents.shape[0], -1)
    latents_2d = reducer.fit_transform(latents_flattened)

    hover_over = [f"{text}" for text in text_labels]

    trace = go.Scatter(
        x=latents_2d[:, 0],
        y=latents_2d[:, 1],
        mode="markers",
        marker=dict(color="blue"),
        name="Latents",
        text=hover_over,  # Audio IDs for hover
        hoverinfo="text",
    )

    # Combine the plots
    fig = go.Figure(data=[trace])
    # Customize layout
    fig.update_layout(
        title="Image Latent Space",
        xaxis_title="Component 1",
        yaxis_title="Component 2",
    )

    return fig
