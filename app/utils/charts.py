import plotly.graph_objects as go
import pandas as pd


def line_scatter(df, cols, title):
    fig = go.Figure()
    if isinstance(df, pd.Series):
        df = df.to_frame()

    for col in cols:
        fig.add_trace(go.Scatter(x=df.index, y=df[col],
                                 mode='lines',
                                 name=col))

    fig.update_layout(
        hovermode="x unified",
        height=500,
        title=title
    )
    return fig


def area_chart(df, cols, title):
    fig = go.Figure()
    if isinstance(df, pd.Series):
        df = df.to_frame()

    for col in cols:
        fig.add_trace(go.Scatter(x=df.index, y=df[col],
                                 mode='lines',
                                 name=col,
                                 fill='tozeroy'))

    fig.update_layout(
        hovermode="x unified",
        height=500,
        title=title
    )
    return fig


def bar_chart(df, cols, title):
    fig = go.Figure()
    if isinstance(df, pd.Series):
        df = df.to_frame()

    for col in cols:
        fig.add_trace(go.Bar(x=df.index, y=df[col],
                                 name=col))

    fig.update_layout(
        height=500,
        title=title
    )
    return fig


def indicators(df, col=0, title="", suffix="%"):
    fig = go.Figure()
    multiplier = 100 if suffix == "%" else 1
    if isinstance(df, pd.DataFrame):
        value = df[col].values[0] * multiplier
    elif isinstance(df, pd.Series):
        value = df[col] * multiplier
    else:
        value = df * multiplier

    fig.add_trace(go.Indicator(
        mode="number",
        number={
            'font': {'size': 60},
            'suffix': suffix,
            "valueformat": ".2f"
        },
        value=value,
        title=f"<span style='font-size:500;'><b>{'Portfolio' if col == 0 else col}</b></span><br>{title}"
    ))

    return fig