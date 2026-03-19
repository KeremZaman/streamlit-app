#!/usr/bin/env python3
"""
Streamlit dashboard for investigating steering evaluation results.

Run with:
    streamlit run maze_generation/steering_dashboard.py
"""

import glob
import json
import os
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

SCRIPT_DIR = Path(__file__).resolve().parent

TASK_DIRS = {
    "maze": SCRIPT_DIR / "steering_eval_output",
    "maze_segment_based": SCRIPT_DIR / "steering_eval_output_segment_based",
    "countdown": SCRIPT_DIR / "countdown_steering_eval_output",
    "pubmedqa": SCRIPT_DIR / "pubmedqa_steering_eval_output",
}

BACKTRACK_PHRASES = [
    "wait", "alternatively", "go back", "backtrack", "let me reconsider",
    "on second thought", "actually", "however", "but then", "re-evaluate",
    "reconsider", "I was wrong", "that's not right", "let me rethink",
    "dead end", "wrong path", "try another", "different path",
]


@st.cache_data
def load_all_results() -> tuple[pd.DataFrame, list[dict]]:
    """Load all steering_results*.json files into a unified DataFrame and raw sample list."""
    rows: list[dict] = []
    raw_samples: list[dict] = []

    for task_name, task_dir in TASK_DIRS.items():
        if not task_dir.exists():
            continue
        for path in sorted(task_dir.glob("steering_results*.json")):
            with open(path) as f:
                data = json.load(f)

            config = data["config"]
            layer = config["layer"]
            model = config.get("model", "unknown")
            method = config.get("method", "unknown")

            for alpha_str, res in data["results_by_alpha"].items():
                alpha = float(alpha_str)
                for sample in res["samples"]:
                    backtrack_mentions = _get_backtrack_mentions(sample, task_name)

                    row = {
                        "task": task_name,
                        "model": model,
                        "layer": layer,
                        "method": method,
                        "alpha": alpha,
                        "index": sample["index"],
                        "correct": bool(sample["correct"]),
                        "total_backtrack_phrases": sample.get("total_backtrack_phrases", 0),
                        "backtrack_mentions": backtrack_mentions,
                        "backtracking_char_ratio": sample.get("backtracking_char_ratio", 0.0),
                        "num_backtracking_segments": sample.get("num_backtracking_segments", 0),
                    }

                    for phrase in BACKTRACK_PHRASES:
                        row[f"phrase_{phrase}"] = sample.get("backtrack_phrase_counts", {}).get(phrase, 0)

                    if task_name.startswith("maze"):
                        row["num_nodes"] = sample.get("num_nodes")
                        row["branching_node_mentions"] = sample.get("branching_node_mentions", 0)
                        row["neighbor_mentions"] = sample.get("neighbor_mentions", 0)
                    elif task_name == "countdown":
                        row["target"] = sample.get("target") or sample.get("metadata", {}).get("target")
                        row["num_count"] = len(sample.get("nums", sample.get("metadata", {}).get("nums", [])))
                    elif task_name == "pubmedqa":
                        row["gold_answer"] = sample.get("gold_answer") or sample.get("metadata", {}).get("final_decision")
                        row["parsed_answer"] = sample.get("parsed_answer")

                    rows.append(row)

                    raw_sample = {**row}
                    raw_sample["prompt"] = sample.get("prompt", "")
                    raw_sample["thinking_text"] = sample.get("thinking_text", "")
                    raw_sample["model_output"] = sample.get("model_output", "")
                    raw_sample["backtrack_phrase_counts"] = sample.get("backtrack_phrase_counts", {})
                    raw_sample["judge_segments"] = sample.get("judge_segments")
                    if task_name == "countdown":
                        raw_sample["parsed_equation"] = sample.get("parsed_equation", "")
                    if task_name.startswith("maze"):
                        raw_sample["parsed_path"] = sample.get("parsed_path")
                    raw_samples.append(raw_sample)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["task", "layer", "alpha", "index"]).reset_index(drop=True)
    return df, raw_samples


def _get_backtrack_mentions(sample: dict, task_name: str) -> float:
    if "branching_node_mentions" in sample:
        return sample["branching_node_mentions"] + sample["neighbor_mentions"]
    return sample.get("num_backtracking_segments", 0)


def highlight_backtrack_phrases(text: str) -> str:
    """Wrap backtracking phrases in colored HTML spans."""
    if not text:
        return ""
    escaped = (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace("\n", "<br>")
    )
    for phrase in sorted(BACKTRACK_PHRASES, key=len, reverse=True):
        escaped_phrase = phrase.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        pattern = re.compile(re.escape(escaped_phrase), re.IGNORECASE)
        escaped = pattern.sub(
            lambda m: f'<span style="background-color:#ff6b6b;color:white;padding:1px 4px;border-radius:3px;font-weight:600">{m.group()}</span>',
            escaped,
        )
    return escaped


def render_judge_segments(segments: list[dict]) -> str:
    """Render judge segments as color-coded HTML blocks."""
    if not segments:
        return ""
    html_parts = []
    for seg in segments:
        label = seg.get("label", "other")
        text = seg.get("segment_text", "")
        escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        if label == "backtracking":
            color = "#ff6b6b"
            bg = "#fff0f0"
            border = "#ff4444"
        else:
            color = "#333"
            bg = "#f0f8f0"
            border = "#66bb6a"
        html_parts.append(
            f'<div style="border-left:4px solid {border};background:{bg};color:{color};'
            f'padding:8px 12px;margin:4px 0;border-radius:0 4px 4px 0;font-size:0.85em">'
            f'<strong>[{label.upper()}]</strong> {escaped}</div>'
        )
    return "".join(html_parts)


def apply_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Build sidebar filters and return filtered DataFrame."""
    st.sidebar.header("Filters")

    tasks = sorted(df["task"].unique())
    selected_tasks = st.sidebar.multiselect("Task", tasks, default=tasks)

    models = sorted(df["model"].unique())
    selected_models = st.sidebar.multiselect("Model", models, default=models)

    layers = sorted(df["layer"].unique())
    selected_layers = st.sidebar.multiselect("Layer", [int(l) for l in layers], default=[int(l) for l in layers])

    alphas = sorted(df["alpha"].unique())
    selected_alphas = st.sidebar.multiselect("Alpha", alphas, default=alphas)

    correctness = st.sidebar.radio("Correctness", ["All", "Correct only", "Incorrect only"])

    mask = (
        df["task"].isin(selected_tasks)
        & df["model"].isin(selected_models)
        & df["layer"].isin(selected_layers)
        & df["alpha"].isin(selected_alphas)
    )
    if correctness == "Correct only":
        mask &= df["correct"]
    elif correctness == "Incorrect only":
        mask &= ~df["correct"]

    return df[mask].copy()


def tab_aggregate(fdf: pd.DataFrame):
    """Render Aggregate Stats tab."""
    st.subheader("Accuracy vs Alpha")

    if fdf.empty:
        st.warning("No data matches the current filters.")
        return

    fdf["layer_str"] = "Layer " + fdf["layer"].astype(str)
    group_col = "task" if fdf["task"].nunique() > 1 else "layer_str"

    agg = fdf.groupby(["task", "layer", "alpha"]).agg(
        count=("correct", "size"),
        accuracy=("correct", "mean"),
        avg_backtrack_phrases=("total_backtrack_phrases", "mean"),
        avg_backtrack_mentions=("backtrack_mentions", "mean"),
        avg_char_ratio=("backtracking_char_ratio", "mean"),
    ).reset_index()
    agg["label"] = agg["task"] + " / Layer " + agg["layer"].astype(str)

    fig_acc = px.line(
        agg, x="alpha", y="accuracy", color="label",
        markers=True, title="Accuracy vs Alpha",
        labels={"alpha": "Alpha", "accuracy": "Accuracy", "label": "Task / Layer"},
    )
    fig_acc.update_layout(yaxis_range=[-0.05, 1.05], hovermode="x unified")
    st.plotly_chart(fig_acc, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_bp = px.line(
            agg, x="alpha", y="avg_backtrack_phrases", color="label",
            markers=True, title="Avg Backtrack Phrase Count vs Alpha",
            labels={"avg_backtrack_phrases": "Avg Backtrack Phrases", "label": "Task / Layer"},
        )
        fig_bp.update_layout(hovermode="x unified")
        st.plotly_chart(fig_bp, use_container_width=True)

    with col2:
        fig_bm = px.line(
            agg, x="alpha", y="avg_backtrack_mentions", color="label",
            markers=True, title="Avg Backtrack Mentions vs Alpha",
            labels={"avg_backtrack_mentions": "Avg Backtrack Mentions", "label": "Task / Layer"},
        )
        fig_bm.update_layout(hovermode="x unified")
        st.plotly_chart(fig_bm, use_container_width=True)

    st.subheader("Summary Table")
    display_agg = agg.rename(columns={
        "task": "Task", "layer": "Layer", "alpha": "Alpha", "count": "N",
        "accuracy": "Accuracy", "avg_backtrack_phrases": "Avg BT Phrases",
        "avg_backtrack_mentions": "Avg BT Mentions", "avg_char_ratio": "Avg BT Char Ratio",
    })
    display_agg = display_agg.drop(columns=["label"])
    st.dataframe(
        display_agg.style.format({
            "Accuracy": "{:.3f}", "Avg BT Phrases": "{:.2f}",
            "Avg BT Mentions": "{:.2f}", "Avg BT Char Ratio": "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )


def tab_sample_explorer(fdf: pd.DataFrame, raw_samples: list[dict]):
    """Render Sample Explorer tab."""
    st.subheader("Sample Explorer")

    if fdf.empty:
        st.warning("No data matches the current filters.")
        return

    display_cols = ["task", "layer", "alpha", "index", "correct", "total_backtrack_phrases", "backtrack_mentions"]
    task_set = set(fdf["task"].unique())
    if task_set & {"maze", "maze_segment_based"}:
        display_cols.append("num_nodes")
    if "countdown" in task_set:
        display_cols.append("num_count")
    if "pubmedqa" in task_set:
        display_cols += ["parsed_answer", "gold_answer"]
    display_cols.append("backtracking_char_ratio")

    available_cols = [c for c in display_cols if c in fdf.columns]
    table_df = fdf[available_cols].reset_index(drop=True)

    sort_col = st.selectbox("Sort by", available_cols, index=available_cols.index("total_backtrack_phrases") if "total_backtrack_phrases" in available_cols else 0)
    sort_asc = st.checkbox("Ascending", value=False)
    table_df = table_df.sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)
    fdf_sorted = fdf.sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)

    st.dataframe(table_df, use_container_width=True, hide_index=True, height=300)

    raw_lookup = {}
    for rs in raw_samples:
        key = (rs["task"], rs["layer"], rs["alpha"], rs["index"])
        raw_lookup[key] = rs

    st.markdown("---")
    st.subheader("Inspect Individual Sample")

    col_task, col_layer, col_alpha, col_idx = st.columns(4)
    with col_task:
        tasks_avail = sorted(fdf_sorted["task"].unique())
        sel_task = st.selectbox("Task", tasks_avail, key="explore_task")
    with col_layer:
        layers_avail = sorted(fdf_sorted[fdf_sorted["task"] == sel_task]["layer"].unique())
        sel_layer = st.selectbox("Layer", layers_avail, key="explore_layer")
    with col_alpha:
        alphas_avail = sorted(fdf_sorted[(fdf_sorted["task"] == sel_task) & (fdf_sorted["layer"] == sel_layer)]["alpha"].unique())
        sel_alpha = st.selectbox("Alpha", alphas_avail, key="explore_alpha")
    with col_idx:
        indices_avail = sorted(fdf_sorted[
            (fdf_sorted["task"] == sel_task) &
            (fdf_sorted["layer"] == sel_layer) &
            (fdf_sorted["alpha"] == sel_alpha)
        ]["index"].unique())
        sel_idx = st.selectbox("Sample Index", indices_avail, key="explore_idx")

    sample = raw_lookup.get((sel_task, sel_layer, sel_alpha, sel_idx))
    if sample is None:
        st.error("Sample not found.")
        return

    correct_badge = "✅ Correct" if sample["correct"] else "❌ Incorrect"
    st.markdown(f"### Sample {sel_idx} &mdash; {correct_badge}")

    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric("Backtrack Phrases", sample["total_backtrack_phrases"])
    mcol2.metric("Backtrack Mentions", f"{sample['backtrack_mentions']:.0f}")
    mcol3.metric("BT Char Ratio", f"{sample.get('backtracking_char_ratio', 0):.3f}")

    if sel_task == "countdown":
        st.info(f"**Target:** {sample.get('target')} | **Parsed Equation:** `{sample.get('parsed_equation', 'N/A')}`")
    elif sel_task == "pubmedqa":
        st.info(f"**Gold Answer:** {sample.get('gold_answer')} | **Model Answer:** {sample.get('parsed_answer', 'N/A')}")
    elif sel_task.startswith("maze"):
        st.info(f"**Nodes:** {sample.get('num_nodes')} | **Parsed Path:** {sample.get('parsed_path')}")

    with st.expander("Prompt", expanded=False):
        st.code(sample.get("prompt", ""), language=None)

    with st.expander("Thinking Text (backtrack phrases highlighted)", expanded=True):
        highlighted = highlight_backtrack_phrases(sample.get("thinking_text", ""))
        st.markdown(f'<div style="max-height:500px;overflow-y:auto;font-size:0.85em;line-height:1.6">{highlighted}</div>', unsafe_allow_html=True)

    phrase_counts = sample.get("backtrack_phrase_counts", {})
    nonzero = {k: v for k, v in phrase_counts.items() if v > 0}
    if nonzero:
        with st.expander("Backtrack Phrase Breakdown", expanded=False):
            phrase_df = pd.DataFrame(list(nonzero.items()), columns=["Phrase", "Count"]).sort_values("Count", ascending=True)
            fig = px.bar(phrase_df, x="Count", y="Phrase", orientation="h", title="Backtrack Phrase Counts")
            fig.update_layout(height=max(200, len(nonzero) * 35))
            st.plotly_chart(fig, use_container_width=True)

    segments = sample.get("judge_segments")
    if segments:
        with st.expander("Judge Segments (color-coded)", expanded=False):
            bt_count = sum(1 for s in segments if s.get("label") == "backtracking")
            other_count = len(segments) - bt_count
            st.caption(f"{len(segments)} segments total: {bt_count} backtracking, {other_count} other")
            html = render_judge_segments(segments)
            st.markdown(html, unsafe_allow_html=True)


def tab_backtracking_analysis(fdf: pd.DataFrame):
    """Render Backtracking Analysis tab."""
    st.subheader("Backtracking Analysis")

    if fdf.empty:
        st.warning("No data matches the current filters.")
        return

    fdf["label"] = fdf["task"] + " / Layer " + fdf["layer"].astype(str)

    st.markdown("#### Distribution of Backtrack Phrase Count by Alpha")
    fig_box = px.box(
        fdf, x="alpha", y="total_backtrack_phrases", color="label",
        title="Backtrack Phrase Count Distribution by Alpha",
        labels={"total_backtrack_phrases": "Total Backtrack Phrases", "alpha": "Alpha"},
    )
    st.plotly_chart(fig_box, use_container_width=True)

    if "backtracking_char_ratio" in fdf.columns:
        st.markdown("#### Backtracking Char Ratio by Alpha")
        fig_ratio = px.box(
            fdf, x="alpha", y="backtracking_char_ratio", color="label",
            title="Backtracking Char Ratio Distribution",
            labels={"backtracking_char_ratio": "BT Char Ratio", "alpha": "Alpha"},
        )
        st.plotly_chart(fig_ratio, use_container_width=True)

    st.markdown("#### Per-Phrase Heatmap")
    phrase_cols = [c for c in fdf.columns if c.startswith("phrase_")]
    if phrase_cols:
        group_keys = ["task", "layer", "alpha"]
        heatmap_data = fdf.groupby(group_keys)[phrase_cols].mean().reset_index()
        heatmap_data["label"] = heatmap_data["task"] + " / L" + heatmap_data["layer"].astype(str) + " / α=" + heatmap_data["alpha"].astype(str)

        phrase_names = [c.replace("phrase_", "") for c in phrase_cols]
        z_data = heatmap_data[phrase_cols].values
        nonzero_mask = z_data.sum(axis=0) > 0
        phrase_names_filtered = [p for p, m in zip(phrase_names, nonzero_mask) if m]
        z_filtered = z_data[:, nonzero_mask]

        if z_filtered.size > 0:
            fig_heat = go.Figure(data=go.Heatmap(
                z=z_filtered,
                x=phrase_names_filtered,
                y=heatmap_data["label"].tolist(),
                colorscale="YlOrRd",
                hovertemplate="Phrase: %{x}<br>Setting: %{y}<br>Avg Count: %{z:.2f}<extra></extra>",
            ))
            fig_heat.update_layout(
                title="Average Phrase Count per Setting",
                xaxis_title="Backtrack Phrase",
                yaxis_title="Task / Layer / Alpha",
                height=max(400, len(heatmap_data) * 22),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("#### Accuracy vs Backtracking (Scatter)")
    scatter_agg = fdf.groupby(["task", "layer", "alpha"]).agg(
        accuracy=("correct", "mean"),
        avg_bt_phrases=("total_backtrack_phrases", "mean"),
    ).reset_index()
    scatter_agg["label"] = scatter_agg["task"] + " / Layer " + scatter_agg["layer"].astype(str)
    fig_scatter = px.scatter(
        scatter_agg, x="avg_bt_phrases", y="accuracy", color="label",
        symbol=scatter_agg["alpha"].apply(lambda a: "positive" if a >= 0 else "negative"),
        hover_data=["alpha"],
        title="Accuracy vs Avg Backtrack Phrases",
        labels={"avg_bt_phrases": "Avg Backtrack Phrases", "accuracy": "Accuracy"},
    )
    fig_scatter.update_layout(yaxis_range=[-0.05, 1.05])
    st.plotly_chart(fig_scatter, use_container_width=True)


def main():
    st.set_page_config(page_title="Steering Eval Dashboard", layout="wide")
    st.title("Steering Evaluation Dashboard")
    st.caption("Investigate steering vector evaluation results across tasks, layers, and alpha values.")

    df, raw_samples = load_all_results()

    if df.empty:
        st.error("No steering results found. Check that the data directories exist.")
        return

    st.sidebar.markdown(f"**{len(df):,}** total samples loaded across **{df['task'].nunique()}** tasks")
    filtered_df = apply_sidebar_filters(df)
    st.sidebar.markdown(f"**{len(filtered_df):,}** samples after filtering")

    tab1, tab2, tab3 = st.tabs(["Aggregate Stats", "Sample Explorer", "Backtracking Analysis"])

    with tab1:
        tab_aggregate(filtered_df)
    with tab2:
        tab_sample_explorer(filtered_df, raw_samples)
    with tab3:
        tab_backtracking_analysis(filtered_df)


if __name__ == "__main__":
    main()