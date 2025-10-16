import os
import streamlit as st
import pandas as pd
import json
import numpy as np
from collections import defaultdict
from typing import Optional
from label_studio_sdk import Client

# Check if label-studio-sdk is available
try:
    from label_studio_sdk import Client
    LABEL_STUDIO_AVAILABLE = True
except ImportError:
    LABEL_STUDIO_AVAILABLE = False

def display_structured_output_analysis(experimentset):
    """Display structured output analysis with fields as rows and models as columns, with pinned rows"""
    st.subheader("📝 Structured Output Analysis")
    
    # Collect data
    all_field_names = set()
    model_data = {}
    
    # First pass: collect all unique field names and data per model
    for experiment in experimentset.get("experiments", []):
        model_name = experiment.get("model", {}).get("name", "Unknown")
        observation_table = []
        
        for result in experiment.get("results", []):
            if result.get("metric_name") == "llm_structured_output":
                observation_table = result.get("observation_table", [])
        
        global_scores = []
        field_scores_by_name = defaultdict(list)
        errors_count = 0
        
        for obs in observation_table:
            if obs.get("observation"):
                try:
                    obs_data = json.loads(obs["observation"])
                    if "score" in obs_data:
                        global_scores.append(obs_data["score"])
                    field_scores = obs_data.get("field_scores", {})
                    for field_name, field_score in field_scores.items():
                        if field_name != "status" and isinstance(field_score, (int, float)):
                            all_field_names.add(field_name)
                            field_scores_by_name[field_name].append(field_score)
                    if "error" in obs_data:
                        errors_count += 1
                except (json.JSONDecodeError, TypeError):
                    errors_count += 1
        
        model_data[model_name] = {
            "global_score": np.mean(global_scores) if global_scores else None,
            "errors": errors_count,
            "n_items": len(observation_table),
            **{field: np.mean(scores) if scores else None for field, scores in field_scores_by_name.items()}
        }
    
    if not model_data:
        st.info("No structured output results found in this experiment set.")
        return
    
    # Create DataFrame with fields as rows and models as columns
    sorted_field_names = sorted(all_field_names)
    pinned_rows = ["global_score", "n_items", "errors"]
    field_rows = sorted_field_names
    index = pinned_rows + field_rows
    df = pd.DataFrame(index=index, columns=list(model_data.keys()))
    
    for model_name, data in model_data.items():
        for field in index:
            df.loc[field, model_name] = data.get(field)
    
    # Split DataFrame into pinned and scrollable sections
    pinned_df = df.loc[pinned_rows]
    scrollable_df = df.loc[field_rows] if field_rows else pd.DataFrame()
    
    # Format the numeric columns
    format_dict = {col: "{:.3f}" for col in df.columns}
    
    # Create column config
    column_config = {
        model: st.column_config.NumberColumn(
            model,
            help=f"Metrics for model {model}",
            format="%.3f" if model in df.columns else None,
            width="medium"
        ) for model in df.columns
    }
    
    # Apply highlighting
    def highlight_scores(df):
        highlight_df = pd.DataFrame("", index=df.index, columns=df.columns)
        score_rows = [row for row in df.index if row != "n_items" and row != "errors"]
        
        for row in score_rows:
            numeric_row = pd.to_numeric(df.loc[row], errors='coerce')
            if numeric_row.notna().any():
                max_val = numeric_row.max()
                min_val = numeric_row.min()
                for col in df.columns:
                    val = numeric_row[col]
                    if pd.notna(val):
                        if val == max_val:
                            highlight_df.loc[row, col] = "font-weight: bold; color: green"
                        elif val == min_val:
                            highlight_df.loc[row, col] = "font-weight: bold; color: red"
        return highlight_df
    
    # Display pinned section
    st.write("**Pinned Metrics** - Global score, number of items, and errors")
    st.dataframe(
        pinned_df.style.apply(highlight_scores, axis=None).format(format_dict, na_rep="N/A"),
        use_container_width=True,
        column_config=column_config
    )
    
    # Display scrollable section
    if not scrollable_df.empty:
        st.write("**Field Scores** - Scores for each field across models")
        st.dataframe(
            scrollable_df.style.apply(highlight_scores, axis=None).format(format_dict, na_rep="N/A"),
            use_container_width=True,
            column_config=column_config,
            height=400  # Adjustable height for scrollable section
        )
    
    # Show summary statistics
    st.write("---")
    st.write("**Field Performance Summary**")
    
    summary_data = []
    for field in sorted_field_names:
        field_values = pd.to_numeric(df.loc[field], errors='coerce').dropna()
        if len(field_values) > 0:
            summary_data.append({
                "Field": field.capitalize(),
                "Mean Score": f"{field_values.mean():.3f}",
                "Min Score": f"{field_values.min():.3f}",
                "Max Score": f"{field_values.max():.3f}",
                "Std Dev": f"{field_values.std():.3f}" if len(field_values) > 1 else "N/A"
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Show error details
    if df.loc["errors"].sum() > 0:
        with st.expander(f"Error Details ({int(df.loc['errors'].sum())} total errors)", expanded=False):
            for experiment in experimentset.get("experiments", []):
                model_name = experiment.get("model", {}).get("name", "Unknown")
                errors = []
                for result in experiment.get("results", []):
                    if result.get("metric_name") == "llm_structured_output":
                        for obs in result.get("observation_table", []):
                            if obs.get("observation"):
                                try:
                                    obs_data = json.loads(obs["observation"])
                                    if "error" in obs_data:
                                        errors.append({
                                            "Line": obs.get("num_line", "N/A"),
                                            "Error": obs_data["error"]
                                        })
                                except:
                                    errors.append({
                                        "Line": obs.get("num_line", "N/A"),
                                        "Error": "Failed to parse JSON observation"
                                    })
                
                if errors:
                    st.write(f"**{experiment.get('name', 'Unknown')} ({model_name})**")
                    error_df = pd.DataFrame(errors)
                    st.dataframe(error_df, use_container_width=True, hide_index=True)

def get_label_studio_client() -> Optional[Client]:
    """
    Create a Label Studio client using environment variables.
    
    Expected environment variables:
    - LABEL_STUDIO_URL: The URL of your Label Studio instance
    - LABEL_STUDIO_API_KEY: Your Label Studio API key
    """
    url = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080")
    api_key = os.getenv("LABEL_STUDIO_API_KEY")
    
    if not api_key:
        st.warning("⚠️ LABEL_STUDIO_API_KEY environment variable is not set.")
        return None
    
    try:
        client = Client(url=url, api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Failed to connect to Label Studio: {str(e)}")
        return None

def extract_raw_api_response(results):
    """Extract data from results where from_name is raw_api_response"""
    for result in results:
        if result.get('from_name') == 'raw_api_response':
            try:
                text = result['value']['text']
                if isinstance(text, list):
                    text = text[0]
                if isinstance(text, str):
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        return text
                return text
            except Exception as e:
                return None
    return None

def evaluate_ground_truth_vs_predictions(tasks):
    """
    Evaluate ground truth vs predictions for all tasks in a project, grouped by model_version.
    Returns data structured for display_structured_output_analysis.
    """
    experimentset = {"experiments": []}
    model_to_observations = defaultdict(list)
    
    for task in tasks:
        task_id = task.get('id', 'Unknown')
        ground_truth = None
        
        # Extract ground truth
        for annotation in task.get('annotations', []):
            if annotation.get('ground_truth'):
                ground_truth = extract_raw_api_response(annotation.get('result', []))
                if ground_truth is None:
                    st.warning(f"Error parsing ground truth for task {task_id}")
        
        if ground_truth:
            # Extract and evaluate predictions
            for prediction in task.get('predictions', []):
                model_version = prediction.get('model_version', 'Unknown')
                pred_data = extract_raw_api_response(prediction.get('result', []))
                
                if pred_data:
                    field_scores = {}
                    errors = []
                    
                    # Compare fields
                    for field in ground_truth.keys():
                        gt_value = ground_truth.get(field)
                        pred_value = pred_data.get(field)
                        
                        # Simple exact match scoring (1 for match, 0 for mismatch or None)
                        if gt_value is not None and pred_value is not None:
                            score = 1.0 if gt_value == pred_value else 0.0
                            field_scores[field] = score
                        else:
                            field_scores[field] = 0.0
                            if gt_value is not None:
                                errors.append(f"Field {field} missing in prediction")
                    
                    # Calculate global score as average of field scores
                    global_score = np.mean(list(field_scores.values())) if field_scores else 0.0
                    
                    observation = {
                        "score": global_score,
                        "field_scores": field_scores
                    }
                    if errors:
                        observation["error"] = "; ".join(errors)
                    
                    obs_entry = {
                        "num_line": f"task_{task_id}_pred_{prediction.get('id', 'Unknown')}",
                        "observation": json.dumps(observation)
                    }
                    model_to_observations[model_version].append(obs_entry)
    
    # Create experiments for each model_version
    for model_version, observations in model_to_observations.items():
        experimentset["experiments"].append({
            "id": model_version,
            "name": model_version,
            "model": {"name": model_version, "aliased_name": model_version},
            "results": [{
                "metric_name": "llm_structured_output",
                "observation_table": observations
            }]
        })
    
    return experimentset

st.title("Label Studio Projects")
st.markdown("Browse and manage your Label Studio annotation projects.")

# Check if SDK is available
if not LABEL_STUDIO_AVAILABLE:
    st.error("❌ Label Studio SDK is not installed. Please install it using:")
    st.code("pip install label-studio-sdk", language="bash")
    st.stop()

# Connection settings in expander
with st.expander("⚙️ Connection Settings", expanded=False):
    st.markdown("""
    Configure your Label Studio connection using environment variables:
    - `LABEL_STUDIO_URL`: URL of your Label Studio instance (default: http://localhost:8080)
    - `LABEL_STUDIO_API_KEY`: Your Label Studio API key (required)
    """)
    
    current_url = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080")
    has_api_key = bool(os.getenv("LABEL_STUDIO_API_KEY"))
    
    st.info(f"**Current URL:** {current_url}")
    st.info(f"**API Key configured:** {'✅ Yes' if has_api_key else '❌ No'}")

# Get Label Studio client
client = get_label_studio_client()

if not client:
    st.stop()

# Add refresh button
if st.button("🔄 Refresh Projects"):
    st.rerun()

# Initialize session state for selected project
if 'selected_project_id' not in st.session_state:
    st.session_state.selected_project_id = None

# Back button if a project is selected
if st.session_state.selected_project_id is not None:
    if st.button("⬅️ Back to Projects"):
        st.session_state.selected_project_id = None
        st.rerun()

# Fetch projects
try:
    with st.spinner("Fetching projects from Label Studio..."):
        projects = client.list_projects()
    
    if not projects:
        st.info("No projects found in your Label Studio instance.")
    
    # If a project is selected, show its tasks and evaluation
    elif st.session_state.selected_project_id is not None:
        # Find the selected project
        selected_project = next(
            (p for p in projects if p.get_params()['id'] == st.session_state.selected_project_id),
            None
        )
        
        if not selected_project:
            st.error("Project not found")
            st.session_state.selected_project_id = None
        else:
            params = selected_project.get_params()
            st.header(f"📊 {params['title']}")
            st.markdown(f"**Project ID:** {params['id']} | **Total Tasks:** {params.get('task_number', 0)}")
            
            # Fetch tasks
            with st.spinner("Fetching tasks..."):
                tasks = selected_project.get_tasks()
            
            if not tasks:
                st.info("No tasks found in this project.")
            else:
                st.success(f"Found {len(tasks)} task(s)")
                
                # Perform evaluation
                experimentset = evaluate_ground_truth_vs_predictions(tasks)
                if experimentset["experiments"]:
                    display_structured_output_analysis(experimentset)
                else:
                    st.info("No valid ground truth or predictions found for evaluation.")
                
                # Display tasks
                for idx, task in enumerate(tasks, 1):
                    with st.expander(f"Task {idx} - ID: {task['id']}", expanded=(idx == 1)):
                        # Task data (input/ground truth)
                        st.subheader("📝 Task Data")
                        if task.get('data'):
                            st.json(task['data'])
                        
                        # Annotations
                        st.subheader("✅ Annotations")
                        annotations = task.get('annotations', [])
                        
                        if not annotations:
                            st.info("No annotations yet for this task.")
                        else:
                            st.markdown(f"**Total Annotations:** {len(annotations)}")
                            
                            for ann_idx, annotation in enumerate(annotations, 1):
                                with st.container():
                                    col1, col2 = st.columns([3, 1])
                                    
                                    with col1:
                                        st.markdown(f"**Annotation #{ann_idx}**")
                                        st.markdown(f"- **ID:** {annotation.get('id')}")
                                        st.markdown(f"- **Created:** {annotation.get('created_at', 'N/A')}")
                                        st.markdown(f"- **Completed by:** {annotation.get('completed_by', 'N/A')}")
                                    
                                    with col2:
                                        if annotation.get('was_cancelled'):
                                            st.error("❌ Cancelled")
                                        elif annotation.get('ground_truth'):
                                            st.success("⭐ Ground Truth")
                                    
                                    # Annotation result
                                    if annotation.get('result'):
                                        if st.button(f"Toggle Result #{ann_idx}", key=f"toggle_result_{task['id']}_{ann_idx}"):
                                            toggle_key = f"show_result_{task['id']}_{ann_idx}"
                                            st.session_state[toggle_key] = not st.session_state.get(toggle_key, False)
                                        
                                        if st.session_state.get(f"show_result_{task['id']}_{ann_idx}", False):
                                            st.json(annotation['result'])
                                    
                                    st.markdown("---")
                        
                        # Predictions
                        predictions = task.get('predictions', [])
                        if predictions:
                            st.subheader("🤖 Predictions")
                            for pred_idx, prediction in enumerate(predictions, 1):
                                if st.button(f"Toggle Prediction #{pred_idx}", key=f"toggle_pred_{task['id']}_{pred_idx}"):
                                    toggle_key = f"show_pred_{task['id']}_{pred_idx}"
                                    st.session_state[toggle_key] = not st.session_state.get(toggle_key, False)
                                
                                if st.session_state.get(f"show_pred_{task['id']}_{pred_idx}", False):
                                    st.json(prediction)
                        
                        st.divider()
    
    else:
        # Display projects list
        st.success(f"Found {len(projects)} project(s)")
        
        for project in projects:
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.subheader(f"📊 {project.get_params()['title']}")
                    
                    # Project details
                    params = project.get_params()
                    st.markdown(f"**ID:** {params['id']}")
                    
                    if params.get('description'):
                        st.markdown(f"**Description:** {params['description']}")
                    
                    # Task statistics
                    st.markdown(f"**Total Tasks:** {params.get('task_number', 0)}")
                    st.markdown(f"**Created:** {params.get('created_at', 'N/A')}")
                
                with col2:
                    # Action buttons
                    if st.button(f"View Tasks", key=f"view_tasks_{params['id']}"):
                        st.session_state.selected_project_id = params['id']
                        st.rerun()
                    
                    if st.button(f"Details", key=f"view_details_{params['id']}"):
                        st.session_state[f"show_details_{params['id']}"] = not st.session_state.get(f"show_details_{params['id']}", False)
                        st.rerun()
                
                # Show detailed information if requested
                if st.session_state.get(f"show_details_{params['id']}", False):
                    with st.expander(f"Project Details", expanded=True):
                        st.json(params)
                
                st.divider()

except Exception as e:
    st.error(f"Error fetching data: {str(e)}")
    st.exception(e)