import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from statsbombpy import sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from mplsoccer import VerticalPitch
from dotenv import load_dotenv
from google import genai

# Set page configuration
st.set_page_config(page_title="Euro 2024 xG & xGOT Analytics Hub", layout="wide")

st.title("⚽ Euro 2024 Interactive xG & xGOT Analytics Hub")

# ==============================================================================
# SECURE API KEY SETUP
# ==============================================================================
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

@st.cache_resource
def init_genai_client(key):
    if key:
        return genai.Client(api_key=key)
    return None

client = init_genai_client(api_key)

# ==============================================================================
# CACHED DATA PIPELINE & MODELS (xG + xGOT)
# ==============================================================================
@st.cache_data(show_spinner="Extracting Euro 2024 dataset from StatsBomb...")
def load_data_and_train_models():
    matches = sb.matches(competition_id=55, season_id=282)
    match_ids = matches['match_id'].tolist()

    all_shots = []
    progress_bar = st.progress(0, text="Fetching match events...")
    
    for idx, mid in enumerate(match_ids):
        events = sb.events(match_id=mid)
        shots = events[(events['type'] == 'Shot') & (events['shot_type'] != 'Penalty')].copy()
        all_shots.append(shots)
        progress_bar.progress((idx + 1) / len(match_ids), text=f"Processing match {idx+1}/{len(match_ids)}...")
    
    progress_bar.empty()

    df = pd.concat(all_shots, ignore_index=True)

    # 1. Base Pre-Shot Features
    df['x'] = df['location'].apply(lambda loc: loc[0])
    df['y'] = df['location'].apply(lambda loc: loc[1])
    df['distance'] = np.sqrt((120 - df['x'])**2 + (40 - df['y'])**2)
    df['dist_post_A'] = np.sqrt((120 - df['x'])**2 + (36 - df['y'])**2)
    df['dist_post_B'] = np.sqrt((120 - df['x'])**2 + (44 - df['y'])**2)
    cos_angle = (df['dist_post_A']**2 + df['dist_post_B']**2 - 8**2) / (2 * df['dist_post_A'] * df['dist_post_B'])
    df['angle'] = np.arccos(np.clip(cos_angle, -1, 1))
    df['is_header'] = np.where(df['shot_body_part'] == 'Head', 1, 0)
    
    if 'under_pressure' in df.columns:
        df['under_pressure'] = df['under_pressure'].fillna(False).astype(int)
    else:
        df['under_pressure'] = 0

    df['is_goal'] = np.where(df['shot_outcome'] == 'Goal', 1, 0)

    # Clean dataset
    model_data = df[['player', 'team', 'distance', 'angle', 'is_header', 'under_pressure', 
                           'is_goal', 'x', 'y', 'shot_body_part', 'shot_end_location', 'shot_outcome']].dropna()

    # --- TRAIN PRE-SHOT xG MODEL ---
    X_xg = model_data[['distance', 'angle', 'is_header', 'under_pressure']]
    y_xg = model_data['is_goal']
    
    xg_model = LogisticRegression()
    xg_model.fit(X_xg, y_xg)
    model_data['custom_xg'] = xg_model.predict_proba(X_xg)[:, 1]

    # --- FEATURE ENGINEERING FOR POST-SHOT xGOT ---
    # Extract end coordinates (y_end: goal width 36-44, z_end: goal height 0-2.67)
    model_data['y_end'] = model_data['shot_end_location'].apply(lambda loc: loc[1] if isinstance(loc, list) and len(loc) > 1 else np.nan)
    model_data['z_end'] = model_data['shot_end_location'].apply(lambda loc: loc[2] if isinstance(loc, list) and len(loc) > 2 else 0.0)

    # Identify shots on target (Goals and Saved shots inside the goal frame)
    on_target_outcomes = ['Goal', 'Saved', 'Saved to Post', 'Saved Off Target']
    model_data['is_on_target'] = model_data['shot_outcome'].isin(on_target_outcomes).astype(int)

    # Goalmouth distance from center (y=40.0, z=0.0)
    model_data['goalmouth_dist_center'] = np.sqrt((model_data['y_end'] - 40.0)**2 + (model_data['z_end'] - 0.0)**2)

    # --- TRAIN POST-SHOT xGOT MODEL ---
    # Model only trains on actual shots on target
    on_target_df = model_data[model_data['is_on_target'] == 1].dropna(subset=['goalmouth_dist_center'])
    
    X_xgot = on_target_df[['custom_xg', 'goalmouth_dist_center']]
    y_xgot = on_target_df['is_goal']

    xgot_model = LogisticRegression()
    xgot_model.fit(X_xgot, y_xgot)

    # Calculate xGOT: On-target shots get probability; off-target shots get 0.0
    model_data['xgot'] = 0.0
    on_target_indices = on_target_df.index
    model_data.loc[on_target_indices, 'xgot'] = xgot_model.predict_proba(X_xgot)[:, 1]

    # Shooting Goals Added (SGA = xGOT - xG)
    model_data['sga'] = model_data['xgot'] - model_data['custom_xg']

    return model_data

model_data = load_data_and_train_models()

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def get_ai_summary(prompt):
    if not client:
        return "⚠️ Gemini API key not detected in `.env`. AI analysis disabled."
    try:
        response = client.models.generate_content(
            model='gemini-3.5-flash',
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        return f"AI Generation Error: {e}"

def draw_shotmap(data, title_name):
    pitch = VerticalPitch(pitch_type='statsbomb', half=True, goal_type='box', pitch_color='#22312b', line_color='#efefef')
    fig, ax = pitch.draw(figsize=(8, 6))
    fig.set_facecolor('#22312b')

    df_non_goals = data[data['is_goal'] == 0]
    pitch.scatter(df_non_goals.x, df_non_goals.y, s=df_non_goals.custom_xg * 1000, edgecolors='#b2b2b2', c='none', alpha=0.5, ax=ax, label='Miss/Save')
    
    df_goals = data[data['is_goal'] == 1]
    pitch.scatter(df_goals.x, df_goals.y, s=df_goals.custom_xg * 1000, edgecolors='#60f51d', c='#60f51d', ax=ax, label='Goal')

    ax.legend(loc='upper right', facecolor='#22312b', labelcolor='#efefef', fontsize=10)
    plt.title(f'{title_name} - Pre-Shot xG Map', color='#efefef', fontsize=14)
    return fig

def draw_goalmouth_placement(data, title_name):
    """Plots on-target shots on a 2D Goal Frame (Y-axis width vs Z-axis height)"""
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.set_facecolor('#22312b')
    ax.set_facecolor('#1e2923')

    # Draw Goal Frame (Width: 36 to 44, Height: 0 to 2.67)
    ax.plot([36, 36, 44, 44], [0, 2.67, 2.67, 0], color='white', linewidth=4)
    ax.axhline(0, color='gray', linestyle='--') # Ground line

    on_target = data[data['is_on_target'] == 1]
    
    # Scatter plot of shot placement sized by xGOT
    goals = on_target[on_target['is_goal'] == 1]
    saved = on_target[on_target['is_goal'] == 0]

    ax.scatter(saved['y_end'], saved['z_end'], s=saved['xgot'] * 800 + 50, color='#ff4b4b', alpha=0.6, label='Saved/On-Target')
    ax.scatter(goals['y_end'], goals['z_end'], s=goals['xgot'] * 800 + 50, color='#60f51d', edgecolors='black', label='Goal')

    ax.set_xlim(34, 46)
    ax.set_ylim(-0.2, 3.2)
    ax.set_xlabel("Goal Width (36 = Left Post, 44 = Right Post)", color='white')
    ax.set_ylabel("Goal Height (Yards)", color='white')
    ax.tick_params(colors='white')
    ax.legend(loc='upper right', facecolor='#22312b', labelcolor='white')
    plt.title(f'{title_name} - Post-Shot Goalmouth Placement (xGOT)', color='white', fontsize=14)
    return fig

# ==============================================================================
# NAVIGATION & MODES
# ==============================================================================
st.sidebar.header("Navigation")
mode = st.sidebar.radio("Select Analysis Mode:", ["1. Individual Player", "2. Team Analysis", "3. Player Comparison"])

if mode == "1. Individual Player":
    st.header("👤 Individual Player Analytics (xG vs xGOT)")
    all_players = sorted(model_data['player'].unique())
    selected_player = st.selectbox("Select or type player name:", all_players)

    if selected_player:
        player_data = model_data[model_data['player'] == selected_player]

        # Metric cards incorporating xGOT and SGA
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Shots", len(player_data))
        col2.metric("Actual Goals", int(player_data['is_goal'].sum()))
        col3.metric("Expected Goals (xG)", f"{player_data['custom_xg'].sum():.2f}")
        col4.metric("Post-Shot xG (xGOT)", f"{player_data['xgot'].sum():.2f}")
        
        sga_val = player_data['sga'].sum()
        col5.metric("Shooting Goals Added", f"{sga_val:+.2f}", delta_color="normal" if sga_val >= 0 else "inverse")

        map_col1, map_col2 = st.columns(2)
        with map_col1:
            st.pyplot(draw_shotmap(player_data, selected_player))
        with map_col2:
            st.pyplot(draw_goalmouth_placement(player_data, selected_player))

        st.subheader("🤖 AI Finishing Evaluation")
        if st.button("Generate Finishing Breakdown"):
            prompt = f"""
            Act as an elite finishing coach evaluating {selected_player} at Euro 2024:
            - Pre-Shot xG: {player_data['custom_xg'].sum():.2f}
            - Post-Shot xGOT: {player_data['xgot'].sum():.2f}
            - Actual Goals: {player_data['is_goal'].sum()}
            - Shooting Goals Added (xGOT - xG): {sga_val:+.2f}
            
            Write a 2-paragraph evaluation comparing their chance quality vs their finishing execution (did their shot placement increase or decrease their chances of scoring?).
            """
            st.info(get_ai_summary(prompt))

elif mode == "2. Team Analysis":
    st.header("🛡️ Team Performance Analytics (xG vs xGOT)")
    all_teams = sorted(model_data['team'].unique())
    selected_team = st.selectbox("Select Team:", all_teams)

    if selected_team:
        team_data = model_data[model_data['team'] == selected_team]

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Shots", len(team_data))
        col2.metric("Actual Goals", int(team_data['is_goal'].sum()))
        col3.metric("Cumulative xG", f"{team_data['custom_xg'].sum():.2f}")
        col4.metric("Cumulative xGOT", f"{team_data['xgot'].sum():.2f}")
        
        team_sga = team_data['sga'].sum()
        col5.metric("Team SGA", f"{team_sga:+.2f}")

        map_col1, map_col2 = st.columns(2)
        with map_col1:
            st.pyplot(draw_shotmap(team_data, selected_team))
        with map_col2:
            st.pyplot(draw_goalmouth_placement(team_data, selected_team))

        st.subheader("🤖 AI Team Finishing Analysis")
        if st.button("Generate Team AI Analysis"):
            prompt = f"""
            Analyze team shot execution for {selected_team} at Euro 2024:
            - Pre-Shot xG: {team_data['custom_xg'].sum():.2f}, Post-Shot xGOT: {team_data['xgot'].sum():.2f}, Goals: {team_data['is_goal'].sum()}
            Provide a 2-paragraph overview evaluating whether the team's strikers enhanced chance quality through superior shot placement.
            """
            st.info(get_ai_summary(prompt))

elif mode == "3. Player Comparison":
    st.header("⚔️ Finishing Quality Comparison (xGOT & SGA)")
    all_players = sorted(model_data['player'].unique())
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        p1 = st.selectbox("Select First Player:", all_players, index=0)
    with col_p2:
        p2 = st.selectbox("Select Second Player:", all_players, index=min(1, len(all_players)-1))

    if p1 and p2:
        data_p1 = model_data[model_data['player'] == p1]
        data_p2 = model_data[model_data['player'] == p2]

        comp_df = pd.DataFrame({
            "Metric": ["Total Shots", "Goals Scored", "Pre-Shot xG", "Post-Shot xGOT", "Shooting Goals Added (SGA)"],
            p1: [len(data_p1), int(data_p1['is_goal'].sum()), f"{data_p1['custom_xg'].sum():.2f}", f"{data_p1['xgot'].sum():.2f}", f"{data_p1['sga'].sum():+.2f}"],
            p2: [len(data_p2), int(data_p2['is_goal'].sum()), f"{data_p2['custom_xg'].sum():.2f}", f"{data_p2['xgot'].sum():.2f}", f"{data_p2['sga'].sum():+.2f}"]
        })
        st.table(comp_df.set_index("Metric"))

        map_col1, map_col2 = st.columns(2)
        with map_col1:
            st.pyplot(draw_goalmouth_placement(data_p1, p1))
        with map_col2:
            st.pyplot(draw_goalmouth_placement(data_p2, p2))

        st.subheader("🤖 AI Finishing Scouting Comparison")
        if st.button("Generate Finishing Scouting Report"):
            prompt = f"""
            Compare the finishing execution of two players at Euro 2024:
            - {p1}: Pre-Shot xG = {data_p1['custom_xg'].sum():.2f}, Post-Shot xGOT = {data_p1['xgot'].sum():.2f}, SGA = {data_p1['sga'].sum():+.2f}
            - {p2}: Pre-Shot xG = {data_p2['custom_xg'].sum():.2f}, Post-Shot xGOT = {data_p2['xgot'].sum():.2f}, SGA = {data_p2['sga'].sum():+.2f}
            Write a 2-paragraph scouting report identifying who was more clinical at altering chance quality through shot placement.
            """
            st.info(get_ai_summary(prompt))