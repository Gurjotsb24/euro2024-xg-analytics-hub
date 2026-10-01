import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from statsbombpy import sb
from sklearn.linear_model import LogisticRegression
from mplsoccer import VerticalPitch
from dotenv import load_dotenv
from google import genai

# Set page configuration
st.set_page_config(page_title="Euro 2024 xG & xGOT Analytics Hub", layout="wide")

st.title("⚽ Euro 2024 Interactive xG, xGOT & Assists Analytics Hub")

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
# CACHED DATA PIPELINE & MODELS (xG + xGOT + Assists + Positions)
# ==============================================================================
@st.cache_data(show_spinner="Extracting Euro 2024 dataset from StatsBomb...")
def load_data_and_train_models():
    matches = sb.matches(competition_id=55, season_id=282)
    match_ids = matches['match_id'].tolist()

    all_shots = []
    all_passes = []
    all_positions = []
    
    progress_bar = st.progress(0, text="Fetching match events...")
    
    for idx, mid in enumerate(match_ids):
        events = sb.events(match_id=mid)
        
        # 1. Filter Shots
        shots = events[(events['type'] == 'Shot') & (events['shot_type'] != 'Penalty')].copy()
        all_shots.append(shots)
        
        # 2. Filter Passes
        passes = events[events['type'] == 'Pass'].copy()
        all_passes.append(passes)
        
        # 3. Filter Player Positions
        if 'player' in events.columns and 'position' in events.columns:
            pos_data = events[['player', 'position']].dropna()
            all_positions.append(pos_data)

        progress_bar.progress((idx + 1) / len(match_ids), text=f"Processing match {idx+1}/{len(match_ids)}...")
    
    progress_bar.empty()

    df_shots = pd.concat(all_shots, ignore_index=True)
    df_passes = pd.concat(all_passes, ignore_index=True)
    df_positions = pd.concat(all_positions, ignore_index=True) if all_positions else pd.DataFrame()

    # --- POSITION TAGGING LOGIC ---
    pos_acronym_map = {
        'Right Wing': 'RW', 'Left Wing': 'LW', 'Center Forward': 'ST',
        'Right Center Forward': 'ST', 'Left Center Forward': 'ST',
        'Attacking Midfield': 'CAM', 'Right Attacking Midfield': 'RAM', 'Left Attacking Midfield': 'LAM',
        'Central Midfield': 'CM', 'Right Midfield': 'RM', 'Left Midfield': 'LM',
        'Defensive Midfield': 'CDM', 'Right Defensive Midfield': 'CDM', 'Left Defensive Midfield': 'CDM',
        'Right Back': 'RB', 'Left Back': 'LB', 'Center Back': 'CB',
        'Right Center Back': 'CB', 'Left Center Back': 'CB', 'Goalkeeper': 'GK'
    }

    if not df_positions.empty:
        primary_positions = df_positions.groupby('player')['position'].agg(
            lambda x: x.mode()[0] if not x.empty else 'FP'
        ).to_dict()
    else:
        primary_positions = {}

    player_display_map = {}
    all_players = set(df_shots['player'].dropna().unique()).union(set(df_passes['player'].dropna().unique()))
    for p in all_players:
        full_pos = primary_positions.get(p, 'FP')
        tag = pos_acronym_map.get(full_pos, 'FP')
        player_display_map[p] = f"{p} ({tag})"

    # --- SHOTS PROCESSING & FEATURE ENGINEERING ---
    df_shots['x'] = df_shots['location'].apply(lambda loc: loc[0])
    df_shots['y'] = df_shots['location'].apply(lambda loc: loc[1])
    df_shots['distance'] = np.sqrt((120 - df_shots['x'])**2 + (40 - df_shots['y'])**2)
    df_shots['dist_post_A'] = np.sqrt((120 - df_shots['x'])**2 + (36 - df_shots['y'])**2)
    df_shots['dist_post_B'] = np.sqrt((120 - df_shots['x'])**2 + (44 - df_shots['y'])**2)
    cos_angle = (df_shots['dist_post_A']**2 + df_shots['dist_post_B']**2 - 8**2) / (2 * df_shots['dist_post_A'] * df_shots['dist_post_B'])
    df_shots['angle'] = np.arccos(np.clip(cos_angle, -1, 1))
    df_shots['is_header'] = np.where(df_shots['shot_body_part'] == 'Head', 1, 0)
    
    if 'under_pressure' in df_shots.columns:
        df_shots['under_pressure'] = df_shots['under_pressure'].fillna(False).astype(int)
    else:
        df_shots['under_pressure'] = 0

    df_shots['is_goal'] = np.where(df_shots['shot_outcome'] == 'Goal', 1, 0)

    model_data = df_shots[['player', 'team', 'distance', 'angle', 'is_header', 'under_pressure', 
                           'is_goal', 'x', 'y', 'shot_body_part', 'shot_end_location', 'shot_outcome']].dropna()

    # --- TRAIN PRE-SHOT xG MODEL ---
    X_xg = model_data[['distance', 'angle', 'is_header', 'under_pressure']]
    y_xg = model_data['is_goal']
    
    xg_model = LogisticRegression()
    xg_model.fit(X_xg, y_xg)
    model_data['custom_xg'] = xg_model.predict_proba(X_xg)[:, 1]

    # --- POST-SHOT xGOT MODEL ---
    model_data['y_end'] = model_data['shot_end_location'].apply(lambda loc: loc[1] if isinstance(loc, list) and len(loc) > 1 else np.nan)
    model_data['z_end'] = model_data['shot_end_location'].apply(lambda loc: loc[2] if isinstance(loc, list) and len(loc) > 2 else 0.0)

    on_target_outcomes = ['Goal', 'Saved', 'Saved to Post', 'Saved Off Target']
    model_data['is_on_target'] = model_data['shot_outcome'].isin(on_target_outcomes).astype(int)
    model_data['goalmouth_dist_center'] = np.sqrt((model_data['y_end'] - 40.0)**2 + (model_data['z_end'] - 0.0)**2)

    on_target_df = model_data[model_data['is_on_target'] == 1].dropna(subset=['goalmouth_dist_center'])
    X_xgot = on_target_df[['custom_xg', 'goalmouth_dist_center']]
    y_xgot = on_target_df['is_goal']

    xgot_model = LogisticRegression()
    xgot_model.fit(X_xgot, y_xgot)

    model_data['xgot'] = 0.0
    on_target_indices = on_target_df.index
    model_data.loc[on_target_indices, 'xgot'] = xgot_model.predict_proba(X_xgot)[:, 1]
    model_data['sga'] = model_data['xgot'] - model_data['custom_xg']

    # --- PASSES & ASSISTS PROCESSING (ROBUST FIX FOR STATSBOMB SCHEMA) ---
    df_passes['x'] = df_passes['location'].apply(lambda loc: loc[0] if isinstance(loc, list) else np.nan)
    df_passes['y'] = df_passes['location'].apply(lambda loc: loc[1] if isinstance(loc, list) else np.nan)
    df_passes['end_x'] = df_passes['pass_end_location'].apply(lambda loc: loc[0] if isinstance(loc, list) else np.nan)
    df_passes['end_y'] = df_passes['pass_end_location'].apply(lambda loc: loc[1] if isinstance(loc, list) else np.nan)

    # Detect Assists accurately across all StatsBomb column variants
    assist_col = None
    for candidate in ['pass_goal_assist', 'pass_goal_assisted']:
        if candidate in df_passes.columns:
            assist_col = candidate
            break

    if assist_col:
        df_passes['is_assist'] = df_passes[assist_col].isin([True, 'True', 1, 1.0])
    else:
        df_passes['is_assist'] = False

    # Key Passes / Shot Assistant Passes
    key_pass_col = None
    for candidate in ['pass_shot_assist', 'pass_shot_assistant']:
        if candidate in df_passes.columns:
            key_pass_col = candidate
            break

    if key_pass_col:
        df_passes['is_key_pass'] = df_passes[key_pass_col].isin([True, 'True', 1, 1.0]) & (~df_passes['is_assist'])
    elif 'pass_assisted_shot_id' in df_passes.columns:
        df_passes['is_key_pass'] = df_passes['pass_assisted_shot_id'].notna() & (~df_passes['is_assist'])
    else:
        df_passes['is_key_pass'] = False

    # Pass Completion
    if 'pass_outcome' in df_passes.columns:
        df_passes['is_complete'] = df_passes['pass_outcome'].isna()
    else:
        df_passes['is_complete'] = True

    passes_clean = df_passes[['player', 'team', 'x', 'y', 'end_x', 'end_y', 'is_assist', 'is_key_pass', 'is_complete']].dropna(subset=['x', 'y', 'player'])

    return model_data, passes_clean, player_display_map

model_data, passes_data, player_display_map = load_data_and_train_models()
reverse_player_map = {v: k for k, v in player_display_map.items()}

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
    if not df_non_goals.empty:
        pitch.scatter(df_non_goals.x, df_non_goals.y, s=df_non_goals.custom_xg * 1000, edgecolors='#b2b2b2', c='none', alpha=0.5, ax=ax, label='Miss/Save')
    
    df_goals = data[data['is_goal'] == 1]
    if not df_goals.empty:
        pitch.scatter(df_goals.x, df_goals.y, s=df_goals.custom_xg * 1000, edgecolors='#60f51d', c='#60f51d', ax=ax, label='Goal')

    ax.legend(loc='upper right', facecolor='#22312b', labelcolor='#efefef', fontsize=10)
    plt.title(f'{title_name} - Pre-Shot xG Map', color='#efefef', fontsize=14)
    return fig

def draw_goalmouth_placement(data, title_name):
    """Plots on-target shots on a 2D Goal Frame (Y-axis width vs Z-axis height)"""
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.set_facecolor('#22312b')
    ax.set_facecolor('#1e2923')

    ax.plot([36, 36, 44, 44], [0, 2.67, 2.67, 0], color='white', linewidth=4)
    ax.axhline(0, color='gray', linestyle='--')

    on_target = data[data['is_on_target'] == 1]
    goals = on_target[on_target['is_goal'] == 1]
    saved = on_target[on_target['is_goal'] == 0]

    if not saved.empty:
        ax.scatter(saved['y_end'], saved['z_end'], s=saved['xgot'] * 800 + 50, color='#ff4b4b', alpha=0.6, label='Saved/On-Target')
    if not goals.empty:
        ax.scatter(goals['y_end'], goals['z_end'], s=goals['xgot'] * 800 + 50, color='#60f51d', edgecolors='black', label='Goal')

    ax.set_xlim(34, 46)
    ax.set_ylim(-0.2, 3.2)
    ax.set_xlabel("Goal Width (36 = Left Post, 44 = Right Post)", color='white')
    ax.set_ylabel("Goal Height (Yards)", color='white')
    ax.tick_params(colors='white')
    ax.legend(loc='upper right', facecolor='#22312b', labelcolor='white')
    plt.title(f'{title_name} - Post-Shot Goalmouth Placement (xGOT)', color='white', fontsize=14)
    return fig

def draw_assist_map(pass_data, title_name):
    """Plots key passes (Cyan) and assists (Bright Green) distinctly on a vertical pitch."""
    pitch = VerticalPitch(pitch_type='statsbomb', half=True, pitch_color='#22312b', line_color='#efefef')
    fig, ax = pitch.draw(figsize=(8, 6))
    fig.set_facecolor('#22312b')

    key_passes = pass_data[pass_data['is_key_pass'] == True]
    assists = pass_data[pass_data['is_assist'] == True]

    # 1. Key Passes in Cyan
    if not key_passes.empty:
        pitch.arrows(
            key_passes.x, key_passes.y, key_passes.end_x, key_passes.end_y,
            color='#00d4ff', ax=ax, width=2, headwidth=4, alpha=0.8, label='Key Pass'
        )

    # 2. Assists in Bright Green (Larger Arrow Width)
    if not assists.empty:
        pitch.arrows(
            assists.x, assists.y, assists.end_x, assists.end_y,
            color='#60f51d', ax=ax, width=4, headwidth=6, alpha=1.0, label='Assist'
        )

    ax.legend(loc='upper left', facecolor='#22312b', labelcolor='#efefef', fontsize=10)
    plt.title(f'{title_name} - Assist & Shot Creation Map', color='#efefef', fontsize=14)
    return fig

# ==============================================================================
# NAVIGATION & MODES
# ==============================================================================
st.sidebar.header("Navigation")
mode = st.sidebar.radio("Select Analysis Mode:", ["1. Individual Player", "2. Team Analysis", "3. Player Comparison"])

# Sorted player choices formatted with positions (e.g., Lamine Yamal (RW))
all_unique_players = set(model_data['player'].unique()).union(set(passes_data['player'].unique()))
formatted_player_choices = sorted([player_display_map.get(p, f"{p} (FP)") for p in all_unique_players])

if mode == "1. Individual Player":
    st.header("👤 Individual Player Analytics (xG, xGOT & Playmaking)")
    selected_display = st.selectbox("Select or type player name (Position tagged):", formatted_player_choices)
    selected_player = reverse_player_map.get(selected_display, selected_display)

    if selected_player:
        player_data = model_data[model_data['player'] == selected_player]
        player_passes = passes_data[passes_data['player'] == selected_player]

        # Metric cards incorporating goals + assists
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Total Shots", len(player_data))
        col2.metric("Actual Goals", int(player_data['is_goal'].sum()) if not player_data.empty else 0)
        col3.metric("Expected Goals (xG)", f"{player_data['custom_xg'].sum():.2f}" if not player_data.empty else "0.00")
        col4.metric("Post-Shot xG (xGOT)", f"{player_data['xgot'].sum():.2f}" if not player_data.empty else "0.00")
        col5.metric("Assists", int(player_passes['is_assist'].sum()) if not player_passes.empty else 0)
        col6.metric("Key Passes", int(player_passes['is_key_pass'].sum()) if not player_passes.empty else 0)

        sga_val = player_data['sga'].sum() if not player_data.empty else 0.0

        map_col1, map_col2 = st.columns(2)
        with map_col1:
            st.pyplot(draw_shotmap(player_data, selected_display))
        with map_col2:
            st.pyplot(draw_assist_map(player_passes, selected_display))

        st.pyplot(draw_goalmouth_placement(player_data, selected_display))

        st.subheader("🤖 AI Performance & Playmaking Evaluation")
        if st.button("Generate Tactical Scouting Report"):
            prompt = f"""
            Act as an elite football analyst evaluating {selected_display} at Euro 2024:
            - Pre-Shot xG: {player_data['custom_xg'].sum():.2f}
            - Post-Shot xGOT: {player_data['xgot'].sum():.2f}
            - Actual Goals: {player_data['is_goal'].sum()}
            - Shooting Goals Added (SGA): {sga_val:+.2f}
            - Assists: {player_passes['is_assist'].sum()}
            - Key Passes / Shot Creations: {player_passes['is_key_pass'].sum()}
            - Pass Completion Rate: {(player_passes['is_complete'].sum()/len(player_passes)*100 if len(player_passes)>0 else 0):.1f}%
            
            Write a 2-paragraph evaluation comparing their goalscoring threat and finishing quality against their creative playmaking and assist threat.
            """
            st.info(get_ai_summary(prompt))

elif mode == "2. Team Analysis":
    st.header("🛡️ Team Performance & Playmaking Analytics")
    all_teams = sorted(set(model_data['team'].dropna().unique()).union(set(passes_data['team'].dropna().unique())))
    selected_team = st.selectbox("Select Team:", all_teams)

    if selected_team:
        team_data = model_data[model_data['team'] == selected_team]
        team_passes = passes_data[passes_data['team'] == selected_team]

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Shots", len(team_data))
        col2.metric("Actual Goals", int(team_data['is_goal'].sum()) if not team_data.empty else 0)
        col3.metric("Cumulative xG", f"{team_data['custom_xg'].sum():.2f}" if not team_data.empty else "0.00")
        col4.metric("Assists Recorded", int(team_passes['is_assist'].sum()) if not team_passes.empty else 0)
        col5.metric("Key Passes Created", int(team_passes['is_key_pass'].sum()) if not team_passes.empty else 0)

        map_col1, map_col2 = st.columns(2)
        with map_col1:
            st.pyplot(draw_shotmap(team_data, selected_team))
        with map_col2:
            st.pyplot(draw_assist_map(team_passes, selected_team))

        st.subheader("🤖 AI Team Finishing & Playmaking Overview")
        if st.button("Generate Team AI Analysis"):
            prompt = f"""
            Analyze team shot execution and creative output for {selected_team} at Euro 2024:
            - Cumulative xG: {team_data['custom_xg'].sum():.2f}, xGOT: {team_data['xgot'].sum():.2f}, Goals: {team_data['is_goal'].sum()}
            - Team Assists: {team_passes['is_assist'].sum()}, Key Passes: {team_passes['is_key_pass'].sum()}
            Provide a 2-paragraph overview evaluating both shot conversion and chance creation efficiency.
            """
            st.info(get_ai_summary(prompt))

elif mode == "3. Player Comparison":
    st.header("⚔ Creative & Finishing Quality Comparison")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        p1_display = st.selectbox("Select First Player:", formatted_player_choices, index=0)
        p1 = reverse_player_map.get(p1_display, p1_display)
    with col_p2:
        p2_display = st.selectbox("Select Second Player:", formatted_player_choices, index=min(1, len(formatted_player_choices)-1))
        p2 = reverse_player_map.get(p2_display, p2_display)

    if p1 and p2:
        data_p1, pass_p1 = model_data[model_data['player'] == p1], passes_data[passes_data['player'] == p1]
        data_p2, pass_p2 = model_data[model_data['player'] == p2], passes_data[passes_data['player'] == p2]

        comp_df = pd.DataFrame({
            "Metric": [
                "Goals Scored", "Assists", "Key Passes (Shot Creates)", 
                "Pre-Shot xG", "Post-Shot xGOT", "Shooting Goals Added (SGA)"
            ],
            p1_display: [
                int(data_p1['is_goal'].sum()) if not data_p1.empty else 0,
                int(pass_p1['is_assist'].sum()) if not pass_p1.empty else 0,
                int(pass_p1['is_key_pass'].sum()) if not pass_p1.empty else 0,
                f"{data_p1['custom_xg'].sum():.2f}" if not data_p1.empty else "0.00",
                f"{data_p1['xgot'].sum():.2f}" if not data_p1.empty else "0.00",
                f"{data_p1['sga'].sum():+.2f}" if not data_p1.empty else "+0.00"
            ],
            p2_display: [
                int(data_p2['is_goal'].sum()) if not data_p2.empty else 0,
                int(pass_p2['is_assist'].sum()) if not pass_p2.empty else 0,
                int(pass_p2['is_key_pass'].sum()) if not pass_p2.empty else 0,
                f"{data_p2['custom_xg'].sum():.2f}" if not data_p2.empty else "0.00",
                f"{data_p2['xgot'].sum():.2f}" if not data_p2.empty else "0.00",
                f"{data_p2['sga'].sum():+.2f}" if not data_p2.empty else "+0.00"
            ]
        })
        st.table(comp_df.set_index("Metric"))

        map_col1, map_col2 = st.columns(2)
        with map_col1:
            st.pyplot(draw_assist_map(pass_p1, p1_display))
        with map_col2:
            st.pyplot(draw_assist_map(pass_p2, p2_display))

        st.subheader("🤖 AI Scouting Comparison")
        if st.button("Generate Tactical Comparison Report"):
            prompt = f"""
            Compare the finishing execution and playmaking threat of two players at Euro 2024:
            - {p1_display}: Goals = {data_p1['is_goal'].sum() if not data_p1.empty else 0}, xG = {data_p1['custom_xg'].sum():.2f} if not data_p1.empty else 0, xGOT = {data_p1['xgot'].sum():.2f} if not data_p1.empty else 0, Assists = {pass_p1['is_assist'].sum() if not pass_p1.empty else 0}, Key Passes = {pass_p1['is_key_pass'].sum() if not pass_p1.empty else 0}
            - {p2_display}: Goals = {data_p2['is_goal'].sum() if not data_p2.empty else 0}, xG = {data_p2['custom_xg'].sum():.2f} if not data_p2.empty else 0, xGOT = {data_p2['xgot'].sum():.2f} if not data_p2.empty else 0, Assists = {pass_p2['is_assist'].sum() if not pass_p2.empty else 0}, Key Passes = {pass_p2['is_key_pass'].sum() if not pass_p2.empty else 0}
            Write a 2-paragraph scouting report identifying who was more complete across finishing clinicality and playmaking creation.
            """
            st.info(get_ai_summary(prompt))