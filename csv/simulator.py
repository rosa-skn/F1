import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.express as px
import plotly.graph_objects as go
import math

st.set_page_config(page_title="F1 Race Simulator", layout="wide")
st.title("Formula 1 Race Simulator")


@st.cache_data
def load_data():
    try:
        results = pd.read_csv("results.csv")
        drivers = pd.read_csv("drivers.csv")
        constructors = pd.read_csv("constructors.csv")
        races = pd.read_csv("races.csv")
        qualifying = pd.read_csv("qualifying.csv")
        if "points" in results.columns:
            results["points"] = pd.to_numeric(results["points"], errors="coerce").fillna(0)
        if "positionOrder" in results.columns:
            results["positionOrder"] = pd.to_numeric(results["positionOrder"], errors="coerce")
        for df in (results, qualifying, races):
            if "raceId" in df.columns:
                df["raceId"] = pd.to_numeric(df["raceId"], errors="coerce")
        for df in (results, qualifying, drivers):
            if "driverId" in df.columns:
                df["driverId"] = pd.to_numeric(df["driverId"], errors="coerce")
        for df in (results, qualifying, constructors):
            if "constructorId" in df.columns:
                df["constructorId"] = pd.to_numeric(df["constructorId"], errors="coerce")
        return results, drivers, constructors, races, qualifying
    except Exception as e:
        st.error(f"Error loading CSV(s): {e}")
        return None, None, None, None, None


results_df, drivers_df, constructors_df, races_df, qualifying_df = load_data()

if any(df is None for df in (results_df, drivers_df, constructors_df, races_df, qualifying_df)):
    st.stop()

TEAM_COLORS = {
    "Ferrari": "#E10600",
    "Mercedes": "#00D2BE",
    "Red Bull": "#1E41FF",
    "McLaren": "#FF8700",
    "Aston Martin": "#006F62",
    "Alpine": "#2293D1",
    "Williams": "#005AFF",
    "AlphaTauri": "#2B4562",
    "Alfa Romeo": "#900000",
    "Haas": "#B6BABD"
}

F1_2025_CALENDAR = [
    (1, "Australian Grand Prix", "Australia"),
    (2, "Chinese Grand Prix", "China"),
    (3, "Japanese Grand Prix", "Japan"),
    (4, "Bahrain Grand Prix", "Bahrain"),
    (5, "Saudi Arabian Grand Prix", "Saudi Arabia"),
    (6, "Miami Grand Prix", "USA"),
    (7, "Emilia Romagna Grand Prix", "Italy"),
    (8, "Monaco Grand Prix", "Monaco"),
    (9, "Spanish Grand Prix", "Spain"),
    (10, "Canadian Grand Prix", "Canada"),
    (11, "Austrian Grand Prix", "Austria"),
    (12, "British Grand Prix", "UK"),
    (13, "Hungarian Grand Prix", "Hungary"),
    (14, "Belgian Grand Prix", "Belgium"),
    (15, "Dutch Grand Prix", "Netherlands"),
    (16, "Italian Grand Prix", "Italy"),
    (17, "Azerbaijan Grand Prix", "Azerbaijan"),
    (18, "Singapore Grand Prix", "Singapore"),
    (19, "United States Grand Prix", "USA"),
    (20, "Mexican Grand Prix", "Mexico"),
    (21, "Brazilian Grand Prix", "Brazil"),
    (22, "Las Vegas Grand Prix", "USA"),
    (23, "Qatar Grand Prix", "Qatar"),
    (24, "Abu Dhabi Grand Prix", "UAE")
]


def build_provisional_grid(raceId, top_n=20):
    try:

        if 'raceId' in results_df.columns and 'year' in races_df.columns:
            merged = results_df.merge(races_df[['raceId', 'year']], on='raceId', how='left')
            years = merged['year'].dropna().unique()
            if len(years) > 0:
                latest_year = int(np.max(years))

                recent_races = races_df[races_df['year'] == latest_year]
                recent_results = results_df[results_df['raceId'].isin(recent_races['raceId'])]
                if not recent_results.empty and 'points' in recent_results.columns:
                    pts = recent_results.groupby('driverId')['points'].sum().reset_index().sort_values('points',
                                                                                                       ascending=False)
                    pts = pts.head(top_n)

                    cons = recent_results.groupby(['driverId', 'constructorId']).size().reset_index(name='cnt')
                    cons = cons.sort_values(['driverId', 'cnt'], ascending=[True, False]).drop_duplicates('driverId')

                    pts = pts.merge(drivers_df[['driverId', 'forename', 'surname']], on='driverId', how='left')
                    pts = pts.merge(cons[['driverId', 'constructorId']], on='driverId', how='left')
                    pts = pts.merge(constructors_df[['constructorId', 'name']], on='constructorId', how='left')
                    df = pd.DataFrame({
                        'raceId': raceId,
                        'driverId': pts['driverId'],
                        'constructorId': pts['constructorId'].fillna(constructors_df['constructorId'].iloc[0]).astype(
                            int),
                        'position': np.arange(1, len(pts) + 1),
                        'grid': np.arange(1, len(pts) + 1),
                        'full_name': pts['forename'].fillna('') + ' ' + pts['surname'].fillna(''),
                        'name': pts['name'].fillna('Unknown')
                    })
                    return df.reset_index(drop=True)
    except Exception:
        pass

    fallback = drivers_df.head(top_n).copy()
    default_constructor = constructors_df['constructorId'].iloc[0] if not constructors_df.empty else np.nan
    constructor_ids = constructors_df['constructorId'].dropna().astype(
        int).tolist() if 'constructorId' in constructors_df.columns else [default_constructor]
    fallback['constructorId'] = np.random.choice(constructor_ids, size=len(fallback), replace=True)
    fallback['raceId'] = raceId
    fallback['position'] = np.arange(1, len(fallback) + 1)
    fallback['grid'] = fallback['position']
    fallback['full_name'] = fallback['forename'].fillna('') + ' ' + fallback['surname'].fillna('')
    fallback = fallback.merge(constructors_df[['constructorId', 'name']], on='constructorId', how='left')
    fallback['name'] = fallback['name'].fillna('Unknown')
    return fallback[['raceId', 'driverId', 'constructorId', 'position', 'grid', 'full_name', 'name']].reset_index(
        drop=True)


def simulate_from_grid(grid_df, chaos_factor=0.3, weather_impact='Dry', safety_cars=1):
    grid_df = grid_df.copy().reset_index(drop=True)
    if 'grid' not in grid_df.columns:
        grid_df['grid'] = np.arange(1, len(grid_df) + 1)

    n = len(grid_df)
    np.random.seed(random.randint(1, 99999))

    base = grid_df['grid'].astype(float).values

    skill = np.ones(n)

    weather_effects = {"Dry": 1.0, "Light Rain": 1.2, "Heavy Rain": 1.8, "Mixed": 1.4}
    wmul = weather_effects.get(weather_impact, 1.0)

    chaos_var = np.random.normal(0, chaos_factor * 2.5, n)
    weather_var = np.random.normal(0, (wmul - 1) * 2.0, n)

    safety_var = np.zeros(n)
    for _ in range(safety_cars):
        idx = np.random.choice(n, size=max(1, n // 3), replace=False)
        safety_var[idx] += np.random.normal(0, 1.8, size=len(idx))

    final_score = base * skill + chaos_var + weather_var + safety_var
    order_idx = np.argsort(final_score)
    finishing_pos = np.empty_like(order_idx)
    finishing_pos[order_idx] = np.arange(1, n + 1)

    res = grid_df.copy()
    res['simulated_position'] = finishing_pos
    res['grid_change'] = res['grid'] - res['simulated_position']
    res = res.sort_values('simulated_position').reset_index(drop=True)

    pts = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
    res['points'] = [pts[i] if i < len(pts) else 0 for i in range(len(res))]
    return res


def predict_2026(chaos_factor=0.3, weather_impact='Dry', safety_cars=1, top_n=5):
    if 'raceId' not in results_df.columns or 'raceId' not in races_df.columns:
        return None, None
    merged = results_df.merge(races_df[['raceId', 'year']], on='raceId', how='left')
    years = merged['year'].dropna().unique()
    if len(years) == 0:
        return None, None
    years_sorted = sorted(years)
    recent_years = years_sorted[-3:]
    recent_races = races_df[races_df['year'].isin(recent_years)]
    recent_results = results_df[results_df['raceId'].isin(recent_races['raceId'])]
    if recent_results.empty:
        return None, None

    driver_pts = recent_results.groupby('driverId')['points'].sum().reset_index()
    driver_pts = driver_pts.merge(drivers_df[['driverId', 'forename', 'surname']], on='driverId', how='left')
    driver_pts['Driver'] = driver_pts['forename'].fillna('') + ' ' + driver_pts['surname'].fillna('')

    cons_freq = recent_results.groupby(['driverId', 'constructorId']).size().reset_index(name='cnt')
    cons_freq = cons_freq.sort_values(['driverId', 'cnt'], ascending=[True, False]).drop_duplicates('driverId')
    driver_pts = driver_pts.merge(cons_freq[['driverId', 'constructorId']], on='driverId', how='left')
    driver_pts = driver_pts.merge(constructors_df[['constructorId', 'name']], on='constructorId', how='left')
    driver_pts['Team'] = driver_pts['name'].fillna('Unknown')

    team_pts = recent_results.groupby('constructorId')['points'].sum().reset_index()
    team_pts = team_pts.merge(constructors_df[['constructorId', 'name']], on='constructorId', how='left')

    chaos_adj = max(0.25, 1 - chaos_factor * 0.45)
    weather_map = {"Dry": 1.0, "Light Rain": 0.96, "Heavy Rain": 0.9, "Mixed": 0.93}
    weather_adj = weather_map.get(weather_impact, 1.0)
    safety_adj = max(0.6, 1 - safety_cars * 0.05)
    total_adj = chaos_adj * weather_adj * safety_adj

    driver_pts['adj_points'] = driver_pts['points'] * total_adj
    team_pts['adj_points'] = team_pts['points'] * total_adj

    top_drivers = driver_pts.sort_values('adj_points', ascending=False).head(top_n).reset_index(drop=True)
    top_teams = team_pts.sort_values('adj_points', ascending=False).head(top_n).reset_index(drop=True)

    return top_drivers, top_teams


years_available = sorted(races_df['year'].dropna().unique().tolist(), reverse=True)
if 2025 not in years_available:
    years_available.insert(0, 2025)
if 2026 not in years_available:
    years_available.insert(0, 2026)

selected_year = st.sidebar.selectbox("Select Season", options=years_available, index=0)

if selected_year == 2025:
    race_options = [f"{name} (Round {rnd}) — {country}" for (rnd, name, country) in F1_2025_CALENDAR]
elif selected_year == 2026:
    race_options = ["2026 Season Projection"]
else:
    year_races = races_df[races_df['year'] == selected_year].sort_values('round')
    race_options = [f"{r['name']} (Round {int(r['round'])})" for _, r in
                    year_races.iterrows()] if not year_races.empty else ["No races available"]

selected_race_display = st.sidebar.selectbox("Select Race", options=race_options)

chaos_factor = st.sidebar.slider("Chaos Factor ", 0.0, 1.0, 0.3, 0.05)
weather_impact = st.sidebar.selectbox("Weather", ["Dry", "Light Rain", "Heavy Rain", "Mixed"])
safety_cars = st.sidebar.slider("Expected number of safety car periods ", 0, 3, 1)
simulate_button = st.sidebar.button("Start Simulation")

if simulate_button:
    try:
        if selected_year == 2026:
            top_drivers, top_teams = predict_2026(chaos_factor, weather_impact, safety_cars, top_n=6)
            if top_drivers is None or top_drivers.empty:
                st.warning("Not enough historical result data to compute 2026 projection.")
            else:
                winner_driver = top_drivers.iloc[0]
                winner_team = top_teams.iloc[0] if (top_teams is not None and not top_teams.empty) else None
                st.subheader("2026 Projection (dynamic)")
                st.markdown(
                    f"**Top projected driver (2026):** {winner_driver['Driver']} — adjusted points: **{int(winner_driver['adj_points'])}**")
                if winner_team is not None:
                    st.markdown(
                        f"**Top projected team (2026):** {winner_team['name']} — adjusted points: **{int(winner_team['adj_points'])}**")

                figd = px.bar(top_drivers, x='Driver', y='adj_points', color='Team',
                              color_discrete_map=TEAM_COLORS, template="plotly_white",
                              title="Top drivers for 2026 (adjusted)")
                figd.update_layout(font=dict(color="black"), yaxis_title="Adjusted points", xaxis_tickangle=-45,
                                   legend=dict(title='Team', font=dict(color='black')))
                figd.update_traces(marker_line_color='black', marker_line_width=1)

                figd.update_traces(hoverlabel=dict(font=dict(color='black'), bgcolor='white'))
                st.plotly_chart(figd, use_container_width=True)

                figt = px.bar(top_teams, x='name', y='adj_points', color='name',
                              color_discrete_map=TEAM_COLORS, template="plotly_white",
                              title="Top teams for 2026 (adjusted)")
                figt.update_layout(showlegend=False, font=dict(color="black"), yaxis_title="Adjusted points")
                figt.update_traces(marker_line_color='black', marker_line_width=1,
                                   hoverlabel=dict(font=dict(color='black'), bgcolor='white'))
                st.plotly_chart(figt, use_container_width=True)

        else:

            if selected_year == 2025:

                try:
                    selected_round = int(selected_race_display.split("Round ")[1].split(")")[0])
                except Exception:
                    selected_round = 1

                candidate = races_df[(races_df['year'] == 2025) & (races_df['round'] == selected_round)]
                if not candidate.empty:
                    race_row = candidate.iloc[0]
                    raceId = int(race_row['raceId'])
                else:
                    raceId = 90000 + selected_round
            else:

                try:
                    parts = selected_race_display.split(" (Round ")
                    name = parts[0].strip()
                    rnd = int(parts[1].split(")")[0])
                    candidate = races_df[
                        (races_df['year'] == selected_year) & (races_df['round'] == rnd) & (races_df['name'] == name)]
                    if candidate.empty:
                        candidate = races_df[(races_df['year'] == selected_year) & (races_df['round'] == rnd)]
                    race_row = candidate.iloc[0]
                    raceId = int(race_row['raceId'])
                except Exception:
                    st.error("Could not determine raceId for the selected race.")
                    raceId = None

            if raceId is None:
                st.stop()

            qdf = qualifying_df[
                qualifying_df['raceId'] == raceId] if 'raceId' in qualifying_df.columns else pd.DataFrame()
            if qdf is None or qdf.empty:
                grid_df = build_provisional_grid(raceId, top_n=20)
                st.info("No qualifying found - building provisional grid from latest standings / drivers.")
            else:

                pos_col = None
                for candidate_col in ['position', 'grid', 'positionOrder', 'pos']:
                    if candidate_col in qdf.columns:
                        pos_col = candidate_col
                        break
                if pos_col is None:
                    qdf = qdf.reset_index(drop=True)
                    qdf['position'] = np.arange(1, len(qdf) + 1)
                    pos_col = 'position'
                grid_df = qdf[['raceId', 'driverId', 'position']].copy()
                if 'constructorId' not in grid_df.columns:
                    tmp = results_df[['driverId', 'constructorId']].drop_duplicates()
                    grid_df = grid_df.merge(tmp, on='driverId', how='left')
                grid_df = pd.merge(grid_df, drivers_df[['driverId', 'forename', 'surname']], on='driverId', how='left')
                grid_df = pd.merge(grid_df, constructors_df[['constructorId', 'name']], on='constructorId', how='left')
                grid_df['full_name'] = grid_df['forename'].fillna('') + ' ' + grid_df['surname'].fillna('')
                grid_df['grid'] = pd.to_numeric(grid_df[pos_col], errors='coerce')
                if grid_df['grid'].isna().all():
                    grid_df['grid'] = np.arange(1, len(grid_df) + 1)

            sim_results = simulate_from_grid(grid_df, chaos_factor=chaos_factor, weather_impact=weather_impact,
                                             safety_cars=safety_cars)
            if sim_results is None or sim_results.empty:
                st.warning("Simulation produced no results.")
            else:
                disp = sim_results[['simulated_position', 'full_name', 'name', 'grid', 'grid_change', 'points']].copy()
                disp.columns = ['Position', 'Driver', 'Team', 'Grid', 'Change', 'Points']

                disp['Change'] = disp['Change'].apply(lambda x: f"+{int(x)}" if x > 0 else str(int(x)))


                def style_rows(row):
                    pos = int(row['Position'])
                    if pos == 1:
                        return ['background-color: #E10600; color: white; font-weight: bold'] * len(row)
                    elif pos == 2:
                        return ['background-color: #D3D3D3; color: black'] * len(row)
                    elif pos == 3:
                        return ['background-color: #CD7F32; color: white'] * len(row)
                    elif pos <= 10:
                        return ['background-color: #E8F4FA; color: black'] * len(row)
                    else:
                        return ['background-color: white; color: black'] * len(row)


                st.subheader("Simulated Race Results")
                st.dataframe(disp.style.apply(style_rows, axis=1), use_container_width=True, hide_index=True)

                winner = sim_results.iloc[0]
                c1, c2, c3 = st.columns(3)
                c1.metric("Winner", winner['full_name'])
                c2.metric("From Grid", int(winner['grid']) if not math.isnan(winner['grid']) else "N/A")
                c3.metric("Positions Gained", int(winner['grid_change']))

                st.subheader("Grid vs Finishing Position")
                fig = px.scatter(sim_results, x="grid", y="simulated_position", color="name",
                                 hover_data=["full_name", "name", "grid", "simulated_position"],
                                 labels={"grid": "Grid Position", "simulated_position": "Finishing Position"},
                                 template="plotly_white", color_discrete_map=TEAM_COLORS)
                max_pos = int(max(sim_results['grid'].max(), sim_results['simulated_position'].max()))
                fig.add_trace(go.Scatter(x=[1, max_pos], y=[1, max_pos], mode="lines", name="No Change",
                                         line=dict(dash="dash", color="black", width=1.5)))
                fig.update_traces(marker=dict(size=12, line=dict(width=1, color="black")))
                fig.update_layout(
                    yaxis=dict(autorange="reversed", tickfont=dict(color="black")),
                    xaxis=dict(autorange="reversed", tickfont=dict(color="black")),
                    font=dict(size=13, color="black"),
                    legend=dict(font=dict(color="black")),
                    hoverlabel=dict(font=dict(color="black"), bgcolor="white"),
                    plot_bgcolor="white", paper_bgcolor="white"
                )
                st.plotly_chart(fig, use_container_width=True)

                if selected_year == 2025:
                    rrow = races_df[(races_df['year'] == 2025) & (races_df['round'] == selected_round)]
                    if not rrow.empty and 'country' in rrow.columns:
                        country = rrow.iloc[0].get('country', '')
                        st.info(f"Selected 2025 race country: {country}")
    except Exception as exc:
        st.error(f"An error occurred during simulation: {exc}")
        st.stop()

st.markdown("---")
st.caption("F1 Race Simulator — simulation + dynamic 2026 projection")
