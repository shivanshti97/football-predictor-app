
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import io

st.set_page_config(page_title="EPL Analyzer", layout="wide")

# Helper functions

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is None:
        # default small example dataset (if user doesn't upload)
        data = {
            'MatchDate': ["2023-08-01","2023-08-10","2023-08-20","2023-09-01","2023-09-10","2023-09-20"],
            'HomeTeam': ['Arsenal','Chelsea','Liverpool','Arsenal','Chelsea','Liverpool'],
            'AwayTeam': ['Chelsea','Arsenal','Arsenal','Liverpool','Liverpool','Chelsea'],
            'FullTimeHomeGoals': [2,1,0,3,2,1],
            'FullTimeAwayGoals': [1,2,2,1,1,3],
            'HomeShots': [12,9,8,14,11,10],
            'AwayShots': [7,10,11,9,8,12],
            'HomeShotsOnTarget':[6,4,3,7,5,4],
            'AwayShotsOnTarget':[3,5,6,4,3,7],
            'HomeFouls':[8,10,9,7,11,8],
            'AwayFouls':[12,9,11,8,7,10],
            'HomeYellowCards':[1,2,2,1,2,1],
            'AwayYellowCards':[2,1,1,2,1,3],
            'HomeRedCards':[0,0,0,0,0,0],
            'AwayRedCards':[0,0,0,0,0,0]
        }
        df = pd.DataFrame(data)
    else:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('utf-8')))
    # normalize column names (strip)
    df.columns = df.columns.str.strip()
    # ensure MatchDate is datetime if exists
    if "MatchDate" in df.columns:
        try:
            df["MatchDate"] = pd.to_datetime(df["MatchDate"])
        except Exception:
            pass
    return df

def add_result_columns(df):
    # create result label (H/D/A) and numeric mapping for simple modelling
    if "FullTimeHomeGoals" in df.columns and "FullTimeAwayGoals" in df.columns:
        df = df.copy()
        df['ResultLabel'] = df.apply(
            lambda x: 'H' if x['FullTimeHomeGoals'] > x['FullTimeAwayGoals']
            else ('A' if x['FullTimeAwayGoals'] > x['FullTimeHomeGoals'] else 'D'),
            axis=1
        )
        df['ResultNum'] = df['ResultLabel'].map({'H':1, 'D':0, 'A':-1})
    return df

def team_total_goals(df):
    home_goals = df.groupby('HomeTeam')['FullTimeHomeGoals'].sum()
    away_goals = df.groupby('AwayTeam')['FullTimeAwayGoals'].sum()
    total_goals = home_goals.add(away_goals, fill_value=0).astype(int).sort_values(ascending=False)
    return total_goals

def team_wins(df):
    df = df.copy()
    df['Winner'] = df.apply(lambda x: x['HomeTeam'] if x['FullTimeHomeGoals'] > x['FullTimeAwayGoals']
                            else (x['AwayTeam'] if x['FullTimeAwayGoals'] > x['FullTimeHomeGoals'] else 'Draw'), axis=1)
    wins = df[df['Winner'] != 'Draw']['Winner'].value_counts().sort_values(ascending=False)
    return wins

def discipline_score(df):
    # penalty score = fouls + 2*yellows + 5*reds (lower = more disciplined)
    home_fouls = df.groupby('HomeTeam')['HomeFouls'].sum()
    away_fouls = df.groupby('AwayTeam')['AwayFouls'].sum()
    total_fouls = home_fouls.add(away_fouls, fill_value=0)

    home_y = df.groupby('HomeTeam')['HomeYellowCards'].sum()
    away_y = df.groupby('AwayTeam')['AwayYellowCards'].sum()
    total_y = home_y.add(away_y, fill_value=0)

    home_r = df.groupby('HomeTeam')['HomeRedCards'].sum() if 'HomeRedCards' in df.columns else pd.Series(dtype=int)
    away_r = df.groupby('AwayTeam')['AwayRedCards'].sum() if 'AwayRedCards' in df.columns else pd.Series(dtype=int)
    total_r = home_r.add(away_r, fill_value=0)

    penalty_score = total_fouls + 2*total_y + 5*total_r
    return penalty_score.sort_values()

def team_accuracy(df):
    # accuracy = goals / shots * 100
    # guard missing shot columns
    if 'HomeShots' in df.columns and 'AwayShots' in df.columns:
        home_goals = df.groupby('HomeTeam')['FullTimeHomeGoals'].sum()
        away_goals = df.groupby('AwayTeam')['FullTimeAwayGoals'].sum()
        total_goals = home_goals.add(away_goals, fill_value=0)

        home_shots = df.groupby('HomeTeam')['HomeShots'].sum()
        away_shots = df.groupby('AwayTeam')['AwayShots'].sum()
        total_shots = home_shots.add(away_shots, fill_value=0)

        accuracy = (total_goals / total_shots) * 100
        return accuracy.fillna(0).sort_values(ascending=False)
    return pd.Series(dtype=float)


# UI - Sidebar
st.sidebar.title("EPL Analyzer")
st.sidebar.write("Upload your CSV (optional) or use example data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
df = load_data(uploaded_file)
df = add_result_columns(df)

st.sidebar.markdown("---")
st.sidebar.write("Model & Display options")
model_train = st.sidebar.checkbox("Train simple win-predictor model", value=False)
show_details = st.sidebar.checkbox("Show raw data", value=False)


# Main page
st.title("⚽ EPL Match Analyzer — Advanced Student Project")
st.markdown("Interactive analysis, team stats, and a simple win predictor. "
            "Upload your full matches CSV or use the sample data.")

# show raw dataframe
if show_details:
    st.subheader("Match Dataset")
    st.dataframe(df)

# Quick stats
st.subheader("Quick Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Matches", df.shape[0])
col2.metric("Teams (approx)", len(pd.unique(df[['HomeTeam','AwayTeam']].values.ravel())))
col3.metric("Has Shots?", "HomeShots" in df.columns)
col4.metric("Has Cards?", "HomeYellowCards" in df.columns)

# Team selector 
teams = sorted(pd.unique(df[['HomeTeam','AwayTeam']].values.ravel()))
team_choice = st.selectbox("Select team to analyze", options=teams, index=0)

st.subheader(f"{team_choice} — Key Stats")
# Goals over matches (time series)
team_matches = df[(df['HomeTeam'] == team_choice) | (df['AwayTeam'] == team_choice)].copy()
team_matches = team_matches.sort_values(by='MatchDate') if 'MatchDate' in team_matches.columns else team_matches.reset_index(drop=True)

team_matches['TeamGoals'] = team_matches.apply(
    lambda x: x['FullTimeHomeGoals'] if x['HomeTeam']==team_choice else (
        x['FullTimeAwayGoals'] if x['AwayTeam']==team_choice else 0), axis=1)

# plot dotted line
fig1, ax1 = plt.subplots(figsize=(8,3))
ax1.plot(range(len(team_matches)), team_matches['TeamGoals'], linestyle=':', marker='o', color='tab:red')
ax1.set_title(f"{team_choice} Goals per Match")
ax1.set_xlabel("Match number")
ax1.set_ylabel("Goals")
ax1.set_xticks(range(len(team_matches)))
ax1.set_xticklabels(team_matches['MatchDate'].dt.strftime('%Y-%m-%d') if 'MatchDate' in team_matches.columns else range(len(team_matches)), rotation=45)
ax1.grid(alpha=0.4)
st.pyplot(fig1)

# Rolling average (smoothing)
if len(team_matches) >= 3:
    rolling = team_matches['TeamGoals'].rolling(window=3, min_periods=1).mean()
    figr, axr = plt.subplots(figsize=(8,3))
    axr.plot(range(len(team_matches)), rolling, linestyle='-', marker='s', color='tab:blue')
    axr.set_title(f"{team_choice} 3-match Rolling Average Goals")
    st.pyplot(figr)

# wins and goals summary
st.markdown("**Team Summary**")
wins = team_wins(df)
goals = team_total_goals(df)
accuracy = team_accuracy(df)
penalty = discipline_score(df)

cols = st.columns(3)
with cols[0]:
    st.write("Total Wins (top)")
    st.bar_chart(wins)
with cols[1]:
    st.write("Total Goals (top)")
    st.bar_chart(goals)
with cols[2]:
    st.write("Scoring Accuracy (%) (top)")
    st.bar_chart(accuracy)

st.subheader("Discipline (lower score = better)")
st.write(penalty.head(10))


# Simple Win Predictor (basic model)
st.subheader("Simple Win Predictor (match-level)")

st.write("This is a simple example model that tries to predict result using shots stats. It's a student-level baseline — not production-ready.")
# Prepare data for model automatically if requested
if model_train:
    if not all(col in df.columns for col in ["HomeShots","AwayShots","HomeShotsOnTarget","AwayShotsOnTarget","ResultNum"]):
        st.warning("Dataset missing required columns for model training. Please upload data with shot columns and full-time goals.")
    else:
        # prepare dataset X,y
        df_model = df.dropna(subset=["ResultNum"])
        X = df_model[["HomeShots","AwayShots","HomeShotsOnTarget","AwayShotsOnTarget"]].fillna(0)
        y = df_model["ResultNum"]

        # split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=None)
        model = RandomForestClassifier(n_estimators=100, random_state=1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        st.success(f"Trained RandomForest model — test accuracy: {acc:.2f}")

        # feature importance
        fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
        figfi, axfi = plt.subplots(figsize=(6,3))
        fi.plot(kind='barh', ax=axfi)
        axfi.set_title("Feature Importance")
        st.pyplot(figfi)

        # User can input match stats to predict
        st.markdown("### Try a prediction")
        c1, c2 = st.columns(2)
        with c1:
            h_team = st.selectbox("Home team", teams, index=teams.index(team_choice))
            home_sh = st.number_input("Home shots", min_value=0, value=10)
            home_sot = st.number_input("Home shots on target", min_value=0, value=5)
        with c2:
            a_team = st.selectbox("Away team", teams, index=0)
            away_sh = st.number_input("Away shots", min_value=0, value=8)
            away_sot = st.number_input("Away shots on target", min_value=0, value=3)

        if st.button("Predict result for this match"):
            newX = pd.DataFrame({
                "HomeShots":[home_sh],
                "AwayShots":[away_sh],
                "HomeShotsOnTarget":[home_sot],
                "AwayShotsOnTarget":[away_sot]
            })
            pred = model.predict(newX)[0]
            prob = model.predict_proba(newX) if hasattr(model, "predict_proba") else None
            if pred == 1:
                st.success(f"Model predicts: Home team ({h_team}) likely to WIN")
            elif pred == 0:
                st.info("Model predicts: Draw")
            else:
                st.warning(f"Model predicts: Away team ({a_team}) likely to WIN")
            if prob is not None:
                st.write("Class probabilities (H, D, A): ", prob.round(2).tolist())

st.markdown("---")
st.write("Tips: Upload a full matches CSV (with columns like HomeTeam, AwayTeam, FullTimeHomeGoals, FullTimeAwayGoals, HomeShots, AwayShots, HomeShotsOnTarget, AwayShotsOnTarget, HomeFouls, AwayFouls, HomeYellowCards, AwayYellowCards).")
st.write("This app is a learning project — to improve it, add more features (xG, injuries, lineup) and more advanced models.")
