import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("⚽ EPL Match Analyzer")
st.write("A simple web app to view team performance and predict results!")

# --- Load dataset ---
data = {
    'HomeTeam': ['Arsenal', 'Chelsea', 'Liverpool', 'Arsenal', 'Chelsea', 'Liverpool'],
    'AwayTeam': ['Chelsea', 'Arsenal', 'Arsenal', 'Liverpool', 'Liverpool', 'Chelsea'],
    'FullTimeHomeGoals': [2, 1, 0, 3, 2, 1],
    'FullTimeAwayGoals': [1, 2, 2, 1, 1, 3]
}

df = pd.DataFrame(data)
st.subheader("Match Data")
st.dataframe(df)

# --- Select team to analyze ---
team = st.selectbox("Choose a team to view its performance:", df['HomeTeam'].unique())

# --- Calculate goals for selected team ---
df["TeamGoals"] = df.apply(
    lambda x: x["FullTimeHomeGoals"] if x["HomeTeam"] == team else (
        x["FullTimeAwayGoals"] if x["AwayTeam"] == team else 0
    ), axis=1
)

# --- Plot dotted line graph ---
st.subheader(f"{team} Goals per Match")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(len(df)), df["TeamGoals"], linestyle=':', marker='o', color='red')
ax.set_title(f"{team} Goals Trend")
ax.set_xlabel("Match Number")
ax.set_ylabel("Goals")
ax.grid(True, linestyle='--', alpha=0.5)
st.pyplot(fig)

# --- Prediction (basic example) ---
st.subheader("🏆 Simple Match Prediction")
home_team = st.selectbox("Select Home Team", df["HomeTeam"].unique())
away_team = st.selectbox("Select Away Team", df["AwayTeam"].unique())
home_shots = st.slider("Home Shots", 0, 20, 10)
away_shots = st.slider("Away Shots", 0, 20, 10)

if st.button("Predict Winner"):
    if home_shots > away_shots:
        st.success(f"Predicted Winner: {home_team}")
    elif away_shots > home_shots:
        st.success(f"Predicted Winner: {away_team}")
    else:
        st.info("It's likely to be a Draw!")
