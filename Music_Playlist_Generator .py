import streamlit as st
import pandas as pd
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import os
st.set_page_config(page_title="VibeSync-Music Playlist Generator",page_icon="🎵",layout="wide")
st.title("VibeSync - AI Playlist Generator")
st.markdown("Generate personalized track lists using **Acoustic Profile Vector Geometry**")
DATASET_PATH = "dataset.csv/dataset.csv"
@st.cache_data
def load_data():
    if not os.path.exists(DATASET_PATH):
        return None
    df=pd.read_csv(DATASET_PATH)
    df=df.dropna(subset=['track_name','artists'])
    
    df=df.drop_duplicates(subset=['track_name','artists'])
    return df
df=load_data()
if df is None:
    st.error(f"s Could not locate the dataset at: {DATASET_PATH}")
    st.info("Please make sure your 'dataset.csv' file is inside the 'dataset.csv' folder as shown in your sidebar.")

else:
    st.sidebar.header("Customize Your Sound Profile")
    st.sidebar.write("Manually adjust the slider parameters to construct a unique acoustic fingerprint.")
    danceability=st.sidebar.slider("Danceability (Trmpo rhythm structure)",0.0,1.0,0.6)
    energy = st.sidebar.slider("Energy (Intensity & fast-paced activity)", 0.0, 1.0, 0.7)
    loudness = st.sidebar.slider("Loudness (Overall dB range)", -40.0, 5.0, -7.0)
    speechiness = st.sidebar.slider("Speechiness (Presence of spoken words)", 0.0, 1.0, 0.1)
    acousticness = st.sidebar.slider("Acousticness (Traditional instruments vs synth)", 0.0, 1.0, 0.2)
    instrumentalness = st.sidebar.slider("Instrumentalness (Likelihood of no vocals)", 0.0, 1.0, 0.0)
    liveness = st.sidebar.slider("Liveness (Audience presence/live track status)", 0.0, 1.0, 0.1)
    valence = st.sidebar.slider("Valence (Positiveness / happy mood profile)", 0.0, 1.0, 0.5)
    tempo = st.sidebar.slider("Tempo (Beats per minute - BPM)", 50.0, 220.0, 120.0)
    st.subheader(" Option A: Anchor with a Song you Love")
    all_songs = (df['track_name'] + " - " + df['artists']).tolist()
    selected_song = st.selectbox("Search and pick an anchor track to copy its acoustic profile instantly:", ["None"] + all_songs[:1000])

    
    feature_cols = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    
    if selected_song != "None":
        track_title = selected_song.split(" - ")[0]
        artist_name = selected_song.split(" - ")[1]
        matched_row = df[(df['track_name'] == track_title) & (df['artists'] == artist_name)].iloc[0]
        user_vector = matched_row[feature_cols].values.reshape(1, -1)
        st.success(f" Successfully locked onto anchor profile for: **{track_title}**")
    else:
        
        user_vector = np.array([danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo]).reshape(1, -1)

   
    if st.button("Synthesize Custom Playlist"):
        with st.spinner("Calculating Cosine Spatial Vectors across 114,000 tracks..."):
            
            scaler = MinMaxScaler()
            scaled_dataset = scaler.fit_transform(df[feature_cols])
            scaled_user_vector = scaler.transform(user_vector)
            
            
            similarity_matrix = cosine_similarity(scaled_user_vector, scaled_dataset)[0]
            df['Match Score'] = similarity_matrix
            
           
            results = df.sort_values(by='Match Score', ascending=False).head(6)
            
           
            if selected_song != "None":
                results = results.iloc[1:]
            else:
                results = results.head(5)
                
            st.subheader(" Generated Playlist Matrix")
            
            
            
            for idx, row in results.iterrows():
                score_pct = int(row['Match Score'] * 100)
                with st.container():
                    col1, col2 = st.columns([1, 4])
                    col1.metric("Match", f"{score_pct}%")
                    col2.markdown(f"### 🎵 {row['track_name']}")
                    col2.markdown(f"**Artist:** {row['artists']} | **Album:** {row['album_name']} | **Genre:** `{row['track_genre']}`")
                    st.divider()