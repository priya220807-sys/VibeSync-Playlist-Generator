# VibeSync - AI Playlist Generator 🎵

An interactive, localized web application designed to synthesize custom music playlists by evaluating multi-dimensional acoustic properties. 

## 📝 Abstract
The exponential growth of digital music streaming platforms has created a critical demand for intelligent recommendation systems capable of personalizing user experiences. This project introduces VibeSync, a local web-based interactive application designed to synthesize custom music playlists by evaluating multi-dimensional acoustic properties. Developed using the Streamlit framework, the system leverages a dataset containing Spotify-curated musical tracks categorized by complex audio signatures, including danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, and tempo. To achieve highly contextual recommendations, the architecture employs the Cosine Similarity metric to calculate the spatial orientation between an engineered user sound profile (or an anchor track vector) and the broader dataset matrix. Quantitative features are normalized utilizing a min-max scaling technique to guarantee geometric equilibrium across varying measurement units. Users can manually construct an acoustic fingerprint via dynamic UI controls or dynamically inherit the profile of an existing track. Experimental execution demonstrates that the mathematical modeling rapidly isolates top-tier matching configurations, producing high-fidelity playlist structures optimized for sonic continuity. This implementation proves the efficacy of localized, mathematical vector geometry models in resolving real-time content-based filtering criteria without requiring massive computational neural network overhead.

## 🚀 Key Features
* **Custom Acoustic Fingerprinting:** Adjust sliders for Danceability, Energy, Loudness, Speechiness, Acousticness, Valence, and Tempo to build a unique sound profile.
* **Anchor Track Integration:** Search and select an existing track to instantly pull its mathematical footprint as your baseline.
* **Vector Geometry Engine:** Uses `MinMaxScaler` and `cosine_similarity` to process and rank matching songs in real-time.
* **Dynamic Frontend:** Built completely with a responsive Streamlit UI layout.

## 📦 Installation & Setup
1. Clone this repository or open the project folder in your terminal.
2. Install the required dependencies:
   ```bash
   pip install streamlit pandas numpy scikit-learn
