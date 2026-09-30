# ⚽ Euro 2024 Interactive xG & xGOT Analytics Hub

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B2?style=for-the-badge&logo=google%20gemini&logoColor=white)

An end-to-end Machine Learning, Spatial Visualization, and Generative AI dashboard designed to analyze shot performance, goal probability, and finishing efficiency at Euro 2024 using StatsBomb data.

---

## 🌟 Key Features

- **Pre-Shot xG Modeling:** Predicts goal probability based on pre-shot spatial geometry (distance, angle) and situational context (headers, defensive pressure) using Logistic Regression.
- **Post-Shot xGOT Modeling:** Measures shot execution and placement quality on target using goalmouth coordinates ($Y$-frame width and $Z$-frame height).
- **Shooting Goals Added (SGA):** Evaluates finishing execution using $\text{SGA} = \text{xGOT} - \text{xG}$ to separate clinical placement from poor finishing.
- **3 Dynamic Analytics Modes:**
  1. **Individual Player Search:** Instant spatial shot maps, 2D goalmouth placement charts, and finishing metrics.
  2. **Team Analysis:** Aggregated attacking metrics and team-wide goal probability heatmaps.
  3. **Head-to-Head Comparison:** Comparative metric tables and side-by-side placement comparisons.
- **AI Tactical Analyst:** Powered by **Gemini 3.5-flash** via `google-genai` to generate real-time tactical and finishing scouting reports.

---

## 🛠️ Tech Stack

- **Frontend & Web App:** Streamlit
- **Machine Learning:** Scikit-Learn (Logistic Regression)
- **Data Source:** StatsBombPy (Euro 2024 open dataset)
- **Data Manipulation:** Pandas, NumPy
- **Spatial Visualizations:** `mplsoccer`, Matplotlib
- **Generative AI:** Google GenAI SDK (`google-genai` / Gemini 3.5-flash)
- **Environment Management:** `python-dotenv`

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Google Gemini API Key ([Get an API key here](https://aistudio.google.com/))

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Gurjotsb24/euro2024-xg-analytics-hub.git](https://github.com/Gurjotsb24/euro2024-xg-analytics-hub.git)
   cd euro2024-xg-analytics-hub

2. Create and activate a virtual environment:
   # On Windows (PowerShell/Command Prompt)
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

3. Install required dependencies:
   pip install -r requirements.txt

4. Configure Environment Variables:
Create a .env file in the root project directory (you can copy .env.example) and add your Gemini API Key:
  GEMINI_API_KEY=your_actual_gemini_api_key_here

5. Launch the Streamlit Dashboard:
  streamlit run app.py

  
