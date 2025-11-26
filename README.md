# Fantasy Basketball Decision Support Tool
*A full analytics + decision support system for Yahoo Fantasy Basketball*

**Live Demo:** https://fantasy-basketball-decision-support-exrbxxpu2z6v3d4aahdxni.streamlit.app/

This project builds a complete fantasy basketball optimization engine using:
- **Baseline fantasy point projections**
- **Win probability modeling**
- **Daily matchup simulation**
- **Add/Drop streaming optimization**
- **Opponent counter-move simulation**
- **What-if scenarios (injuries, added games)**
The app runs locally using demo CSVs (included in this repo), so it works without needing Yahoo API credentials.

It is designed for a **Yahoo points league** and helps answer:
- “What is my projected final score this week?”
- “Which players should I stream today, tomorrow, or the next day?”
- “What is my win probability vs my opponent?”
- “How do add/drops change my chance to win?”
- “Which teams offer the best schedule for streaming?”
- “What changes if specific players are injured?”
  
**The add/drop strategy evaluation is essential since the league structure limits adds to 75 per season and 5 per week.**
---

## Key Features

### **Projection Engine**
- Player-level rest-of-week fantasy point simulation  
- Max 10 active players per day (Yahoo rule)  
- Per-game “DNP” override selection  
- Per-week “player out” toggle  
- Custom logistic win probability model  

### **Streaming Optimization Engine**
- Simulate 0–5 add/drops  
- Identify optimal streaming **days**  
- Recommend **teams** to pull streamers from  
- Recommend **which players to drop** (Avg FP < 28 rule)  
- Model opponent add/drop strategy as well  

### **Yahoo Data Pull Notebook**
- Included clean version: `FBBK_Data_Pull_for_Github.ipynb`
- Shows:
  - OAuth2 setup
  - Team roster extraction
  - Schedule extraction
  - Player projections

---

## Project Structure

fantasy-basketball-decision-support/
│
├── app.py                         # Streamlit UI
├── baseline_projection.py         # Baseline weekly projections
├── projection_core.py             # Core modeling logic
├── strategy_engine.py             # Add/drop optimization engine
├── requirements.txt
├── README.md
├── FBBK_Data_Pull_for_Github.ipynb
│
├── src/
│   ├── core/                      # Utility modules
│   │   ├── config.py
│   │   ├── constants.py
│   │   └── __init__.py
│   │
│   ├── data/                      # Demo CSVs (safe for GitHub)
│   │   ├── my_team.csv
│   │   ├── opp_team.csv
│   │   ├── current_score.csv
│   │   ├── nba_schedule_next_7_days.csv
│   │   └── README.md              # Info about demo data
│   │
│   └── yahoo_api.py               # Empty placeholder (no secrets)
│
└── .gitignore


---

## Running Locally

```bash
git clone https://github.com/joshsing629/fantasy-basketball-decision-support.git
cd fantasy-basketball-decision-support
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py

---

## Sample Data for Demo Included

Includes sample CSVs here so the app runs as a demo:
- `my_team.csv`  
- `opp_team.csv`  
- `nba_schedule_next_7_days.csv`  
- `current_score.csv`

## Yahoo API Credentials

Real Yahoo OAuth credentials are intentionally NOT included. If you want to pull real data:
- create a Yahoo developer app
- add your CLIENT_ID to oauth2.json (and CLIENT_SECRET if necessary)
- Run: python src/data/yahoo_api.py
(Instructions provided inside the notebook.)



