# Fantasy Basketball Decision Support Tool
*A full analytics + decision support system for Yahoo Fantasy Basketball*

This project builds a complete fantasy basketball optimization engine using:
- **Python**
- **Streamlit UI**
- **Custom projection engine**
- **Win probability modeling**
- **Add/Drop streaming optimization**
- **Yahoo Fantasy API (OAuth2)**

It is designed for a **Yahoo points league** and helps answer:
- “What is my projected final score this week?”
- “Which players should I stream today, tomorrow, or the next day?”
- “What is my win probability vs my opponent?”
- “How do add/drops change my chance to win?”
- “Which teams offer the best schedule for streaming?”
The add drop evaluation is essential since the league structure limits adds to 75 per season and 5 per week.
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

├── app.py # Streamlit frontend
├── projection_core.py # Core projection + win probability engine
├── strategy_engine.py # Add/drop & streaming optimizer
├── baseline_projection.py # Batch projection script
├── src/ # Local data directory
│ └── data/ # Sample CSVs for demo
└── FBBK_Data_Pull_for_Github.ipynb # Clean Yahoo API notebook

---

## Sample Data

Includes sample CSVs here so the app runs as a demo:
- `my_team.csv`  
- `opp_team.csv`  
- `nba_schedule_next_7_days.csv`  
- `current_score.csv`

---

## Running Locally

```bash
git clone https://github.com/joshsing629/fantasy-basketball-decision-support.git
cd fantasy-basketball-decision-support
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py


