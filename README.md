# Walsoft BI Program: Student Success Optimization

## 📖 Project Overview
This project analyzes student data from the Walsoft Computer Institute to determine how to improve admissions, academic support, and resource allocation. The goal is to maximize student success and program ROI using data-driven insights.

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn
- **Environment:** VS Code

## 📊 Key Findings
### 1. The "Effort vs. Aptitude" Discovery
- **Python Success:** Is primarily driven by **Effort**. `studyHOURS` accounts for 68% of the variance in final scores.
- **DB Success:** Is primarily driven by **Aptitude**. `entryEXAM` accounts for 60% of the variance in final scores.

### 2. Admissions Strategy
- **Entry Exams:** Are excellent predictors for Database Systems but less reliable for Programming potential.
- **Recommendation:** Retain exams but add a "Discipline Screening" question to predict study habits.

### 3. At-Risk Segments
- **Python Risk:** Students residing in **Sognsvann** show the lowest average study hours.
- **DB Risk:** Students with **High School** education background score lowest on entry exams (aptitude).

## 📈 Visualizations
This analysis generates the following key assets:
1. `correlation_heatmap.png` - Overall relationships between variables.
2. `feature_importance_ranking.png` - Random Forest model results showing top drivers.
3. `education_boxplot.png` - Performance distribution by education level.
4. `entry_vs_python_scatter.png` - Correlation between admission scores and final grades.

## 💡 Strategic Recommendations (The "One Intervention")
**Implement a 'Minimum Study Hour' Gamification System.**
- **Why:** Study hours are the #1 predictor of Python success.
- **How:** Weekly dashboards and rewards for students hitting >150 hours.
- **ROI:** A 10% increase in study hours correlates to an approximate **7.8% increase in Python scores**.

## 🚀 How to Run
1. Ensure you have the virtual environment installed.
2. Place the raw `bi.csv` file in the project directory.
3. Run the analysis script:
   ```bash
   python analysis.py
   ```
