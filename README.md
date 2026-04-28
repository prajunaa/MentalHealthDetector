Overview!

Backend (ML/Pipeline) + API: Arjun Prabhakaran
Frontend: Abhinav Balaganesh
Presentation/Slides: Satyanarayana Rudraraju

Analyzes user text and predicts mental health categories with probability scores.
Designed for early-stage screening and insight, not diagnosis.

Categories
Anxiety • Depression • Stress • Suicidal • Bipolar • Personality Disorder • Normal (In order to avoid false positives and prove we can recognize when nothing is wrong).

```mermaid
graph TD
A[User Input] --> B[Clean & Tokenize]
B --> C[Vectorize]
C --> D[ML Model]
D --> E[Predictions + Probabilities]
