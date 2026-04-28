Overview!

Backend (ML/Pipeline) + API: Arjun Prabhakaran<br>
Frontend: Abhinav Balaganesh<br>
Presentation/Slides: Satyanarayana Rudraraju<br>

**Analyzes user text and predicts mental health categories with probability scores.**<br>
Designed for early-stage screening and insight, not diagnosis.

**Categories**<br>

Anxiety • Depression • Stress • Suicidal • Bipolar • Personality Disorder • Normal <br>
(Normal category in order to avoid false positives and prove we can recognize when nothing is wrong).

```mermaid
graph TD
A[User Input] --> B[Extract Text from Image]
B --> C[Clean & Tokenize]
C --> D[Vectorize]
D --> E[ML Model]
E --> F[Predictions + Probabilities]
