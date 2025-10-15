from flask import Flask, render_template, request, session
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import os
import pdfkit

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Use a strong, random key in production

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))  # corrected filename

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        study_time = float(request.form['study_time'])
        attendance = float(request.form['attendance'])
        past_scores = float(request.form['past_scores'])
        extra_classes = float(request.form['extra_classes'])

        # Input validation
        errors = []
        if study_time < 0:
            errors.append("Study Hours cannot be negative.")
        if attendance < 0 or attendance > 100:
            errors.append("Attendance must be between 0 and 100.")
        if past_scores < 0:
            errors.append("Previous Score cannot be negative.")
        if extra_classes < 0:
            errors.append("Extra Classes cannot be negative.")

        if errors:
            return render_template('result.html',
                                   prediction_text="Input Error",
                                   comment="<br>".join(errors))

        input_data = pd.DataFrame(
            [[study_time, attendance, past_scores, extra_classes]],
            columns=['Study Hours', 'Attendance', 'Previous Score', 'Extra Classes']
        )

        prediction = model.predict(input_data)[0]

        if prediction >= 75:
            comment = "Good 👍"
        elif prediction >= 50:
            comment = "Average 😐"
        else:
            comment = "Bad 👎"

        # Data for chart
        features = ['Study Hours', 'Attendance', 'Previous Score', 'Extra Classes']
        values = [study_time, attendance, past_scores, extra_classes]

        # Create bar chart
        plt.figure(figsize=(6,4))
        bars = plt.bar(features, values, color=['#3949ab', '#2193b0', '#1a237e', '#263238'])
        plt.title('Your Inputs')
        plt.ylabel('Value')
        plt.tight_layout()

        chart_path = os.path.join('static', 'input_chart.png')
        plt.savefig(chart_path)
        plt.close()

        history_item = {
            'study_time': study_time,
            'attendance': attendance,
            'past_scores': past_scores,
            'extra_classes': extra_classes,
            'prediction': f"{prediction:.2f}",
            'comment': comment
        }

        if 'history' not in session:
            session['history'] = []
        session['history'].append(history_item)
        session.modified = True

        return render_template('result.html',
                               prediction_text=f"Predicted Score: {prediction:.2f}",
                               comment=comment,
                               study_time=study_time,
                               attendance=attendance,
                               past_scores=past_scores,
                               extra_classes=extra_classes,
                               chart_url=chart_path)
    except Exception as e:
        return render_template('result.html',
                               prediction_text="Error",
                               comment=str(e))

@app.route('/download', methods=['POST'])
def download():
    study_time = request.form['study_time']
    attendance = request.form['attendance']
    past_scores = request.form['past_scores']
    extra_classes = request.form['extra_classes']
    prediction = request.form['prediction']
    comment = request.form['comment']

    # Extract numeric value from prediction text (e.g., "Predicted Score: 79.41" -> 79.41)
    try:
        prediction_value = float(prediction.split(': ')[1])
    except (IndexError, ValueError):
        prediction_value = 0.0

    # When preparing data for PDF:
    if prediction_value >= 75:
        pdf_comment = "Good"
    elif prediction_value >= 50:
        pdf_comment = "Average"
    else:
        pdf_comment = "Bad"

    html = render_template('pdf_template.html',
                           prediction_text=prediction,
                           study_time=study_time,
                           attendance=attendance,
                           past_scores=past_scores,
                           extra_classes=extra_classes,
                           comment=pdf_comment)  # Pass pdf_comment to the PDF template
    config = pdfkit.configuration(wkhtmltopdf=r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe')  # Update path if needed
    pdf = pdfkit.from_string(html, False, configuration=config)
    return (pdf, 200, {
        'Content-Type': 'application/pdf',
        'Content-Disposition': 'attachment; filename="prediction_result.pdf"'
    })

if __name__ == "__main__":
    app.run(debug=True)
