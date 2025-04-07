from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, send_file, jsonify
from pymongo import MongoClient, errors
from werkzeug.security import generate_password_hash, check_password_hash
from bson.objectid import ObjectId
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
import requests
from bs4 import BeautifulSoup
import os
import re
from google.oauth2 import service_account
from googleapiclient.discovery import build
from fpdf import FPDF
import google.generativeai as genai
import mimetypes
from transformers import pipeline
import pdfplumber
from datetime import datetime, timedelta
import logging

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "your_secret_key")  # Use environment variable for secret key

# Session timeout (in seconds)
app.permanent_session_lifetime = 1800  # 30 minutes

# MongoDB connection
try:
    client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
    db = client["itutor"]
    users_collection = db["users"]
    syllabus_collection = db["syllabus"]    
    quiz_results_collection = db["quiz_results"]
    quizzes_collection = db["quizzes"]  # Stores quiz questions
    user_attempts_collection = db["user_attempts"]  # Stores user quiz attempts
    user_progress_collection = db["user_progress"] 
except errors.ServerSelectionTimeoutError as e:
    print("MongoDB connection failed:", e)
    exit(1)

genai.configure(api_key="AIzaSyDkv7RqOtVO4CR0wMws77-CyILJcNNw1qI")

# File upload configuration
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Question generation pipeline
question_generator = pipeline("text2text-generation", model="valhalla/t5-small-qg-prepend")

# Utility function: Allowed file check
def allowed_file(filename):
    if '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']:
        mime_type, _ = mimetypes.guess_type(filename)
        return mime_type == 'application/pdf'
    return False

def extract_text_from_pdf(pdf_file):
    from PyPDF2 import PdfReader
    reader = PdfReader(pdf_file)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    # Log the extracted text for debugging
    print("Extracted Text:", text)
    
    return text
    
from transformers import T5ForConditionalGeneration, T5Tokenizer
import re
import random

# Load the T5 model and tokenizer
model_name = "valhalla/t5-small-qg-prepend"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def preprocess_text(text):
    """
    Preprocess the extracted text to make it suitable for MCQ generation.
    """
    # Remove code snippets (lines containing "import", "public class", etc.)
    lines = text.split("\n")
    filtered_lines = [line for line in lines if not any(keyword in line for keyword in ["import", "public class", "<?xml", "android:", "setContentView", "Intent", "Bundle"])]
    
    # Join the filtered lines into a single string
    filtered_text = "\n".join(filtered_lines)
    
    # Remove extra spaces and special characters
    filtered_text = re.sub(r'\s+', ' ', filtered_text)  # Replace multiple spaces with a single space
    filtered_text = re.sub(r'[^\w\s.,;!?]', '', filtered_text)  # Remove special characters
    
    return filtered_text

def generate_mcq_rule_based(text):
    """
    Generate MCQs using a rule-based approach.
    """
    # Extract keywords (naive approach)
    keywords = [word for word in text.split() if word.isalpha() and len(word) > 5]
    
    mcqs = []
    for i in range(5):  # Generate 5 questions
        if not keywords:
            break
        
        keyword = random.choice(keywords)
        question = f"What is {keyword}?"
        options = [keyword, random.choice(keywords), random.choice(keywords), random.choice(keywords)]
        random.shuffle(options)
        correct_answer = keyword
        mcqs.append({
            "question": question,
            "options": options,
            "correct_answer": correct_answer
        })
    
    return mcqs

def generate_mcq(text):
    """
    Generate multiple-choice questions (MCQs) using the T5 model or a rule-based fallback.
    """
    # Preprocess the text
    preprocessed_text = preprocess_text(text)
    print("Preprocessed Text:", preprocessed_text)  # Debugging

    # Try generating MCQs with the T5 model
    try:
        # Split the text into smaller chunks
        chunk_size = 500  # Adjust based on your needs
        text_chunks = [preprocessed_text[i:i + chunk_size] for i in range(0, len(preprocessed_text), chunk_size)]

        mcqs = []
        for chunk in text_chunks:
            # Prepare the input text for the model
            input_text = f"generate questions: {chunk}"
            inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

            # Generate questions
            outputs = model.generate(inputs, max_length=500, num_return_sequences=1)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Log the raw generated text for debugging
            print("Raw Generated Text:", generated_text)

            # Parse the generated text into MCQs
            lines = generated_text.split("\n")
            i = 0

            while i < len(lines):
                if "Question" in lines[i]:  # Look for lines containing "Question"
                    # Extract question
                    question = lines[i].strip().replace("**", "").replace("Question", "").strip()
                    i += 1

                    # Extract options
                    options = []
                    while i < len(lines) and (lines[i].strip().startswith("a)") or lines[i].strip().startswith("b)") or lines[i].strip().startswith("c)")):
                        options.append(lines[i].strip())
                        i += 1

                    # Extract correct answer from the answer key
                    if i < len(lines) and ("Correct Answer:" in lines[i]):
                        correct_answer_line = lines[i].strip()
                        correct_answer = correct_answer_line.split(":")[1].strip().replace("**", "")
                        mcqs.append({
                            "question": question,
                            "options": options,
                            "correct_answer": correct_answer
                        })
                i += 1

        if mcqs:
            return mcqs
        else:
            print("T5 model failed to generate MCQs. Falling back to rule-based method.")
            return generate_mcq_rule_based(preprocessed_text)
    except Exception as e:
        print(f"Error generating MCQs with T5 model: {e}. Falling back to rule-based method.")
        return generate_mcq_rule_based(preprocessed_text)

# Routes
@app.route('/')
def dashboard():
    user = None
    if "user" in session:
        user = users_collection.find_one({"email": session['user']})
    return render_template("dashboard.html", user=user)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        if users_collection.find_one({"email": email}):
            print("User already exists!")
            return redirect(url_for("signup"))

        hashed_password = generate_password_hash(password)
        users_collection.insert_one({"name": name, "email": email, "password": hashed_password, "interests": []})
        return redirect(url_for("login"))
    return render_template("signup.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = users_collection.find_one({"email": email})
        if user and check_password_hash(user['password'], password):
            session['user'] = user['email']
            session.permanent = True
            return redirect(url_for("dashboard"))
        print("Invalid credentials!")
        return redirect(url_for("login"))
    return render_template("login.html")

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for("dashboard"))

@app.route('/profile')
def profile():
    if "user" not in session:
        return redirect(url_for("login"))

    user = users_collection.find_one({"email": session['user']})
    return render_template("profile.html", user=user)

@app.route('/update_profile', methods=['POST'])
def update_profile():
    if "user" not in session:
        return redirect(url_for("login"))

    updated_name = request.form.get("name")
    updated_email = request.form.get("email")
    updated_interests = request.form.get("interests")

    users_collection.update_one(
        {"email": session['user']},
        {"$set": {
            "name": updated_name,
            "email": updated_email,
            "interests": updated_interests.split(",")
        }}
    )
    session['user'] = updated_email
    return redirect(url_for("profile"))

@app.route('/upload_syllabus', methods=['GET', 'POST'])
def upload_syllabus():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file selected", 400
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Extract chapters
            chapters = extract_chapters_from_pdf(file_path)
            session['chapters'] = chapters

            return redirect(url_for('syllabus', filename=filename))
    return render_template('upload_syllabus.html')

@app.route('/syllabus')
def syllabus():
    filename = request.args.get('filename')
    if not filename:
        return "Filename is required", 400
    chapters = session.get('chapters', [])
    file_url = url_for('uploaded_file', filename=filename)
    return render_template('syllabus.html', file_url=file_url, chapters=chapters)

def extract_chapters_from_pdf(file_path):
    reader = PdfReader(file_path)
    chapters = []
    chapter_pattern = re.compile(r'\bChapter\s+\d+\b', re.IGNORECASE)
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        matches = chapter_pattern.findall(text)
        if matches:
            chapters.append((i, matches[0]))
    return chapters

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/read_now', methods=['POST'])
def read_now():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Redirect to a new route that displays the PDF
        return redirect(url_for('view_pdf', filename=filename))

@app.route('/view_pdf')
def view_pdf():
    filename = request.args.get('filename')
    if not filename:
        return "Filename is required", 400
    return render_template('pdf_viewer.html', filename=filename)

@app.route('/quiz_now', methods=['POST'])
def quiz_now():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Extract text from the PDF
        extracted_text = extract_text_from_pdf(file_path)

        # Generate questions
        questions = generate_questions(extracted_text)

        # Create a question paper PDF
        question_paper_path = create_question_paper(questions)

        # Provide the generated PDF for download
        return send_file(question_paper_path, as_attachment=True)

def generate_questions(text):
    """Generates relevant questions from the extracted PDF text."""
    chunk_size = 1500  # Adjust the chunk size for better AI context
    text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    questions = []
    
    for chunk in text_chunks:
        question_prompt = f"Generate 5 multiple-choice and 5 descriptive questions from this text:\n{chunk}"
        response = question_generator(question_prompt, max_length=300)
        questions.extend(response[0]['generated_text'].split("\n"))
    
    return questions[:25]  # Limit to 25 questions

# Fix question paper formatting in PDF

def create_question_paper(questions):
    """Creates a well-formatted PDF question paper."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(200, 10, "Generated Question Paper", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    
    for i, question in enumerate(questions, start=1):
        pdf.multi_cell(0, 8, f"{i}. {question}")
        pdf.ln(5)
    
    question_paper_path = os.path.join(UPLOAD_FOLDER, "question_paper.pdf")
    pdf.output(question_paper_path)
    
    return question_paper_path


def fetch_research_papers(topic):
    url = f"https://scholar.google.com/scholar?q={topic}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    papers = []
    for item in soup.find_all('div', class_='gs_ri'):
        title = item.find('h3', class_='gs_rt').text
        link = item.find('a')['href']
        snippet = item.find('div', class_='gs_rs').text
        papers.append({
            "title": title,
            "link": link,
            "snippet": snippet
        })

    return papers

# Route to handle the research paper request
@app.route('/research', methods=['POST'])
def research():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json
    topic = data.get("topic", "").strip()

    if not topic:
        return jsonify({"error": "No topic provided"}), 400

    try:
        papers = fetch_research_papers(topic)
        return jsonify({"papers": papers})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to render the research page
@app.route('/research_page')
def research_page():
    return render_template("research.html")

@app.route('/generate_mcq_quiz', methods=['GET', 'POST'])
def generate_mcq_quiz():
    if request.method == 'POST':
        # Handle file upload and MCQ generation
        if 'file' not in request.files:
            return render_template('upload_mcq_quiz.html', error="No file uploaded. Please choose a PDF file.")

        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Extract text from the PDF
            extracted_text = extract_text_from_pdf(file_path)
            print("Extracted Text:", extracted_text)  # Debugging

            if not extracted_text:
                return render_template('upload_mcq_quiz.html', error="Failed to extract text from the PDF. Please upload a valid PDF.")

            # Generate MCQs using the T5 model
            mcqs = generate_mcq(extracted_text)
            print("Generated MCQs:", mcqs)  # Debugging

            if not mcqs:
                return render_template('upload_mcq_quiz.html', error="Failed to generate MCQs. Please try again with a different PDF.")

            # Store MCQs in session for later use
            session['mcqs'] = mcqs
            print("Session MCQs:", session['mcqs'])  # Debugging

            # Redirect to the quiz page
            return redirect(url_for('take_quiz'))

        return render_template('upload_mcq_quiz.html', error="Invalid file type. Please upload a PDF.")
    else:
        # Render the upload form for GET requests
        return render_template('upload_mcq_quiz.html')

@app.route('/take_quiz')
def take_quiz():
    if 'mcqs' not in session:
        return redirect(url_for('dashboard'))

    mcqs = session['mcqs']
    return render_template('mcq_quiz.html', mcqs=mcqs)

@app.route('/submit_quiz', methods=['POST'])
def submit_quiz():
    if 'mcqs' not in session:
        return redirect(url_for('dashboard'))

    mcqs = session['mcqs']
    user_answers = request.form.to_dict()
    
    # Evaluate answers
    score = 0
    results = []
    for i, mcq in enumerate(mcqs):
        user_answer = user_answers.get(f'question_{i}')
        is_correct = user_answer == mcq['correct_answer']
        if is_correct:
            score += 1
        results.append({
            'question': mcq['question'],
            'user_answer': user_answer,
            'correct_answer': mcq['correct_answer'],
            'is_correct': is_correct
        })

    # Identify weak areas
    weak_areas = analyze_quiz_results(results)

    # Generate study schedule
    study_schedule = generate_study_schedule(weak_areas)

    # Store results and schedule in session for display
    session['quiz_results'] = {
        'score': score,
        'total': len(mcqs),
        'results': results,
        'study_schedule': study_schedule
    }

    return redirect(url_for('quiz_results'))

@app.route('/quiz_results')
def quiz_results():
    if 'quiz_results' not in session:
        return redirect(url_for('dashboard'))

    results = session['quiz_results']
    return render_template('quiz_results.html', results=results)

@app.route('/generate_schedule')
def generate_schedule():
    if 'quiz_results' not in session:
        return redirect(url_for('dashboard'))

    results = session['quiz_results']
    weak_areas = [result['question'] for result in results['results'] if not result['is_correct']]

    # Generate a study schedule based on weak areas
    schedule = []
    for i, area in enumerate(weak_areas):
        schedule.append({
            'day': i + 1,
            'topic': area,
            'tasks': [
                f"Review {area}",
                f"Practice questions on {area}",
                f"Take a mini-quiz on {area}"
            ]
        })

    return render_template('schedule.html', schedule=schedule)

def analyze_quiz_results(results):
    """
    Analyze quiz results to identify weak areas.
    """
    weak_areas = []
    for result in results:
        if not result["is_correct"]:
            weak_areas.append(result["question"])
    return weak_areas

def generate_study_schedule(weak_areas):
    """
    Generate a study schedule based on weak areas.
    """
    schedule = []
    for i, area in enumerate(weak_areas):
        schedule.append({
            "day": i + 1,
            "topic": area,
            "tasks": [
                f"Review {area}",
                f"Practice questions on {area}",
                f"Take a mini-quiz on {area}"
            ]
        })
    return schedule

@app.route('/progress')
def progress():
    if "user" not in session:
        return redirect(url_for("login"))

    user_email = session['user']
    user_progress = user_progress_collection.find_one({"user_email": user_email})
    study_schedule = generate_study_schedule(user_email)

    return render_template("progress.html", user_progress=user_progress, study_schedule=study_schedule)

from datetime import datetime, timedelta

@app.route('/mark_day_completed/<int:day_index>', methods=['POST'])
def mark_day_completed(day_index):
    if "study_plan" not in session:
        return redirect(url_for("dashboard"))

    # Mark the day as completed
    study_plan = session['study_plan']
    study_plan[day_index]["completed"] = True
    session['study_plan'] = study_plan

    return redirect(url_for('view_study_plan'))

def adjust_study_plan(study_plan):
    """
    Adjust the study plan if the user falls behind.
    """
    for i in range(len(study_plan) - 1):
        if not study_plan[i]["completed"]:
            # Move incomplete tasks to the next day
            study_plan[i + 1]["tasks"].extend(study_plan[i]["tasks"])
            study_plan[i + 1]["resources"].extend(study_plan[i]["resources"])
            study_plan[i]["tasks"] = []
            study_plan[i]["resources"] = []

    return study_plan

@app.route('/view_study_plan')
def view_study_plan():
    if "study_plan" not in session:
        return redirect(url_for("dashboard"))

    study_plan = session['study_plan']
    study_plan = adjust_study_plan(study_plan)  # Adjust the plan dynamically
    session['study_plan'] = study_plan

    return render_template("study_plan.html", study_plan=study_plan)

def generate_ai_study_plan(topics, deadline, hours_per_day):
    """
    Generate a detailed study plan using AI.
    """
    # Calculate the number of days until the deadline
    days_until_deadline = (deadline - datetime.now()).days

    # Simulate an AI-generated study plan
    study_plan = []
    for i in range(days_until_deadline):
        day_plan = {
            "day": i + 1,
            "date": (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d"),
            "topics": [],
            "tasks": [],
            "resources": [],
            "completed": False  # Track if the day's tasks are completed
        }

        # Assign topics to each day
        topic_index = i % len(topics)
        topic = topics[topic_index]
        day_plan["topics"].append(topic)

        # Add detailed tasks
        day_plan["tasks"].append(f"Study {topic} for {hours_per_day} hours")
        day_plan["tasks"].append(f"Practice questions on {topic}")
        day_plan["tasks"].append(f"Take a mini-quiz on {topic}")

        # Add resource recommendations
        if topic.lower() == "photosynthesis":
            day_plan["resources"].append("Book: 'Photosynthesis and Respiration' by John Doe")
            day_plan["resources"].append("Video: 'Introduction to Photosynthesis' on YouTube")
        elif topic.lower() == "cellular respiration":
            day_plan["resources"].append("Book: 'Cellular Respiration Explained' by Jane Smith")
            day_plan["resources"].append("Article: 'Understanding Cellular Respiration' on Khan Academy")

        # Add revision days every 3 days
        if (i + 1) % 3 == 0:
            day_plan["topics"].append("Revision")
            day_plan["tasks"].append("Revise all previously studied topics")
            day_plan["tasks"].append("Take a full-length practice test")

        study_plan.append(day_plan)

    return study_plan

@app.route('/generate_study_plan', methods=['POST'])
def generate_study_plan():
    if "user" not in session:
        return redirect(url_for("login"))

    # Get user inputs from the form
    topics = request.form.get("topics").split(",")
    deadline = datetime.strptime(request.form.get("deadline"), "%Y-%m-%d")
    hours_per_day = int(request.form.get("hours_per_day"))

    # Generate a study plan using AI
    study_plan = generate_ai_study_plan(topics, deadline, hours_per_day)

    # Store the study plan in the session for display
    session['study_plan'] = study_plan

    return redirect(url_for('view_study_plan'))

if __name__ == '__main__':
    app.run(debug=True)
