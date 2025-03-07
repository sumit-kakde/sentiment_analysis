from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

#from     import HTMLResponse
from transformers import pipeline

# Initialize FastAPI app
app = FastAPI()

# Load the Hugging Face model pipeline
classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

# Define the homepage route
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Emotion Analysis</title>
        </head>
        <body style="font-family: Arial, sans-serif; text-align: center; margin-top: 50px;">
            <h1>Emotion Analysis Web App</h1>
            <form method="post" action="/analyze">
                <textarea name="text" rows="4" cols="50" placeholder="Enter text here..." required></textarea><br><br>
                <button type="submit">Analyze Emotion</button>
            </form>
        </body>
    </html>
    """

# Define the analysis route
@app.post("/analyze", response_class=HTMLResponse)
async def analyze(text: str = Form(...)):
    # Get emotion predictions from the model
    results = classifier(text)
    
    # Format results into HTML
    response_html = f"""
    <!DOCTYPE html>
    <html>
        <head>
            <title>Emotion Analysis Results</title>
        </head>
        <body style="font-family: Arial, sans-serif; text-align: center; margin-top: 50px;">
            <h1>Analysis Results</h1>
            <p><strong>Input Text:</strong> {text}</p>
            <h2>Detected Emotions:</h2>
            <ul style="list-style-type: none;">
    """
    for result in results:
        #label = result['label']
        label = result[0]['label']
        #score = round(result['score'], 2)
        score = round(result[0]['score'], 2)
        response_html += f"<li>{label}: {score}</li>"
    
    response_html += """
            </ul>
            <a href="/">Analyze Another Text</a>
        </body>
    </html>
    """
    return response_html
