"""
Task 3: FastAPI app to serve model predictions
- Endpoint: POST /predict (multipart/form-data for image) or JSON for tabular
- Also includes a simple GET root page
Run: uvicorn src.task3_end_to_end_api:app --reload --port 8000
"""

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn
from PIL import Image
import io
import torch
from torchvision import transforms, models
import os

app = FastAPI(title="CODTECH Internship - Task 3 API")
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'saved_model.pth')

# load a simple model skeleton
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # adjust classes if needed
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
model.to(device)

transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html><body>
    <h2>CODTECH Internship - Model API</h2>
    <form action="/predict" enctype="multipart/form-data" method="post">
      <input name="file" type="file">
      <input type="submit" value="Upload and Predict">
    </form>
    </body></html>
    """

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # read image
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        preds = torch.softmax(out, dim=1).cpu().numpy().tolist()[0]
    return JSONResponse({"predictions": preds})

if __name__ == "__main__":
    uvicorn.run("src.task3_end_to_end_api:app", host="0.0.0.0", port=8000, reload=True)
