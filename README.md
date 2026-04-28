### 1. Create Virtual Environment

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# On Windows (Command Prompt):
.\.venv\Scripts\activate.bat

# On macOS/Linux:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
# Make sure you're in the backend directory with venv activated
pip install --upgrade pip

# Install required packages
pip install fastapi uvicorn ultralytics opencv-python numpy torch scikit-learn tqdm PyYAML python-multipart
```

### 3. Run Backend Server

```bash
# From backend directory with venv activated
python endpoints4.py

# Or using uvicorn directly
uvicorn endpoints4:app --reload --host 0.0.0.0 --port 8000
```

The backend will be available at `http://localhost:8000`

---

## Frontend Setup (React + TypeScript)

### 1. Install Dependencies

The frontend uses **Bun** as the package manager

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies with bun
bun install

# Or with npm if you prefer
npm install
```

### 2. Configure API Base URL

Update the backend API URL in the frontend configuration if needed.

### 3. Run Development Server

```bash
# From frontend directory
bun run dev

# Or with npm
npm run dev
```

The frontend will typically be available at `http://localhost:5173`

---

## Running Both Services

### Option 1: Two Terminal Windows

**Terminal 1 - Backend:**
```bash
cd backend
.\.venv\Scripts\Activate.ps1  # or activate on macOS/Linux
python endpoints4.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
bun run dev
```

### Option 2: Using VS Code Terminal Tabs

Create separate terminals in VS Code for each service and run the commands above.

---

## Troubleshooting

### Backend Issues

**Module not found errors:**
- Ensure virtual environment is activated
- Run: `pip install -r requirements.txt` (if available)
- Or install dependencies manually: `pip install fastapi uvicorn ultralytics opencv-python numpy torch scikit-learn tqdm`

**Port already in use:**
- Change port: `python endpoints4.py --port 8001`
- Or kill existing process on port 8000

### Frontend Issues

**Dependencies not installing:**
```bash
# Clear cache and reinstall
rm -rf node_modules
bun install  # or npm install
```

**Port 5173 already in use:**
- Vite will automatically use the next available port
- Check terminal output for actual URL

---

## API Endpoints

The backend provides several endpoints for detection:

- `GET /stream/start` - Start webcam stream
- `GET /stream/stop` - Stop webcam stream
- `GET /stream/status` - Get stream status
- `GET /stream/webcam` - Get MJPEG webcam stream
- `POST /detect` - Upload image for detection

---

## Development

### Backend Development
- Edit files in `backend/`
- Server auto-reloads with `--reload` flag
- Check `endpoints4.py` for API definitions

### Frontend Development
- Edit files in `frontend/src/`
- Hot reload enabled with Vite
- Uses TanStack Router for routing
- Radix UI components for UI

---

## Notes

- The YOLO model files (`yolov8n.pt`, `yolo11n.pt`) should be in the `backend/` directory
- Training dataset is located in `backend/training_dataset/`
- Results from training runs are saved in `backend/runs/`