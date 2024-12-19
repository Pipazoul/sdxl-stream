# sdxl-stream

## Quick Start

```bash
conda create -n sdxl-stream python=3.11
conda activate sdxl-stream
pip install -r requirements.txt
```

**Start fastapi server**

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
**Start frontend**

```bash
cd frontend
npm run dev

Access the stream on
http://localhost:8000/video_feed

And remote control on
http://localhost:8000/