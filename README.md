# Sitecore Content Hub DAM GenAI Campaign Studio - Backend REST API

Production pure backend REST API integrating **Sitecore Content Hub DAM** with OpenAI's **GPT Image** model (`gpt-image-2.5-sunburst`) for authentic, reference-guided commercial advertising visual generation, editing, and automated DAM ingestion.

---

## 🚀 Core Features

- **Direct Sitecore Content Hub DAM Integration**:
  - Dynamically searches and queries assets (`M.Asset`) and brands (`M.Brand`).
  - Supports disambiguation when multiple DAM assets match a query.
  - Downloads authentic reference assets and linked transparent layers.
  - Four-step ingestion workflow to create and verify new assets in the Sitecore Content Hub **Create Tab** (`/en-us/create?p840=1`) with rich HTML provenance and `RelatedAsset` linking.

- **Reference-Guided Scene Generation**:
  - Reusable `GPTImageService` using official OpenAI SDK and `gpt-image-2.5-sunburst`.
  - Genuine reference-image guidance preserving authentic brand packaging, logos, and labels.
  - Automated multi-modal creative prompt understanding using GPT vision (`gpt-4o-mini`).

- **Automated Commercial QA Validation**:
  - Comprehensive technical and semantic checks (color signature presence, absence of border step artifacts, dimension accuracy, dynamic range).
  - Multi-modal vision QA inspection with automatic refinement loop.

- **Pure REST API Architecture**:
  - Built with **FastAPI** and **Pydantic v2**.
  - CORS-enabled for seamless integration with modern web frontends (e.g., Next.js, React).
  - Interactive OpenAPI Swagger documentation at `/docs` and ReDoc at `/redoc`.

---

## 📁 Repository Structure

```
├── app/
│   ├── __init__.py
│   ├── asset_analyzer.py      # Non-destructive asset understanding & alpha channel detection
│   ├── engine.py              # Core image generation & refinement engine
│   ├── gpt_image_service.py   # OpenAI GPT Image & Multimodal Vision QA service
│   ├── main.py                # Pure FastAPI REST application & endpoint routers
│   ├── prompt_parser.py       # Dynamic natural-language creative prompt parser
│   ├── sitecore_dam.py        # Universal Sitecore Content Hub DAM Client (OAuth & Ingestion)
│   └── validator.py           # Commercial QA validation & defect diagnosis
├── .env.example               # Environment template
├── .gitignore                 # Git ignore rules (secrets, caches, builds)
├── README.md                  # Project documentation
└── requirements.txt           # Python runtime dependencies
```

---

## ⚙️ Configuration (`.env`)

Copy `.env.example` to `.env` and fill in your credentials:

```env
# Sitecore Content Hub DAM Settings
SITECORE_CH_URL=https://your-instance.sitecorecontenthub.cloud
SITECORE_USERNAME=your_username@domain.com
SITECORE_PASSWORD=your_password
SITECORE_CLIENT_ID=your_client_id
SITECORE_CLIENT_SECRET=your_client_secret

# OpenAI API Settings
OPENAI_API_KEY=sk-proj-your_openai_api_key_here
OPENAI_IMAGE_MODEL=gpt-image-2.5-sunburst
OPENAI_VISION_MODEL=gpt-4o-mini
LOG_LEVEL=INFO
```

---

## 🛠️ Installation & Running

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the API Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- **Base API**: `http://localhost:8000`
- **Interactive Swagger Docs**: `http://localhost:8000/docs`
- **ReDoc Documentation**: `http://localhost:8000/redoc`
- **Health Check**: `http://localhost:8000/health`

---

## 📡 REST API Reference

### Health & Status
- **`GET /`**: Service info, status, and documentation URLs.
- **`GET /health`**: Health status of the image engine, model, and DAM configuration.

### Sitecore DAM Assets
- **`GET /api/dam/assets?take=40&query=<search>`**: Browse and filter available image assets in Sitecore Content Hub.
- **`GET /api/dam/asset/{asset_id}`**: Retrieve full metadata, dimensions, and linked layers for an asset.

### Visual Generation & Variations
- **`POST /api/generate`**: Generate a commercial advertising visual using authentic DAM reference assets.
  ```json
  {
    "prompt": "Create a realistic advertisement showing this biscuit packet on a wooden cafe table beside a cup of coffee.",
    "asset_id": 69124,
    "aspect_ratio": "1:1",
    "sync_to_dam": false
  }
  ```
- **`POST /api/generate-variation`**: Generate an aesthetic variation with alternative lighting, framing, and styling.
  ```json
  {
    "prompt": "Create a realistic advertisement showing this biscuit packet on a wooden cafe table beside a cup of coffee.",
    "asset_id": 69124,
    "aspect_ratio": "1:1",
    "variation_index": 2
  }
  ```

### Ingestion & Session Control
- **`POST /api/approve-and-save`**: Ingest approved campaign image into Sitecore Content Hub DAM Create Sandbox with `RelatedAsset` links and HTML provenance.
  ```json
  {
    "image_base64": "<base64_encoded_png>",
    "source_asset_id": 69124,
    "source_asset_name": "Britannia Jim Jam Pops biscuit",
    "prompt": "Create a realistic advertisement showing this biscuit packet on a wooden cafe table beside a cup of coffee.",
    "filename": "Britannia_JimJam_Coffee_Campaign.png"
  }
  ```
- **`POST /api/cancel`**: Cancel current preview session without modifying DAM.

### Conversational Assistant
- **`POST /api/chat`**: Conversational agent endpoint supporting natural-language requests, asset disambiguation, and generation preview.
