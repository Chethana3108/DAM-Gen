# Sitecore DAM GenAI Brand Campaign Assistant (FLUX.2 Edition)

A professional, prompt-only GenAI campaign visual generator powered by open-source diffusion models (**FLUX.2-dev** / **FLUX.2-klein** / **FLUX.1**) and integrated directly with **Sitecore Content Hub DAM**.

---

## 🌟 Key Capabilities

1. **Native FLUX.2 Multi-Reference Flow Transformer (No 2D Sticker Pasting)**:
   - The engine automatically retrieves authentic product packshots, linked layers, and official logos from Sitecore DAM.
   - Using **FLUX.2-dev** (32B parameter multi-reference rectified flow transformer) powered by Google's 11B **T5-XXL** text reasoning encoder, the AI synthesizes the entire scene **organically in 3D**.
   - Renders accurate brand typography, exact packaging details, and genuine textures with zero fake gibberish labels.
   - Completely eliminates fake jars, mismatched containers, and artificial 2D cutout layers.

2. **Dynamic Color & Style Intelligence**:
   - Automatically detects color palettes mentioned in your prompt (*"warm golden sunlight"*, *"ruby red berry accents"*, *"emerald green"*, *"deep chocolate brown"*).
   - Dynamically harmonizes lighting, background tones, and environmental reflections to match requested colors.

3. **Universal Brand & Industry Grounding**:
   - Automatically recognizes brand and product context from Sitecore DAM (Britannia Good Day, Britannia Jim Jam, Himalaya, etc.) and injects physical grounding rules into the diffusion pipeline.

4. **Direct DAM Ingestion (Create Tab)**:
   - Generated visuals are automatically ingested into Sitecore Content Hub as `M.Asset` entities with status `Created` (`542`).
   - Instantly appear in your **Create Tab** (`https://btr-q-002.sitecorecontenthub.cloud/en-us/create?p840=1`) for marketing review and approval.

5. **Interactive Web Chat Interface**:
   - Built-in modern web chat served at `http://localhost:8000`.

---

## 🚀 One-Click Deployment on Vast.ai GPU Server

### 1. Provision a Vast.ai Instance
- **Recommended GPU**: RTX 3090 (24GB), RTX 4090 (24GB), A5000, A6000, or A100.
- **Template / Image**: Official PyTorch Docker image (`pytorch/pytorch:2.x-cuda12.x-cudnn8-devel` or Vast.ai default PyTorch).

### 2. Clone the Repository
```bash
git clone https://github.com/Chethana3108/DAM-Gen.git
cd DAM-Gen
```

### 3. Configure `.env`
Create `.env` or edit the existing file:
```env
# Open-Source Diffusion Model: 'flux.2' (FLUX.2-dev), 'flux' (FLUX.1-schnell), or 'ip-adapter'
DIFFUSION_MODEL_TYPE=flux.2
IMAGE_MODEL=FLUX.2-dev (Black Forest Labs 32B Multi-Reference)
MODEL_ENDPOINT=black-forest-labs/FLUX.2-dev

# Sitecore Content Hub DAM Settings
SITECORE_CH_URL=https://btr-q-002.sitecorecontenthub.cloud
SITECORE_USERNAME=kalpesh@biztechnosys.com
SITECORE_PASSWORD=P@ssw0rd7932
SITECORE_CLIENT_ID=ImageUpload
SITECORE_CLIENT_SECRET=URXfOKv-j-GoHXp0KhiRhQS9oEpm1uNlFE_ZXLrb2qg
```

### 4. Run the 1-Click Startup Script
```bash
bash deploy_vastai.sh
```

Open `http://<YOUR-VAST-AI-IP>:8000` (or the mapped port in your Vast.ai dashboard) to access the chat UI!

---

## 📡 API Endpoints

### 1. Prompt-Only Generation (`POST /api/generate`)
```bash
curl -X POST http://127.0.0.1:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create a vibrant social media ad for Britannia Jim Jam biscuits with fresh strawberries and ruby red berry accents",
    "aspect_ratio": "1:1",
    "sync_to_dam": true
  }'
```

### 2. Conversational Chat API (`POST /api/chat`)
```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Create a vibrant social media ad for Britannia Jim Jam biscuits with fresh strawberries and ruby red berry accents"
  }'
```

---

## 📂 Sitecore Content Hub Review
Whenever a campaign visual is generated, review and approve it inside your Content Hub instance:
👉 **`https://btr-q-002.sitecorecontenthub.cloud/en-us/create?p840=1`**
