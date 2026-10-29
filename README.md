# Sitecore DAM GenAI Brand Campaign Assistant (Vast.ai GPU Edition)

A professional, prompt-only multimodal GenAI campaign visual generator powered by **Qwen-Image-2.1** and integrated directly with **Sitecore Content Hub DAM**.

---

## 🌟 Key Capabilities

1. **Pure Multimodal Reference Conditioning**:
   - The engine automatically pulls the authentic product packshots and official logos from Sitecore DAM.
   - Feeds them directly as **visual reference inputs** (`input_images`) to the vision diffusion model.
   - The AI synthesizes the scene **organically around the genuine product and branding**, with native lighting reflections, true 3D perspective, and natural contact shadows.
2. **Dynamic Color & Style Intelligence**:
   - Automatically detects any color palette mentioned in your prompt (e.g., *"warm golden sunlight"*, *"ruby red berry accents"*, *"emerald green botanical leaves"*, *"deep chocolate brown"*).
   - Dynamically harmonizes lighting, background tones, and environmental reflections to match the requested colors.
3. **Cross-Industry Universal Support**:
   - Works across FMCG (biscuits, dairy, snacks), Pharma / Healthcare, Automotive, Fashion, Electronics, etc.
4. **Direct DAM Ingestion (Create Tab)**:
   - Newly generated visuals are automatically uploaded back into Sitecore Content Hub as `M.Asset` entities with status `Created` (`542`).
   - Instantly appear in your **Create Tab** (`https://<tenant>/en-us/create?p840=1`) for marketing review and approval.
5. **Interactive Web Chat Interface**:
   - Built-in modern chat interface served at `http://localhost:8000`.

---

## 🚀 Deploying on Vast.ai GPU Server

### 1. Provision an Instance on Vast.ai
- **Recommended GPU**: RTX 3090, RTX 4090, A5000, or A100.
- **Image**: PyTorch default or Ubuntu with CUDA 12+.

### 2. Clone / Copy Repository to Vast.ai
```bash
git clone <your-repo-url>
cd gemini_brand_image_api
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables (`.env`)
```env
# Hugging Face token (if calling HF Space from Vast.ai)
HF_TOKEN=your_huggingface_token

# Model endpoint:
# If calling Qwen Space: Qwen/Qwen-Image-2.1
# If running local model server on Vast.ai: http://127.0.0.1:7860
MODEL_ENDPOINT=Qwen/Qwen-Image-2.1

# Sitecore Content Hub DAM Settings
SITECORE_CH_URL=https://btr-q-002.sitecorecontenthub.cloud
SITECORE_USERNAME=your_username@domain.com
SITECORE_PASSWORD=your_password
SITECORE_CLIENT_ID=ImageUpload
SITECORE_CLIENT_SECRET=your_client_secret
```

### 5. Launch the Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open `http://<YOUR-VAST-AI-IP>:8000` or map port 8000 in your Vast.ai dashboard to access the chat interface directly.

---

## 📡 API Endpoints

### 1. Prompt-Only Generation (`POST /api/generate`)
```bash
curl -X POST http://127.0.0.1:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create a warm breakfast scene with Britannia Good Day butter cookies and a steaming cup of tea in warm golden sunlight",
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
