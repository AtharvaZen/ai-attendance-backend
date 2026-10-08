# AI Attendance System — Backend

FastAPI backend that recognises students from classroom photos using
**pretrained ArcFace face embeddings** (InsightFace). No neural network is
trained in this project — we only run inference with ready-made weights.

> **Status:** FastAPI backend with pretrained ArcFace recognition, student/face
> management, recognition, attendance and MongoDB persistence. Runs entirely on
> a normal Python install — **no Docker required**.

---

## 1. Requirements

| Item | Version |
|---|---|
| Python | 3.11+ (verified on 3.13.15, Windows x64) |
| MongoDB | Atlas (M0 free tier is enough) or a local server — optional |
| Node.js | 20.19+ (only to run the frontend) |
| Disk | ~600 MB free (`buffalo_l` weights are ≈275 MB) |

**Docker is not used anywhere in this project.** There is no `docker-compose.yml`
and no `Dockerfile`. MongoDB runs either as a managed Atlas cluster or as a
normal local process; everything else is plain `pip` and `npm`.

---

## 2. Install

```bash
cd backend

# create + activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows (Git Bash: source .venv/Scripts/activate)

# install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Notes:

* **No C++ compiler is required.** InsightFace 2.0 ships a pure-Python wheel.
* The pretrained `buffalo_l` package is downloaded automatically on the first
  run into `%USERPROFILE%\.insightface\models\` (or `FACE_MODEL_ROOT`).
* **NVIDIA GPU (optional):** `pip uninstall -y onnxruntime && pip install onnxruntime-gpu`
  then set `FACE_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider`.

---

## 3. Configure

```bash
copy .env.example .env          # then edit values (bash: cp .env.example .env)
```

Nothing is hard-coded: all settings come from environment variables.
Key variables:

| Variable | Default | Meaning |
|---|---|---|
| `MONGODB_URI` | *(empty)* | MongoDB connection string — empty ⇒ in-memory store |
| `MONGODB_DATABASE` | `attendance` | database name |
| `SCHOOL_ID` | `default` | school every document is scoped to |
| `SCHOOL_NAME` | `Default School` | display name for that school |
| `EMBEDDING_DIM` | `512` | face-embedding vector length (matches `buffalo_l`) |
| `UPLOAD_DIR` | `uploads` | where temporary uploads are written |
| `FACE_MATCH_THRESHOLD` | `0.45` | cosine-similarity accept threshold |
| `FACE_MATCH_MARGIN` | `0.05` | minimum lead over the best *other* student (0 = off) |
| `FACE_MODEL_NAME` | `buffalo_l` | InsightFace model package |
| `FACE_DET_SIZE` | `640` | detector input size |
| `FACE_MIN_FACE_SIZE` | `60` | reject faces smaller than N px |
| `FACE_PROVIDERS` | *(auto)* | Force `CPUExecutionProvider`, `CUDAExecutionProvider`, … |
| `ENVIRONMENT` | `development` | runtime environment label |

### Storage (MongoDB)

MongoDB is the persistent backend. There are **three** ways to run it — none of
them require Docker:

| Option | What you do | `$vectorSearch` |
| --- | --- | --- |
| **Atlas (recommended)** | Create a free M0 cluster, copy its connection string | ✅ supported |
| **Local server** | Install MongoDB Community, run `mongod` as a normal Windows service/process | ❌ exact cosine fallback |
| **None (demo)** | Leave `MONGODB_URI` empty — data is in-memory and lost on restart | n/a |

Then create the collections and indexes once:

```bash
python scripts/init_mongo.py
```

**Atlas:** put the string in `backend/.env` →
`MONGODB_URI=mongodb+srv://<user>:<password>@<cluster>/?retryWrites=true&w=majority`

**Local server:** install MongoDB Community (no container), start `mongod`, then
`MONGODB_URI=mongodb://127.0.0.1:27017`

Collections: `schools`, `students`, `face_embeddings`, `attendance`,
`attendance_audit` (plus an internal `counters` collection for integer ids).

* Students are unique on **(`schoolId`, `rollNumber`)** — roll numbers are *not*
  globally unique, they are unique per school.
* Attendance is unique on
  (**`schoolId`, `studentId`, `className`, `section`, `attendanceDate`**), which
  makes attendance processing idempotent.
* Face embeddings are stored per photo, so a student can hold several.
* Similarity search uses MongoDB's native `$vectorSearch` (cosine, filtered by
  `schoolId`) when the deployment supports it, and otherwise exact cosine
  similarity in Python — both return the same results. The school filter is
  applied *inside* the query, so one school can never match another's students.

---

## 4. Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

Then open **http://127.0.0.1:8000/docs** (Swagger) and check:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","model_loaded":true,"model":"buffalo_l","providers":["CPUExecutionProvider"],...}
```

The model is loaded **once** during startup (see the `lifespan` block in
`app/main.py`) and reused for every request.

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Service banner |
| `GET` | `/health` | Liveness + face-model readiness |
| `GET`,`POST` | `/api/students` | List / register students |
| `GET` | `/api/students/classes` | Known classes and sections |
| `GET`,`PATCH`,`DELETE` | `/api/students/{student_id}` | Read / correct / delete a student |
| `GET`,`POST` | `/api/students/{student_id}/faces` | List / register face photos |
| `DELETE` | `/api/students/{student_id}/faces/{face_id}` | Delete one embedding |
| `POST` | `/api/recognition/recognize` | Dry-run recognition (writes nothing) |
| `POST` | `/api/attendance/process` | Classroom photo → attendance rows |
| `GET` | `/api/attendance` | Register for a class (optionally one date) |
| `GET`,`PATCH` | `/api/attendance/{attendance_id}` | Read / manually correct a row |
| `GET` | `/api/attendance/{attendance_id}/audit` | Audit trail for one row |

```bash
curl http://127.0.0.1:8000/api/students/999/faces
# {"detail":"Student 999 not found","code":"student_not_found"}          (404)

curl http://127.0.0.1:8000/api/students/abc/faces
# {"detail":[...],"code":"invalid_request"}                              (422)
```

### Error contract

Every failure uses the same envelope — `{"detail": ..., "code": ...}` — and
never leaks an internal stack trace:

| Status | `code` | When |
|---|---|---|
| 400 | `invalid_image` | undecodable body, empty payload, wrong content type |
| 400 | `no_face_detected` | zero faces found |
| 400 | `multiple_faces_detected` | >1 face during registration |
| 400 | `face_too_small` | face below `FACE_MIN_FACE_SIZE` |
| 404 | `student_not_found` | unknown student id |
| 413 | `payload_too_large` | upload exceeds `MAX_UPLOAD_SIZE_MB` |
| 422 | `invalid_request` | path/query/body failed validation |
| 500 | `model_unavailable` | face model failed to initialise |
| 500 | `internal_error` | unexpected error (details only in server logs) |

Uploads are size-checked **while streaming**, so an oversized body is rejected
without being fully buffered.

---

## 5. Face-detection test (Phase 3)

```bash
# bundled InsightFace sample image
python scripts/detect_faces.py

# your own image, with an annotated copy written out
python scripts/detect_faces.py path/to/photo.jpg --save out/annotated.jpg
```

Expected output:

```
Image          : path/to/photo.jpg
Resolution     : 1280x720
Detected faces : 3
Detection time : 412.5 ms

  Face #1
    bbox      : (231, 118) -> (402, 331)  (171px)
    det score : 0.912
    embedding : shape=(512,)  L2-norm=1.0000
...
```

---

## 6. About the score (important)

The numbers this system reports are **cosine similarities between embeddings
(a confidence/similarity score)** — they are **not** probabilities of identity.

* `0.93` similarity does **not** mean "93 % probability this is the right student".
* The threshold in `.env` is a **starting point**. Tune it on your own test set
  in Phase 12 (0.35–0.60 are typical values) — different cameras, lighting and
  class sizes shift the sweet spot.

---

## 7. Project layout (Phase 1)

```
backend/
├── app/
│   ├── main.py                   # FastAPI app, lifespan (loads model once), routers
│   ├── api/
│   │   ├── dependencies.py       # upload validation (size/content-type/decode), model readiness
│   │   ├── health.py             # GET / and GET /health
│   │   └── students.py           # GET /api/students/{id}/faces (metadata only)
│   ├── core/
│   │   ├── config.py             # pydantic-settings + logging
│   │   └── errors.py             # AppError hierarchy + {"detail","code"} handlers
│   ├── database/
│   │   └── repository.py         # storage interface (MongoDB / in-memory)
│   ├── schemas/
│   │   ├── common.py             # ErrorResponse, ValidationErrorResponse, HealthResponse
│   │   └── student.py            # FaceMetadataResponse, StudentFacesResponse
│   └── services/
│       └── face_service.py       # InsightFace wrapper: detect + normalised embeddings
├── scripts/detect_faces.py       # Phase 3 detection test
├── requirements.txt
├── .env.example
└── README.md
```

Later phases add `api/{recognition,attendance}.py`, `models/`, `database/connection.py`,
`schemas/attendance.py` and `services/{embedding,matching,attendance}_service.py`.

---

## 8. Privacy & licensing

* **Face embeddings are sensitive biometric data.** Raw images are not kept
  permanently, embeddings are never returned by the API, and management
  endpoints will be authenticated from Phase 13.
* **Model licence:** the pretrained InsightFace weights (`buffalo_l`) are
  released **for non-commercial research use only**. Verify licensing before
  any commercial deployment.
