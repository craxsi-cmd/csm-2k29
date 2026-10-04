# CSM 2K29

CSM 2K29 is an AI-powered classroom attendance prototype.

## Features

- Teacher login screen
- Student registration
- Face embedding generation
- Multi-face classroom recognition
- Present and absent marking
- Date-wise Excel register
- PDF attendance export
- Local biometric-data storage

## Run locally

`powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
`

## Demo login

Username: teacher

Password: csm2k29

## Privacy note

This academic prototype stores student photos and face embeddings locally.
Production deployment requires consent, encrypted storage, proper authentication,
access control, audit logs, retention rules, and a manual review process.
