"""
Push to HuggingFace Spaces - Enter Token Below
================================================
"""

# ======== ENTER YOUR HF TOKEN HERE ========
HF_TOKEN = "hf_your_token_here"  # Replace with your actual token from https://huggingface.co/settings/tokens
# ===========================================

if HF_TOKEN == "hf_your_token_here" or not HF_TOKEN:
    print("ERROR: Please edit this file and add your HF token!")
    print("1. Go to: https://huggingface.co/settings/tokens")
    print("2. Create a new token (if not exists)")
    print("3. Copy the token")
    print("4. Paste it in this file where it says 'hf_your_token_here'")
    exit(1)

from huggingface_hub import HfApi, create_repo
import os

REPO_ID = "rajatagrawal18/dataclean-ops"
SPACE_TITLE = "DataClean-Ops"

print(f"Authenticating as user...")

try:
    api = HfApi(token=HF_TOKEN)
    user_info = api.whoami()
    print(f"Logged in as: {user_info['name']}")
except Exception as e:
    print(f"Authentication failed: {e}")
    exit(1)

# Create Space
print(f"\nCreating Space: {REPO_ID}")
try:
    create_repo(
        repo_id=REPO_ID,
        repo_type="space",
        token=HF_TOKEN,
        private=False,
        space_sdk="docker",
        space_title=SPACE_TITLE,
        space_description="RL Environment for Data Engineering - Meta OpenEnv Challenge",
        license="mit"
    )
    print(f"[OK] Created Space")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"[OK] Space already exists")
    else:
        print(f"[ERROR] {e}")

# Upload files
print("\nUploading files...")

files_to_upload = ["app.py", "env.py", "models.py", "tasks.py", "openenv.yaml", 
                  "inference.py", "Dockerfile", "requirements.txt", "baseline.py", "__init__.py"]

for file in files_to_upload:
    if os.path.exists(file):
        try:
            api.upload_file(path_or_fileobj=file, path_in_repo=file, 
                           repo_id=REPO_ID, repo_type="space")
            print(f"  [OK] {file}")
        except Exception as e:
            print(f"  [FAIL] {file}: {e}")

print(f"\n✓ Upload Complete!")
print(f"URL: https://huggingface.co/spaces/{REPO_ID}")
print(f"Wait 2-3 minutes for build, then test your Space!")