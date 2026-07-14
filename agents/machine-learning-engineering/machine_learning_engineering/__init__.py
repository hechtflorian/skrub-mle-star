# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Machine Learning Engineer: automate the implementation of ML models."""

import os
from pathlib import Path

from dotenv import load_dotenv
from google.adk.utils.model_name_utils import is_gemini_model

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

model = os.environ.get("ROOT_AGENT_MODEL", "gemini-2.5-flash")
if is_gemini_model(model):
    import google.auth

    _, project_id = google.auth.default()
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")

os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")

from . import agent  # noqa: E402
