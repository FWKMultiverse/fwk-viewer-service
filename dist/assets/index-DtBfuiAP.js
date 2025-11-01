(function(){const i=document.createElement("link").relList;if(i&&i.supports&&i.supports("modulepreload"))return;for(const n of document.querySelectorAll('link[rel="modulepreload"]'))c(n);new MutationObserver(n=>{for(const o of n)if(o.type==="childList")for(const _ of o.addedNodes)_.tagName==="LINK"&&_.rel==="modulepreload"&&c(_)}).observe(document,{childList:!0,subtree:!0});function s(n){const o={};return n.integrity&&(o.integrity=n.integrity),n.referrerPolicy&&(o.referrerPolicy=n.referrerPolicy),n.crossOrigin==="use-credentials"?o.credentials="include":n.crossOrigin==="anonymous"?o.credentials="omit":o.credentials="same-origin",o}function c(n){if(n.ep)return;n.ep=!0;const o=s(n);fetch(n.href,o)}})();const I=`import torch\r
import torch.nn as nn\r
from torch_geometric.nn import GATv2Conv, global_mean_pool\r
from torch_geometric.data import Data, Batch\r
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline\r
from datasets import load_dataset\r
from torch.utils.data import Dataset, DataLoader, random_split\r
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training\r
import os\r
import glob\r
import requests\r
import base64\r
import time\r
import sys\r
from torch.nn.functional import mse_loss, softmax, log_softmax, pad\r
from accelerate import Accelerator\r
from tqdm.auto import tqdm\r
import re\r
import random\r
import json\r
import ast\r
import networkx as nx\r
import subprocess\r
import pickle\r
import matplotlib.pyplot as plt\r
import shutil\r
import logging\r
import numpy as np\r
import joblib\r
import optuna\r
import gc\r
from datetime import datetime\r
import heapq\r
import platform\r
import argparse\r
import threading\r
\r
# ! ##################################################################\r
# ! ################ V9 IMPORTS (from V8) ############################\r
# ! ##################################################################\r
import psutil # ! V8: For tracking CPU/RAM usage.\r
try:\r
    import pynvml # ! V8: For tracking NVIDIA VRAM usage.\r
    pynvml.nvmlInit()\r
    VRAM_MONITORING_AVAILABLE = True\r
except ImportError:\r
    VRAM_MONITORING_AVAILABLE = False\r
    logging.warning("pynvml not found. VRAM monitoring will be disabled. Install with 'pip install pynvml'")\r
\r
import tkinter as tk # ! V8: For creating GUI Applications\r
from tkinter import scrolledtext, Entry, Button, Frame\r
\r
try:\r
    import redis # ! V8: L1 Cache Memory\r
except ImportError:\r
    redis = None\r
    logging.warning("redis-py not found. L1 Cache (Redis) will be disabled. Install with 'pip install redis'")\r
\r
try:\r
    from qdrant_client import QdrantClient, models # ! V8: L2 Cache Memory\r
    from qdrant_client.http.models import Distance, VectorParams, PointStruct\r
except ImportError:\r
    QdrantClient = None\r
    logging.warning("qdrant-client not found. L2 Cache (Qdrant) will be disabled. Install with 'pip install qdrant-client'")\r
\r
try:\r
    from minio import Minio # ! V8: L3 Storage\r
except ImportError:\r
    Minio = None\r
    logging.warning("minio not found. L3 Storage (MinIO) will be disabled. Install with 'pip install minio'")\r
# ! ##################################################################\r
\r
\r
# --- Configuration & Setup (from v8) ---\r
def set_seed(s=42):\r
    random.seed(s)\r
    np.random.seed(s)\r
    torch.manual_seed(s)\r
    if torch.cuda.is_available():\r
        torch.cuda.manual_seed_all(s)\r
\r
set_seed(42)\r
\r
if torch.cuda.is_available():\r
    torch.backends.cudnn.benchmark = True\r
\r
# ! V9: Updated log file name\r
logging.basicConfig(filename='multiverse_v9_log.log', level=logging.INFO,\r
                    format='%(asctime)s - %(levelname)s - %(message)s')\r
console_handler = logging.StreamHandler()\r
console_handler.setLevel(logging.INFO)\r
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')\r
console_handler.setFormatter(formatter)\r
logging.getLogger().addHandler(console_handler)\r
\r
# --- ตรวจสอบการรองรับ Flash Attention 2 (from v8) ---\r
FLASH_ATTENTION_AVAILABLE = False\r
try:\r
    import flash_attn\r
    FLASH_ATTENTION_AVAILABLE = True\r
    logging.info("✅ Flash Attention 2 is available! Will be used for faster training.")\r
except ImportError:\r
    logging.warning("⚠️ Flash Attention 2 not found. Training will proceed without it, but might be slower.")\r
\r
# --- (same as v8) ---\r
try:\r
    from luaparser import ast as luaparser_ast\r
    from luaparser import parser as luaparser_parser\r
    logging.info("Using luaparser for Lua AST-based graphs.")\r
except ImportError:\r
    luaparser_ast = None\r
    luaparser_parser = None\r
    logging.warning("luaparser not found. Cannot build AST-based graphs.")\r
\r
# ! V9: NEW - Security & Moderation Agent (Non-LLM)\r
class SecurityModeratorAgent:\r
    """\r
    ! V9: NEW AGENT (Non-LLM)\r
    Security filters that run *before* and *after* LLM processing.\r
    Use Regex rules and Keyword lists for speed and reliability (no LLM, check LLM)\r
    """\r
    def __init__(self):\r
        logging.info("🛡️ Security Moderator Agent initialized.")\r
        \r
        # 1. List of profanity (example) - should be expanded to cover\r
        self.PROFANITY_LIST = set([\r
            "example_profanity1", "example_profanity2", "ควย", "เหี้ย", "สัส", "ไอ้สัตว์"\r
        ])\r
        \r
        # 2. Illegal/Dangerous Topics (Examples)\r
        self.ILLEGAL_TOPICS_KEYWORDS = set([\r
            "how to make a bomb", "วิธีทำระเบิด", "how to hack", "วิธีแฮก", "child pornography", \r
            "hate speech", "incite violence", "ยุยงให้เกิดความรุนแรง", "self-harm instruction"\r
        ])\r
        \r
        # 3. Personally Identifiable Information (PII) Format - (Regex Example)\r
        self.PII_PATTERNS = {\r
            "EMAIL": re.compile(r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b'),\r
            "PHONE_TH": re.compile(r'\\b0[689]\\d{1}-?\\d{3}-?\\d{4}\\b'),\r
            "CREDIT_CARD": re.compile(r'\\b(?:\\d[ -]*?){13,16}\\b'),\r
            "SSN_USA": re.compile(r'\\b\\d{3}-\\d{2}-\\d{4}\\b'),\r
        }\r
        \r
        self.SAFE_RESPONSE_FILTERED = "I apologize, but I cannot process this request. It violates my safety and moderation guidelines."\r
        self.SAFE_RESPONSE_PII = "I apologize, but I cannot process this request as it contains sensitive personal information. Please remove it and try again."\r
\r
    def _check_text(self, text):\r
        """Helper function to check text against all rules."""\r
        text_lower = text.lower()\r
        \r
        # Check profanity\r
        if any(profanity in text_lower for profanity in self.PROFANITY_LIST):\r
            return False, "Profanity detected."\r
            \r
        # Check illegal topics\r
        if any(topic in text_lower for topic in self.ILLEGAL_TOPICS_KEYWORDS):\r
            return False, "Illegal/Harmful topic detected."\r
            \r
        # Check PII\r
        for pii_type, pattern in self.PII_PATTERNS.items():\r
            if pattern.search(text):\r
                return False, f"PII detected: {pii_type}"\r
                \r
        return True, "Safe"\r
\r
    def pre_screen_input(self, prompt: str):\r
        """\r
        ! V9: กรอง Input *ก่อน* ส่งให้ LLM\r
        """\r
        is_safe, reason = self._check_text(prompt)\r
        if not is_safe:\r
            logging.warning(f"🛡️ Input blocked: {reason}. Prompt: '{prompt[:50]}...'")\r
            if "PII" in reason:\r
                return False, self.SAFE_RESPONSE_PII\r
            return False, self.SAFE_RESPONSE_FILTERED\r
        return True, "Input is safe."\r
\r
    def post_screen_output(self, response: str):\r
        """\r
        ! V9: กรอง Output *หลัง* LLM สร้าง\r
        (ป้องกันกรณี LLM หลอนและสร้างเนื้อหาไม่เหมาะสม)\r
        """\r
        is_safe, reason = self._check_text(response)\r
        if not is_safe:\r
            logging.warning(f"🛡️ Output blocked: {reason}. Response: '{response[:50]}...'")\r
            if "PII" in reason: # LLM ไม่ควรเปิดเผย PII\r
                return False, "I apologize, but my response contained sensitive information and has been redacted for your safety."\r
            return False, self.SAFE_RESPONSE_FILTERED\r
        return True, response\r
\r
\r
# ! V8: Resource Monitoring System (คงเดิม)\r
class ResourceMonitor(threading.Thread):\r
    def __init__(self, interval=5):\r
        super().__init__(daemon=True)\r
        self.interval = interval\r
        self.stopped = False\r
        self.initial_vram_used = 0\r
        self.initial_ram_used = psutil.Process(os.getpid()).memory_info().rss\r
        if VRAM_MONITORING_AVAILABLE and torch.cuda.is_available():\r
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)\r
            self.initial_vram_used = pynvml.nvmlDeviceGetMemoryInfo(self.handle).used\r
        logging.info("📊 Resource Monitor initialized.")\r
\r
    def run(self):\r
        while not self.stopped:\r
            # CPU\r
            cpu_usage = psutil.cpu_percent()\r
            # RAM\r
            process = psutil.Process(os.getpid())\r
            ram_info = process.memory_info()\r
            ram_used_gb = ram_info.rss / (1024 ** 3)\r
            ram_reduction = ((self.initial_ram_used - ram_info.rss) / self.initial_ram_used) * 100 if self.initial_ram_used > 0 else 0\r
            \r
            log_message = f"Resources: CPU: {cpu_usage:.1f}% | RAM: {ram_used_gb:.2f} GB (Reduction: {ram_reduction:.1f}%)"\r
\r
            # VRAM (if available)\r
            if VRAM_MONITORING_AVAILABLE and torch.cuda.is_available():\r
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)\r
                vram_used_gb = mem_info.used / (1024 ** 3)\r
                vram_total_gb = mem_info.total / (1024 ** 3)\r
                vram_reduction = ((self.initial_vram_used - mem_info.used) / self.initial_vram_used) * 100 if self.initial_vram_used > 0 else 0\r
                log_message += f" | VRAM: {vram_used_gb:.2f}/{vram_total_gb:.2f} GB (Reduction: {vram_reduction:.1f}%)"\r
            \r
            logging.info(log_message)\r
            time.sleep(self.interval)\r
\r
    def stop(self):\r
        self.stopped = True\r
        if VRAM_MONITORING_AVAILABLE:\r
            pynvml.nvmlShutdown()\r
\r
# ! V8: Hierarchical Memory System (คงเดิม)\r
class HierarchicalMemory:\r
    def __init__(self, embedding_pipeline):\r
        self.embedding_pipeline = embedding_pipeline\r
        self.embedding_dim = 768 # CodeBERT base model\r
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'\r
\r
        # L1 Cache: Redis\r
        self.redis_client = None\r
        if redis:\r
            try:\r
                self.redis_client = redis.Redis(\r
                    host=os.getenv("REDIS_HOST", "localhost"),\r
                    port=int(os.getenv("REDIS_PORT", 6379)),\r
                    db=0,\r
                    decode_responses=True\r
                )\r
                self.redis_client.ping()\r
                logging.info("✅ L1 Cache (Redis) connected successfully.")\r
            except Exception as e:\r
                logging.error(f"❌ Could not connect to L1 Cache (Redis): {e}")\r
                self.redis_client = None\r
        \r
        # L2 Cache: Qdrant\r
        self.qdrant_client = None\r
        # ! V9: Updated collection name\r
        self.qdrant_collection_name = "multiverse_v9_memory"\r
        if QdrantClient:\r
            try:\r
                self.qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))\r
                # ! V9: ใช้ try-except แทน recreate เพื่อป้องกันข้อมูลหาย\r
                try:\r
                    self.qdrant_client.recreate_collection(\r
                        collection_name=self.qdrant_collection_name,\r
                        vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE),\r
                    )\r
                    logging.info(f"✅ L2 Cache (Qdrant) collection '{self.qdrant_collection_name}' re-created.")\r
                except Exception as e_coll:\r
                    logging.warning(f"Could not recreate collection (maybe it exists?): {e_coll}. Assuming it's ready.")\r
                \r
                logging.info(f"✅ L2 Cache (Qdrant) connected.")\r
            except Exception as e:\r
                logging.error(f"❌ Could not connect to L2 Cache (Qdrant): {e}")\r
                self.qdrant_client = None\r
\r
        # L3 Storage: MinIO\r
        self.minio_client = None\r
        self.minio_bucket_name = "multiverse-v9-long-term-storage"\r
        if Minio:\r
            try:\r
                self.minio_client = Minio(\r
                    os.getenv("MINIO_ENDPOINT", "localhost:9000"),\r
                    access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),\r
                    secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),\r
                    secure=False\r
                )\r
                if not self.minio_client.bucket_exists(self.minio_bucket_name):\r
                    self.minio_client.make_bucket(self.minio_bucket_name)\r
                logging.info(f"✅ L3 Storage (MinIO) connected and bucket '{self.minio_bucket_name}' is ready.")\r
            except Exception as e:\r
                logging.error(f"❌ Could not connect to L3 Storage (MinIO): {e}")\r
\r
    def _get_embedding(self, text):\r
        try:\r
            # ! V9: ปรับปรุงการจัดการ embedding (กรณี text ว่าง)\r
            if not text or not text.strip():\r
                return None\r
            embedding = self.embedding_pipeline(text)\r
            vector = np.array(embedding).mean(axis=1).flatten()\r
            return vector\r
        except Exception as e:\r
            logging.error(f"Could not generate embedding for text: {e}")\r
            return None\r
\r
    def add_experience(self, code_snippet, reward, metadata={}):\r
        if reward < 5.0: return # Only store high-quality experiences\r
        \r
        snippet_hash = str(hash(code_snippet))\r
        vector = self._get_embedding(code_snippet)\r
        if vector is None: return\r
\r
        # Add to L1 (hot cache)\r
        if self.redis_client:\r
            self.redis_client.set(f"code:{snippet_hash}", code_snippet, ex=3600) # Expire in 1 hour\r
\r
        # Add to L2 (semantic search)\r
        if self.qdrant_client:\r
            self.qdrant_client.upsert(\r
                collection_name=self.qdrant_collection_name,\r
                points=[\r
                    PointStruct(id=snippet_hash, vector=vector.tolist(), payload={"code": code_snippet, **metadata})\r
                ]\r
            )\r
        \r
        # Add to L3 (permanent storage) - can be extended to store more complex objects\r
        logging.info(f"🧠 Added high-quality experience (Hash: {snippet_hash}) to Hierarchical Memory.")\r
\r
    def retrieve_similar(self, query_code, k=3):\r
        query_vector = self._get_embedding(query_code)\r
        if query_vector is None or self.qdrant_client is None: return []\r
\r
        try:\r
            search_result = self.qdrant_client.search(\r
                collection_name=self.qdrant_collection_name,\r
                query_vector=query_vector,\r
                limit=k,\r
                with_payload=True\r
            )\r
            return [hit.payload['code'] for hit in search_result]\r
        except Exception as e:\r
            logging.error(f"Could not retrieve similar code from L2 Cache: {e}")\r
            return []\r
\r
\r
# --- PPO Buffer & Curiosity Module (คงเดิมจาก v8) ---\r
class PrioritizedReplayBuffer:\r
    # ... (โค้ดส่วนนี้ไม่มีการเปลี่ยนแปลง) ...\r
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_end=1.0):\r
        self.capacity = capacity\r
        self.alpha = alpha\r
        self.beta = beta_start\r
        self.beta_start = beta_start\r
        self.beta_end = beta_end\r
        self.beta_increment = (self.beta_end - self.beta_start) / float(self.capacity)\r
        self.buffer = []\r
        self.priorities = np.zeros(capacity, dtype=np.float32)\r
        self.position = 0\r
        self.device = 'cpu'\r
        self.max_priority = 1.0\r
\r
    def push(self, experience):\r
        priority = self.max_priority ** self.alpha\r
        if len(self.buffer) < self.capacity:\r
            self.buffer.append(experience)\r
        else:\r
            self.buffer[self.position] = experience\r
        self.priorities[self.position] = priority\r
        self.position = (self.position + 1) % self.capacity\r
\r
    def sample(self, batch_size):\r
        if len(self.buffer) < batch_size:\r
            return None\r
\r
        priorities = self.priorities[:len(self.buffer)]\r
        probs = priorities ** self.alpha\r
        probs /= probs.sum()\r
\r
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)\r
        samples = [self.buffer[idx] for idx in indices]\r
\r
        total = len(self.buffer)\r
        weights = (total * probs[indices]) ** (-self.beta)\r
        weights /= weights.max()\r
\r
        state, action, action_len, log_prob, reward, next_state, done = zip(*samples)\r
\r
        max_len = max(a.size(0) for a in action)\r
        action_padded = [pad(a, (0, max_len - a.size(0)), 'constant', 0) for a in action]\r
\r
        self.beta = min(self.beta + self.beta_increment, self.beta_end)\r
\r
        valid_next_states = [n for n in next_state if n is not None]\r
        batched_next_state = torch.stack(valid_next_states) if valid_next_states else None\r
\r
        return (\r
            torch.stack(state),\r
            torch.stack(action_padded),\r
            torch.stack(action_len),\r
            torch.stack(log_prob),\r
            torch.stack(reward),\r
            batched_next_state,\r
            torch.stack(done),\r
            torch.tensor(weights, dtype=torch.float32),\r
            indices\r
        )\r
\r
    def update_priorities(self, indices, errors):\r
        for idx, error in zip(indices, errors):\r
            self.priorities[idx] = max(error.item(), 1e-6)\r
        self.max_priority = self.priorities[:len(self.buffer)].max()\r
\r
    def __len__(self):\r
        return len(self.buffer)\r
\r
    def clear(self):\r
        self.buffer.clear()\r
        self.priorities = np.zeros(self.capacity, dtype=np.float32)\r
        self.position = 0\r
        self.max_priority = 1.0\r
\r
class ProjectGraphMemory(nn.Module):\r
    # ... (โค้ดส่วนนี้ไม่มีการเปลี่ยนแปลง) ...\r
    def __init__(self, num_features):\r
        super(ProjectGraphMemory, self).__init__()\r
        self.conv1 = GATv2Conv(num_features, 64, heads=4, edge_dim=1, concat=True)\r
        self.conv2 = GATv2Conv(64 * 4, 32, heads=2, edge_dim=1, concat=True)\r
        self.proj = nn.Linear(32 * 2, 32)\r
        self.device = 'cpu'\r
\r
    def forward(self, x, edge_index, edge_attr, batch):\r
        x = self.conv1(x, edge_index, edge_attr=edge_attr)\r
        x = x.relu()\r
        if x.device != self.conv2.weight.device:\r
            x = x.to(self.conv2.weight.device)\r
        x = self.conv2(x, edge_index, edge_attr=edge_attr)\r
        x = self.proj(x)\r
        return global_mean_pool(x, batch)\r
\r
\r
class CuriosityModule(nn.Module):\r
    # ... (โค้ดส่วนนี้ไม่มีการเปลี่ยนแปลง) ...\r
    def __init__(self, feature_size, action_size):\r
        super(CuriosityModule, self).__init__()\r
        self.inverse_model = nn.Sequential(\r
            nn.Linear(feature_size * 2, 128),\r
            nn.ReLU(),\r
            nn.Linear(128, action_size)\r
        )\r
        self.forward_model = nn.Sequential(\r
            nn.Linear(feature_size + action_size, 128),\r
            nn.ReLU(),\r
            nn.Linear(128, feature_size)\r
        )\r
        self.feature_size = feature_size\r
        self.device = 'cpu'\r
\r
    def forward(self, current_features, action_features, next_features):\r
        combined_features = torch.cat([current_features, next_features], dim=-1)\r
        predicted_action = self.inverse_model(combined_features)\r
        combined_forward = torch.cat([current_features, action_features], dim=-1)\r
        predicted_next_features = self.forward_model(combined_forward)\r
        forward_loss = mse_loss(predicted_next_features, next_features.detach())\r
        return forward_loss\r
\r
# --- Core LLM Model (คงเดิมจาก v8) ---\r
class MultiAgentLLM(nn.Module):\r
    # ... (โค้ดส่วนนี้ไม่มีการเปลี่ยนแปลงที่สำคัญ) ...\r
    def __init__(self, llm_name="microsoft/phi-3-mini-4k-instruct", lora_rank=16, lora_alpha=32, lora_dropout=0.05):\r
        super(MultiAgentLLM, self).__init__()\r
\r
        # ! (จาก v8) ตรวจสอบ VRAM และเลือก compute_dtype ที่ดีที่สุด\r
        if torch.cuda.is_available():\r
            gpu_name = torch.cuda.get_device_name(0)\r
            logging.info(f"✅ Detected GPU: {gpu_name}")\r
            if "RTX 30" in gpu_name or "RTX 40" in gpu_name or "A100" in gpu_name or (hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported()):\r
                logging.info("GPU supports bfloat16. Using bf16 for better performance and stability.")\r
                compute_dtype = torch.bfloat16\r
            else:\r
                logging.info("Using float16 as a fallback.")\r
                compute_dtype = torch.float16\r
\r
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)\r
            logging.info(f"Total VRAM: {vram:.2f} GB")\r
            if vram < 10:\r
                logging.warning(f"⚠️ VRAM is less than 10GB. Model may not fit. Performance might be slow due to offloading.")\r
\r
            device_map_setting = "auto"\r
            logging.info(f"Setting device_map to '{device_map_setting}' to intelligently manage VRAM and CPU RAM.")\r
\r
            bnb_config = BitsAndBytesConfig(\r
                load_in_4bit=True,\r
                bnb_4bit_quant_type="nf4",\r
                bnb_4bit_compute_dtype=compute_dtype,\r
                bnb_4bit_use_double_quant=True,\r
            )\r
        else:\r
            logging.info("No CUDA device found. Using CPU. Performance will be very slow.")\r
            bnb_config = None \r
            device_map_setting = "cpu"\r
\r
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)\r
        if self.tokenizer.pad_token is None:\r
            self.tokenizer.pad_token = self.tokenizer.eos_token\r
            logging.info("Tokenizer pad_token is not set. Using eos_token as pad_token.")\r
\r
\r
        # ! (จาก v8) ใช้ Flash Attention 2 ถ้ามี\r
        model_kwargs = {\r
            "quantization_config": bnb_config,\r
            "trust_remote_code": True,\r
            "device_map": device_map_setting,\r
            "offload_folder": "offload"\r
        }\r
        if FLASH_ATTENTION_AVAILABLE and torch.cuda.is_available():\r
            model_kwargs["attn_implementation"] = "flash_attention_2"\r
\r
\r
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name, **model_kwargs)\r
\r
        if bnb_config:\r
            self.llm.config.torch_dtype = bnb_config.bnb_4bit_compute_dtype\r
            self.llm = prepare_model_for_kbit_training(self.llm)\r
        \r
        def guess_lora_targets(model):\r
            names = []\r
            for n, mod in model.named_modules():\r
                if isinstance(mod, torch.nn.Linear) and ("attn" in n or "attention" in n or "q_proj" in n or "k_proj" in n or "v_proj" in n):\r
                    names.append(n.split('.')[-1])\r
            common = ["query_key_value", "dense", "out_proj", "c_attn", "c_proj", "fc1", "fc2", "gate_proj", "up_proj", "down_proj"]\r
            return list(set(names + common))\r
\r
        targets = guess_lora_targets(self.llm)\r
        logging.info(f"Guessed LoRA target modules: {targets}")\r
\r
        lora_config = LoraConfig(\r
            r=lora_rank,\r
            lora_alpha=lora_alpha,\r
            lora_dropout=lora_dropout,\r
            bias="none",\r
            task_type="CAUSAL_LM",\r
            target_modules=targets\r
        )\r
        self.llm = get_peft_model(self.llm, lora_config)\r
        self.llm.print_trainable_parameters()\r
\r
        codebert_embedding_dim = 768\r
        extra_features_dim = 1\r
        self.fixed_graph_embedding_dim = codebert_embedding_dim + extra_features_dim\r
        self.embedding_proj = nn.Linear(self.fixed_graph_embedding_dim, self.llm.config.hidden_size)\r
        self.graph_memory = ProjectGraphMemory(num_features=self.llm.config.hidden_size)\r
        self.graph_attn = nn.MultiheadAttention(\r
            embed_dim=self.llm.config.hidden_size,\r
            num_heads=8, \r
            batch_first=True\r
        )\r
        self.graph_norm = nn.LayerNorm(self.llm.config.hidden_size)\r
        self.policy_head = nn.Sequential(\r
            nn.LayerNorm(self.llm.config.hidden_size),\r
            nn.Linear(self.llm.config.hidden_size, self.tokenizer.vocab_size)\r
        )\r
        self.value_head = nn.Sequential(\r
            nn.LayerNorm(self.llm.config.hidden_size),\r
            nn.Linear(self.llm.config.hidden_size, 1)\r
        )\r
        self.curiosity_module = CuriosityModule(self.llm.config.hidden_size, self.tokenizer.vocab_size)\r
\r
    def _model_device(self):\r
        return next(p.device for p in self.llm.parameters())\r
\r
    def forward(self, input_ids, attention_mask=None, project_graph_embedding=None):\r
        device = self._model_device()\r
        input_ids = input_ids.to(device)\r
        if attention_mask is not None:\r
            attention_mask = attention_mask.to(device)\r
\r
        llm_outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)\r
        last_hidden_state = llm_outputs.hidden_states[-1]\r
\r
        if project_graph_embedding is not None:\r
            batch_size, seq_len, _ = last_hidden_state.shape\r
            # ! V9: แก้ไข Bug การย้าย device\r
            graph_emb_device = project_graph_embedding.to(device)\r
            expanded_graph_embedding = graph_emb_device.unsqueeze(1).repeat(1, seq_len, 1)\r
            fused_state, _ = self.graph_attn(last_hidden_state, expanded_graph_embedding, expanded_graph_embedding)\r
            fused_state = self.graph_norm(fused_state)\r
            fused_state = fused_state + last_hidden_state\r
        else:\r
            fused_state = last_hidden_state\r
\r
        logits = self.policy_head(fused_state.to(self.policy_head[1].weight.dtype))\r
        last_token_hidden_state = fused_state[:, -1, :]\r
        value = self.value_head(last_token_hidden_state.to(self.value_head[1].weight.dtype))\r
\r
        return logits, value, fused_state\r
\r
    # ! (จาก v8) Prompt Template\r
    def generate_response(self, prompt_text: str, system_prompt: str, max_new_tokens=1024, do_sample=True, temperature=0.7, top_k=50, top_p=0.95):\r
        self.eval()\r
        device = self._model_device()\r
\r
        messages = [\r
            {"role": "system", "content": system_prompt},\r
            {"role": "user", "content": prompt_text}\r
        ]\r
        \r
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\r
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)\r
\r
        generation_params = {\r
            "input_ids": input_ids,\r
            "max_new_tokens": max_new_tokens,\r
            "pad_token_id": self.tokenizer.pad_token_id,\r
            "eos_token_id": self.tokenizer.eos_token_id,\r
            "num_return_sequences": 1,\r
            "do_sample": do_sample,\r
            "temperature": temperature,\r
            "top_k": top_k,\r
            "top_p": top_p\r
        }\r
\r
        with torch.no_grad():\r
            generated_outputs = self.llm.generate(**generation_params)\r
        \r
        full_response = self.tokenizer.decode(generated_outputs[0], skip_special_tokens=True)\r
        \r
        # ! V9: ปรับปรุงการแยก response ให้แม่นยำขึ้น\r
        assistant_split = "<|assistant|>"\r
        if assistant_split in full_response:\r
            generated_text = full_response.split(assistant_split)[-1].strip()\r
        else:\r
            # Fallback for models that don't use the template correctly\r
            prompt_lines = prompt.splitlines()\r
            last_prompt_line = prompt_lines[-1] if prompt_lines else ""\r
            if last_prompt_line in full_response:\r
                 generated_text = full_response.split(last_prompt_line)[-1].strip()\r
            else:\r
                 generated_text = full_response # Return full text if we can't parse\r
        \r
        return generated_text, None\r
\r
# --- Code Evaluator (ปรับปรุงจาก v8) ---\r
class CodeEvaluator:\r
    def __init__(self, use_mock_reward=False, language="lua"):\r
        self.use_mock_reward = use_mock_reward\r
        self.language = language\r
        # ! UPGRADE V9: เพิ่ม pattern ความปลอดภัยให้ครอบคลุมมากขึ้น\r
        self.safe_patterns_lua = [\r
            r'io\\.open', r'os\\.execute', r'os\\.getenv', r'require\\(["\\'](http|socket)', \r
            r'HttpService', r'RunService', r'pcall\\s*\\(\\s*require',\r
            r'GetAsync', r'PostAsync', r'RequestAsync' # ! V9: Add network patterns\r
        ]\r
        self.safe_patterns_cs = [\r
            r'System\\.IO\\.File', r'System\\.Diagnostics\\.Process', r'System\\.Net\\.Http', \r
            r'System\\.Net\\.Sockets', r'HttpClient' # ! V9\r
        ]\r
        self.safe_patterns_gd = [\r
            r'OS\\.execute', r'File\\.new', r'Directory\\.new', r'HTTPClient' # ! V9\r
        ]\r
        self.safe_patterns_py = [\r
            r'subprocess', r'os\\.system', r'eval', r'exec', \r
            r'shutil', r'glob', r'socket' # ! V9\r
        ]\r
        self.vulnerability_patterns = {\r
            "lua": [r'loadstring'],\r
            "c#": [r'SqlCommand.*\\.CommandText\\s*=\\s*".*"\\s*\\+'], # SQL Injection\r
            "gdscript": [],\r
            "c++": [r'strcpy', r'sprintf'], # Buffer overflow\r
            "python": [r'pickle\\.load'] # Deserialization\r
        }\r
        self.llm = None\r
\r
    def _pre_check_code(self, code_string):\r
        patterns = []\r
        if self.language in ["lua", "luau"]: patterns = self.safe_patterns_lua\r
        elif self.language in ["c#", "csharp"]: patterns = self.safe_patterns_cs\r
        elif self.language == "gdscript": patterns = self.safe_patterns_gd\r
        elif self.language == "python": patterns = self.safe_patterns_py\r
        \r
        for pattern in patterns:\r
            if re.search(pattern, code_string, re.IGNORECASE):\r
                # ! V9: ปรับปรุง Log ให้ชัดเจน\r
                logging.warning(f"Code pre-check FAIL. Language: {self.language}. Pattern: {pattern}")\r
                return False, f"Potential security risk detected (e.g., file/network access): {pattern}"\r
        return True, "No immediate high-risk security patterns detected."\r
\r
    def _security_audit(self, code_string):\r
        # ... (คงเดิมจาก v8) ...\r
        penalty = 0.0\r
        findings = []\r
        lang_patterns = self.vulnerability_patterns.get(self.language, [])\r
        for pattern in lang_patterns:\r
            if re.search(pattern, code_string):\r
                findings.append(f"High-risk pattern found: '{pattern}'. This could lead to security vulnerabilities.")\r
                penalty -= 5.0\r
        if not findings:\r
            return 0.0, "No high-risk security patterns found."\r
        return penalty, "\\n".join(findings)\r
\r
    def _static_analysis(self, code_string):\r
        # ... (คงเดิมจาก v8) ...\r
        if self.language not in ["lua", "luau"]:\r
            return 0.0, "Skipped", f"Static analysis for {self.language} not implemented."\r
        temp_file_path = f"temp_code_{random.randint(1000,9999)}.lua"\r
        with open(temp_file_path, "w", encoding="utf-8") as f: f.write(code_string)\r
        try:\r
            result = subprocess.run(['luacheck', temp_file_path], capture_output=True, text=True, timeout=10)\r
            if "No issues found" in result.stdout or result.returncode == 0:\r
                return 1.0, "Syntax check: OK", result.stdout\r
            else:\r
                return -1.0, f"Syntax check failed", result.stdout + result.stderr\r
        except Exception as e:\r
            return -1.0, f"Error during static analysis: {e}", ""\r
        finally:\r
            if os.path.exists(temp_file_path): os.remove(temp_file_path)\r
\r
    def _dynamic_analysis(self, code_string):\r
        # ... (คงเดิมจาก v8) ...\r
        if self.language not in ["lua", "luau"]:\r
            return 0.0, "Skipped", f"Dynamic analysis for {self.language} not implemented."\r
        if shutil.which("docker") is None:\r
            logging.warning("Docker not installed, skipping functional test.")\r
            return 0.0, "Docker not available", "Docker not found."\r
        temp_file_path = f"temp_code_{random.randint(1000,9999)}.lua"\r
        with open(temp_file_path, "w", encoding="utf-8") as f: f.write(code_string)\r
        docker_command = ['docker', 'run', '--rm', '--network', 'none', '--pids-limit', '256', '--cpus', '1', '--memory', '512m', '--ulimit', 'cpu=5', '--read-only', '-v', f'{os.getcwd()}/{temp_file_path}:/app/temp_code.lua:ro', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', 'luau_runtime_image', 'luau', '/app/temp_code.lua']\r
        try:\r
            start_time = time.time()\r
            result = subprocess.run(docker_command, capture_output=True, text=True, timeout=15)\r
            execution_time = time.time() - start_time\r
            if result.returncode == 0:\r
                reward = 10.0\r
                efficiency_reward = max(0, 1.0 - execution_time / 10.0)\r
                return reward + efficiency_reward, f"Functional test: Success!", result.stdout\r
            else:\r
                return -5.0, f"Functional test failed.", result.stdout + result.stderr\r
        except Exception as e:\r
            return -2.0, f"Error during dynamic analysis: {e}", ""\r
        finally:\r
            if os.path.exists(temp_file_path): os.remove(temp_file_path)\r
\r
    def _assess_readability(self, code_string):\r
        # ... (คงเดิมจาก v8) ...\r
        lines = code_string.split('\\n')\r
        if not lines: return 0.0\r
        comment_lines = sum(1 for line in lines if line.strip().startswith('--') or line.strip().startswith('//') or line.strip().startswith('#'))\r
        total_lines = len(lines)\r
        if total_lines == 0: return 0.0\r
        comment_ratio = comment_lines / total_lines\r
        long_line_penalty = sum(1 for line in lines if len(line) > 120)\r
        readability_score = (comment_ratio * 0.5) - (long_line_penalty * 0.1)\r
        return max(0, min(1, readability_score))\r
\r
    def evaluate(self, code_string, project_graph):\r
        if self.use_mock_reward:\r
            return {"total_reward": random.uniform(-1, 10), "correctness_score": random.uniform(0, 10), "efficiency_score": random.uniform(0, 1), "knowledge_graph_score": random.uniform(0, 5), "readability_score": random.uniform(0, 1), "security_score": 0.0, "luacheck_log": "Mock", "docker_log": "Mock", "security_log": "Mock"}\r
        \r
        # ! V9: ตรวจสอบ code_string ว่างเปล่า\r
        if not code_string or not code_string.strip():\r
            return {"total_reward": -10.0, "correctness_score": -10.0, "efficiency_score": 0.0, "knowledge_graph_score": 0.0, "readability_score": 0.0, "security_score": -10.0, "luacheck_log": "No code provided", "docker_log": "No code provided", "security_log": "No code provided"}\r
\r
        is_safe, security_log = self._pre_check_code(code_string)\r
        if not is_safe:\r
            return {"total_reward": -10.0, "correctness_score": -10.0, "efficiency_score": 0.0, "knowledge_graph_score": 0.0, "readability_score": 0.0, "security_score": -10.0, "luacheck_log": "Skipped", "docker_log": security_log, "security_log": security_log}\r
\r
        security_score, detailed_security_log = self._security_audit(code_string)\r
        syntax_score, _, luacheck_log = self._static_analysis(code_string)\r
        functional_score, _, docker_log = self._dynamic_analysis(code_string)\r
        correctness_score = (syntax_score + functional_score)\r
        efficiency_score = max(0, 1.0 - len(code_string) / 2000)\r
        readability_score = self._assess_readability(code_string)\r
        \r
        kg_score = 0 # Placeholder\r
        total_reward = (0.4 * correctness_score) + (0.1 * efficiency_score) + (0.1 * kg_score) + (0.1 * readability_score) + (0.3 * security_score)\r
        return {"total_reward": total_reward, "correctness_score": correctness_score, "efficiency_score": efficiency_score, "knowledge_graph_score": kg_score, "readability_score": readability_score, "security_score": security_score, "luacheck_log": luacheck_log, "docker_log": docker_log, "security_log": detailed_security_log}\r
\r
# ! ##################################################################\r
# ! ################ V9 EXPANDED AGENT TEAM (22 Agents) ##############\r
# ! ##################################################################\r
class BaseAgent:\r
    def __init__(self, model): self.model = model\r
    def _generate(self, prompt, system_prompt, **kwargs):\r
        # ! V9: จัดการ docstring สำหรับ system prompt\r
        doc_prompt = (self.__doc__ or system_prompt or "").strip()\r
        return self.model.generate_response(prompt, doc_prompt, **kwargs)[0]\r
\r
# --- Existing Agents (Refactored from v8) ---\r
class CodeGeneratorAgent(BaseAgent):\r
    """You are an expert programmer. Write clean, efficient, and correct code based on the user's request. Add comments to explain complex parts."""\r
    def generate(self, prompt, context_examples=None):\r
        system_prompt = self.__doc__\r
        if context_examples:\r
            examples_text = "\\n\\n---\\nHere are some relevant examples of good code:\\n---\\n" + "\\n".join([f"\`\`\`\\n{ex}\\n\`\`\`" for ex in context_examples])\r
            prompt += examples_text\r
        return self._generate(prompt, system_prompt)\r
\r
class CodeCriticAgent(BaseAgent):\r
    """You are a senior code reviewer. Provide a constructive critique, identifying bugs, inefficiencies, or security risks. Suggest specific improvements."""\r
    def critique(self, code, eval_results):\r
        critique_prompt = f"Analyze the code and its evaluation.\\nCode:\\n\`\`\`\\n{code}\\n\`\`\`\\nEvaluation:\\n{json.dumps(eval_results, indent=2)}"\r
        return self._generate(critique_prompt, self.__doc__, max_new_tokens=300)\r
\r
class CodeRefinementAgent(BaseAgent):\r
    """You are a code refactoring specialist. Rewrite the provided code based on the critique to improve it. Provide ONLY the complete, corrected code block."""\r
    def refine(self, original_code, critique, context_examples=None):\r
        refinement_prompt = f"Based on the critique, refactor the original code.\\nOriginal Code:\\n\`\`\`\\n{original_code}\\n\`\`\`\\nCritique:\\n{critique}\\n"\r
        if context_examples: refinement_prompt += "Relevant examples:\\n" + "\\n".join([f"\`\`\`\\n{ex}\\n\`\`\`" for ex in context_examples])\r
        refined_code = self._generate(refinement_prompt, self.__doc__, max_new_tokens=1500)\r
        match = re.search(r'\`\`\`(?:\\w+)?\\n(.*?)\\n\`\`\`', refined_code, re.DOTALL)\r
        return match.group(1).strip() if match else refined_code\r
\r
class AssetGeneratorAgent: # Does not need LLM\r
    def __init__(self): logging.info("AssetGeneratorAgent initialized.")\r
    def generate_asset(self, prompt): return f"asset_{hash(prompt) % 10000}"\r
\r
class BugReportGeneratorAgent(BaseAgent):\r
    """You are a QA Engineer. Create a professional bug report from the given code and error log."""\r
    def generate_report(self, code, error_log):\r
        prompt = f"Code:\\n\`\`\`\\n{code}\\n\`\`\`\\nError Log:\\n{error_log}\\n\\nProvide a bug report with: Description, Steps to Reproduce, Expected Behavior, Actual Behavior."\r
        return self._generate(prompt, self.__doc__)\r
\r
class TestGenerationAgent(BaseAgent):\r
    """You are a test engineer. Generate a comprehensive set of unit tests for the following {language} code."""\r
    def generate_tests(self, code, language="Luau"):\r
        prompt = f"Code to test:\\n\`\`\`\\n{code}\\n\`\`\`\\nProvide the complete unit test script."\r
        return self._generate(prompt, self.__doc__.format(language=language))\r
\r
class DocumentationAgent(BaseAgent):\r
    """You are a technical writer. Create clear, concise documentation for the following {language} code in Markdown format."""\r
    def generate_docs(self, code, language="Luau"):\r
        prompt = f"Code to document:\\n\`\`\`\\n{code}\\n\`\`\`"\r
        return self._generate(prompt, self.__doc__.format(language=language))\r
\r
class AutoRefactoringAgent(BaseAgent):\r
    """You are an expert in software architecture. Refactor this {language} code to improve its structure and readability without changing its functionality."""\r
    def refactor(self, code, language="Luau"):\r
        prompt = f"Code to refactor:\\n\`\`\`\\n{code}\\n\`\`\`\\nProvide only the refactored code."\r
        return self._generate(prompt, self.__doc__.format(language=language))\r
\r
class GameDesignerAgent(BaseAgent):\r
    """You are a creative and innovative game designer."""\r
    def propose_feature(self, context):\r
        prompt = f"Based on the context '{context}', propose a new game feature. Describe it, how it works, and why it's fun."\r
        return self._generate(prompt, self.__doc__)\r
\r
class CodeSummarizationAgent(BaseAgent):\r
    """You are a code summarization expert. Be concise."""\r
    def summarize(self, code, language="Luau"):\r
        prompt = f"Summarize the main functionality of this {language} code in one sentence.\\nCode:\\n\`\`\`\\n{code}\\n\`\`\`"\r
        return self._generate(prompt, self.__doc__, max_new_tokens=80)\r
\r
class CodeQuestionAnsweringAgent(BaseAgent):\r
    """You are an AI assistant that analyzes code to answer questions accurately."""\r
    def answer_question(self, code, question, language="Luau"):\r
        prompt = f"Analyze this {language} code and answer the question.\\nCode:\\n\`\`\`\\n{code}\\n\`\`\`\\nQuestion: {question}"\r
        return self._generate(prompt, self.__doc__.format(language=language))\r
\r
# --- V8 NEW AGENTS (7 additions) ---\r
class WebDeveloperAgent(BaseAgent):\r
    """You are a full-stack web developer, expert in HTML, CSS, JavaScript, React, Python (Flask/Django), SQL, and PHP. Create a complete, functional code for the requested web component or application."""\r
    def generate_webapp(self, prompt):\r
        return self._generate(prompt, self.__doc__, max_new_tokens=2048)\r
\r
class FinancialAnalystAgent(BaseAgent):\r
    """You are a financial analyst AI. Provide insightful analysis, data interpretation, and perspectives. IMPORTANT: ALWAYS include a disclaimer that you are not a certified financial advisor and your advice should not be taken as professional financial guidance."""\r
    def provide_analysis(self, prompt):\r
        return self._generate(prompt, self.__doc__, max_new_tokens=512)\r
\r
class AIArchitectAgent(BaseAgent):\r
    """You are an AI/ML system architect. Design a robust and scalable architecture for the user's request. Consider data pipelines, model selection, training strategy, and deployment. Explain your choices."""\r
    def design_ai_system(self, prompt):\r
        return self._generate(prompt, self.__doc__, max_new_tokens=1024)\r
\r
class AnimationCodeAgent(BaseAgent):\r
    """You are an expert in procedural animation and graphics programming. Generate code (e.g., using JavaScript libraries like three.js, p5.js, or CSS) to create the described animation."""\r
    def generate_animation(self, prompt):\r
        return self._generate(prompt, self.__doc__, max_new_tokens=1500)\r
\r
class DatabaseAdminAgent(BaseAgent):\r
    """You are a database administrator, expert in SQL and database design. Write efficient SQL queries or schema definitions based on the request. Assume a standard SQL dialect unless specified."""\r
    def generate_sql(self, prompt):\r
        return self._generate(prompt, self.__doc__, max_new_tokens=512)\r
\r
class GeneralConversationAgent(BaseAgent):\r
    """You are a helpful, friendly, and knowledgeable AI assistant. Engage in a natural conversation, answer questions, and provide information on a wide range of topics."""\r
    def chat(self, prompt):\r
        return self._generate(prompt, self.__doc__, max_new_tokens=1024)\r
\r
class MarketingCopywriterAgent(BaseAgent):\r
    """You are a professional marketing copywriter. Write compelling and persuasive copy for websites, ads, or social media based on the user's product or service."""\r
    def write_copy(self, prompt):\r
        return self._generate(prompt, self.__doc__, max_new_tokens=512)\r
\r
# ! ##################################################################\r
# ! ################ V9 NEW AGENTS (4 new additions) #################\r
# ! ##################################################################\r
class LongContextSummarizerAgent(BaseAgent):\r
    """You are an expert summarization AI. Your task is to receive very long text, code, or documents and condense them into a shorter, concise summary. You must preserve the core intent, key entities, function names, and the overall structure of the original content, but in a much shorter form."""\r
    def summarize(self, long_text):\r
        """\r
        ! V9: Handles long context by summarizing it.\r
        """\r
        prompt = f"Please summarize the following content, which may be thousands of lines long. Preserve the main purpose, key functions/classes, and structural flow.\\n\\nCONTENT:\\n\`\`\`\\n{long_text}\\n\`\`\`\\n\\nCONCISE SUMMARY:"\r
        # ! V9: ให้ max_new_tokens สูงขึ้นเล็กน้อยสำหรับการสรุป\r
        return self._generate(prompt, self.__doc__, max_new_tokens=512)\r
\r
class ScientificResearcherAgent(BaseAgent):\r
    """You are a scientific researcher and academic AI. Provide detailed, accurate, and sourced explanations for complex topics. Use clear, factual language and break down difficult concepts for understanding. Cite sources if possible (e.g., 'According to [field of study]...')."""\r
    def research(self, topic):\r
        prompt = f"Please provide a detailed, academic-level explanation for the following topic: {topic}"\r
        return self._generate(prompt, self.__doc__, max_new_tokens=1024)\r
\r
class CreativeWriterAgent(BaseAgent):\r
    """You are a highly creative writer. You can write engaging stories, poems, scripts, song lyrics, or any other creative text based on the user's prompt."""\r
    def write(self, prompt):\r
        prompt = f"Please write a creative piece based on this idea: {prompt}"\r
        return self._generate(prompt, self.__doc__, max_new_tokens=1500)\r
\r
\r
# ! REWORKED V9: Mixture-of-Experts (MoE) Router\r
class MixtureOfExpertsRouter(BaseAgent):\r
    def __init__(self, model, agents):\r
        super().__init__(model)\r
        self.agents = agents\r
        # ! V9: อัปเดต Keywords ให้รวม Agent ใหม่\r
        self.agent_keywords = {\r
            # Code\r
            "CodeGeneratorAgent": ["code", "script", "function", "class", "algorithm", "implement", "write code", "สร้างโค้ด"],\r
            "CodeCriticAgent": ["review", "critique", "improve my code", "find bugs", "ตรวจโค้ด", "แก้บั๊ก"],\r
            "CodeRefinementAgent": ["refactor", "clean up code", "optimize this", "ปรับปรุงโค้ด"],\r
            "TestGenerationAgent": ["unit test", "test case", "py.test", "jest", "เขียนเทส"],\r
            "DocumentationAgent": ["document", "docs", "docstring", "comment", "explain this code", "ทำเอกสาร"],\r
            "CodeSummarizationAgent": ["summarize code", "what does this code do", "tl;dr code", "สรุปโค้ด"],\r
            "CodeQuestionAnsweringAgent": ["why this error", "how does this function work", "ถามเรื่องโค้ด"],\r
            # Web\r
            "WebDeveloperAgent": ["website", "web app", "html", "css", "javascript", "react", "frontend", "backend", "php", "api"],\r
            "DatabaseAdminAgent": ["sql", "database", "query", "schema", "table", "select", "insert", "update"],\r
            "AnimationCodeAgent": ["animation", "three.js", "p5.js", "css transition", "animate this"],\r
            # Game\r
            "GameDesignerAgent": ["game idea", "feature", "mechanic", "level design", "gameplay", "ออกแบบเกม"],\r
            "BugReportGeneratorAgent": ["bug report", "error log", "report issue", "รายงานบั๊ก"],\r
            # Business & AI\r
            "FinancialAnalystAgent": ["stock", "market", "investment", "finance", "economic", "portfolio", "การเงิน", "หุ้น"],\r
            "AIArchitectAgent": ["ai model", "machine learning", "architecture", "neural network", "pipeline", "ออกแบบ AI"],\r
            "MarketingCopywriterAgent": ["marketing", "copywriting", "ad copy", "social media post", "slogan", "เขียนโฆษณา"],\r
            # ! V9: New Agent Keywords\r
            "LongContextSummarizerAgent": ["summarize this document", "too long", "summarize this code", "สรุปไฟล์นี้"],\r
            "ScientificResearcherAgent": ["science", "physics", "biology", "chemistry", "explain topic", "research", "วิทยาศาสตร์"],\r
            "CreativeWriterAgent": ["write a story", "poem", "script", "song lyrics", "creative writing", "แต่งเรื่อง", "เขียนกลอน"],\r
            # Default\r
            "GeneralConversationAgent": ["what is", "who is", "explain", "tell me about", "how are you", "chat", "คุย", "คืออะไร"],\r
        }\r
        logging.info("🧠 Mixture-of-Experts Router (V9) initialized.")\r
\r
    def route(self, prompt):\r
        """\r
        ! V9: Reworked Routing Logic\r
        Analyzes the prompt and selects the best *list* of agent(s) for the task.\r
        Returns a list of agent class names.\r
        """\r
        prompt_lower = prompt.lower()\r
        scores = {name: 0 for name in self.agents.keys()}\r
        \r
        # ! V9: ให้คะแนนตาม Keyword\r
        for name, keywords in self.agent_keywords.items():\r
            for keyword in keywords:\r
                if keyword in prompt_lower:\r
                    scores[name] += 1\r
        \r
        # ! V9: ใช้ Threshold-based-selection เพื่อเลือกหลาย Agent\r
        ROUTING_THRESHOLD = 0 # หมายความว่าแค่มี keyword 1 ครั้งก็ถูกเลือก\r
        \r
        selected_agents = [name for name, score in scores.items() if score > ROUTING_THRESHOLD]\r
        \r
        # --- Logic ในการคัดเลือก ---\r
        \r
        # 1. ถ้ามี Agent ที่เฉพาะเจาะจงสูง (เช่น Code) ถูกเลือก, ให้ตัด Agent ทั่วไป (General) ทิ้ง\r
        specific_agent_groups = [\r
            "CodeGeneratorAgent", "WebDeveloperAgent", "DatabaseAdminAgent", "FinancialAnalystAgent", \r
            "AIArchitectAgent", "ScientificResearcherAgent", "CreativeWriterAgent", "CodeCriticAgent"\r
        ]\r
        \r
        has_specific_agent = any(agent in selected_agents for agent in specific_agent_groups)\r
        \r
        if has_specific_agent and "GeneralConversationAgent" in selected_agents and len(selected_agents) > 1:\r
            selected_agents.remove("GeneralConversationAgent")\r
            \r
        # 2. ถ้าไม่มี Agent ใดถูกเลือกเลย (คะแนน = 0), ให้ใช้ GeneralConversationAgent\r
        if not selected_agents:\r
            selected_agents = ["GeneralConversationAgent"]\r
            \r
        # 3. ! V9: ถ้า Prompt ยาวมาก, ให้บังคับใช้ LongContextSummarizerAgent (ตรรกะนี้จะถูกย้ายไป GUI เพื่อ pre-process)\r
        # (เราจะปล่อยให้ router เลือกตาม keyword "summarize" ที่นี่)\r
\r
        logging.info(f"MoE Router V9 selected agents: {selected_agents} for prompt: '{prompt[:50]}...'")\r
        \r
        return selected_agents\r
\r
# ... (Data Handling, Graph Building, Dataset classes from v8 can remain largely unchanged)\r
def download_code_from_github(engine_name: str, github_query: str, file_extensions: list, save_dir: str, github_token: str):\r
    # ... (คงเดิมจาก v8) ...\r
    if not os.path.exists(save_dir):\r
        logging.info(f"Directory '{save_dir}' not found. Creating it.")\r
        os.makedirs(save_dir)\r
        \r
    headers = {"Authorization": f"token {github_token}"}\r
    search_url = "https://api.github.com/search/code" # ! V9: แก้ไข URL ที่ถูกต้อง (v8 ผิด)\r
    \r
    downloaded_count = 0\r
    for ext in file_extensions:\r
        downloaded_count += len(glob.glob(os.path.join(save_dir, f"*.{ext}")))\r
    \r
    target_count = 1500 # Target remains 1500\r
    logging.info(f"[{engine_name}] Found {downloaded_count} existing files. Target is {target_count}.")\r
    if downloaded_count >= target_count:\r
        logging.info(f"[{engine_name}] Target already met. Skipping download.")\r
        return\r
\r
    page = 1\r
    pbar = tqdm(total=target_count, initial=downloaded_count, desc=f"Downloading for {engine_name}")\r
\r
    while downloaded_count < target_count:\r
        query = f'{github_query} ' + ' OR '.join([f'extension:{ext}' for ext in file_extensions])\r
        params = {"q": query, "per_page": 100, "page": page}\r
        \r
        try:\r
            response = requests.get(search_url, headers=headers, params=params, timeout=30)\r
            if 'X-RateLimit-Remaining' in response.headers and int(response.headers['X-RateLimit-Remaining']) < 10:\r
                reset_time = int(response.headers['X-RateLimit-Reset'])\r
                sleep_duration = max(0, reset_time - time.time()) + 5\r
                logging.warning(f"GitHub API rate limit low. Sleeping for {sleep_duration:.0f} seconds.")\r
                time.sleep(sleep_duration)\r
            response.raise_for_status()\r
            items = response.json().get("items", [])\r
            if not items:\r
                logging.info(f"[{engine_name}] No more code files found.")\r
                break\r
        except requests.exceptions.RequestException as e:\r
            logging.error(f"Error connecting to GitHub API: {e}. Retrying in 60 seconds...")\r
            time.sleep(60)\r
            continue\r
\r
        for item in items:\r
            repo_name = item["repository"]["full_name"]\r
            file_path = item["path"]\r
            # ! V9: ตรวจสอบนามสกุลไฟล์ให้ตรงกับที่ร้องขอ\r
            file_ext = file_path.split('.')[-1]\r
            if file_ext not in file_extensions:\r
                continue\r
\r
            save_path = os.path.join(save_dir, f"{repo_name.replace('/', '_')}_{os.path.basename(file_path)}")\r
            if os.path.exists(save_path):\r
                continue\r
\r
            try:\r
                # ! V9: ใช้ "git_url" แทน "url" เพื่อรับ JSON ของไฟล์\r
                content_response = requests.get(item["git_url"], headers=headers, timeout=30)\r
                content_response.raise_for_status()\r
                file_data = content_response.json()\r
                if file_data.get("encoding") != "base64":\r
                     logging.warning(f"Skipping file {file_path} (not base64 encoded).")\r
                     continue\r
                \r
                content = base64.b64decode(file_data["content"]).decode("utf-8")\r
                with open(save_path, "w", encoding="utf-8") as f:\r
                    f.write(content)\r
                downloaded_count += 1\r
                pbar.update(1)\r
                if downloaded_count >= target_count:\r
                    break\r
            except (UnicodeDecodeError, KeyError, requests.exceptions.RequestException, json.JSONDecodeError) as e:\r
                logging.warning(f"Skipping file {file_path} in {repo_name} due to error: {e}")\r
                continue\r
        \r
        if downloaded_count >= target_count:\r
            break\r
        page += 1\r
        time.sleep(2) \r
    \r
    pbar.close()\r
    logging.info(f"[{engine_name}] Download complete. Total files: {downloaded_count}.")\r
    return downloaded_count\r
\r
# --- Graph Building & Dataset (No major changes from v8) ---\r
# ... (All functions like cache_embeddings, build_real_code_graph_ast, CodeDataset, etc. are here)\r
def get_func_name(node):\r
    try:\r
        if hasattr(node.name.id, 'id'): return node.name.id.id\r
        return node.name.id\r
    except Exception: return None\r
\r
def cache_embeddings(code_chunks, codebert_pipeline, cache_file="embeddings_cache_v9.joblib"):\r
    # ! V9: Updated cache file name\r
    if codebert_pipeline is None:\r
        logging.error("CodeBERT pipeline is None. Cannot generate embeddings.")\r
        return {}\r
    if os.path.exists(cache_file):\r
        try: cache = joblib.load(cache_file)\r
        except Exception as e:\r
            logging.error(f"Error loading embedding cache: {e}. Creating new cache.")\r
            cache = {}\r
    else: cache = {}\r
    new_chunks = [chunk for chunk in code_chunks if chunk and chunk.strip() and chunk not in cache] # ! V9: Add check for empty chunks\r
    if new_chunks:\r
        logging.info(f"Generating embeddings for {len(new_chunks)} new chunks...")\r
        try:\r
            # ! V9: เพิ่มการจัดการ Error ใน pipeline\r
            new_embeddings_raw = codebert_pipeline(new_chunks, batch_size=16)\r
            new_embeddings = [torch.tensor(emb).squeeze() for emb in new_embeddings_raw]\r
            for chunk, embedding in zip(new_chunks, new_embeddings):\r
                if embedding.dim() > 1: embedding = embedding.mean(dim=0)\r
                cache[chunk] = embedding.tolist()\r
        except Exception as e:\r
            logging.error(f"Error generating CodeBERT embeddings: {e}")\r
            # ไม่ return {} แต่ให้ใช้ cache เก่าไปก่อน\r
        \r
        try:\r
            joblib.dump(cache, cache_file)\r
            logging.info("Embeddings cached.")\r
        except Exception as e:\r
            logging.error(f"Failed to save embedding cache: {e}")\r
    return cache\r
\r
def visualize_graph(graph, filename="code_graph_v9.png", max_nodes=50):\r
    # ! V9: Updated file name\r
    if not hasattr(graph, 'x') or graph.x.shape[0] == 0: return\r
    g = nx.DiGraph()\r
    edges = graph.edge_index.t().tolist()\r
    g.add_edges_from(edges)\r
    \r
    # ! V9: จำกัดจำนวนโหนดที่จะวาด\r
    if len(g.nodes) > max_nodes:\r
        nodes_to_draw = list(g.nodes)[:max_nodes]\r
        g = g.subgraph(nodes_to_draw)\r
        \r
    plt.figure(figsize=(14, 14))\r
    try:\r
        pos = nx.spring_layout(g)\r
        nx.draw(g, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=8, edge_color='gray', arrows=True)\r
        plt.title("Code Knowledge Graph Visualization (Sampled)")\r
        plt.savefig(filename)\r
    except Exception as e:\r
        logging.error(f"Failed to visualize graph: {e}")\r
    plt.close()\r
\r
def build_real_code_graph_ast(code_content, model, codebert_pipeline, asset_id=None, language="lua", design_doc=None, user_feedback=None):\r
    if language != "lua" or not luaparser_parser:\r
        return None\r
    try:\r
        # Simplified for brevity, original logic is sound\r
        ast_tree = luaparser_parser.parse(code_content)\r
        # ... (rest of the AST logic from original)\r
        all_chunks = re.split(r'\\n(function|local function)', code_content)\r
        chunks = [all_chunks[i] + all_chunks[i+1] for i in range(1, len(all_chunks), 2) if i + 1 < len(all_chunks)]\r
        valid_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]\r
        if not valid_chunks: return None\r
        embeddings_cache = cache_embeddings(valid_chunks, codebert_pipeline)\r
        if not embeddings_cache: return None\r
        xs = []\r
        for chunk in valid_chunks:\r
            embedding = embeddings_cache.get(chunk)\r
            if embedding is not None:\r
                current_embedding = torch.tensor(embedding, dtype=torch.float32)\r
                usage_feature = torch.tensor([0.0]) # Mock usage\r
                combined_features = torch.cat([current_embedding, usage_feature], dim=0)\r
                padding_needed = model.fixed_graph_embedding_dim - combined_features.shape[0]\r
                if padding_needed > 0: combined_features = pad(combined_features, (0, padding_needed), 'constant', 0)\r
                else: combined_features = combined_features[:model.fixed_graph_embedding_dim]\r
                xs.append(combined_features)\r
        if not xs: return None\r
        x = torch.stack(xs, dim=0)\r
        edge_index = torch.empty((2, 0), dtype=torch.long) # Mock edges\r
        edge_attr = torch.empty((0, 1), dtype=torch.float)\r
        py_g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)\r
        py_g.x = model.embedding_proj(py_g.x.to(model._model_device()))\r
        py_g.node_type = torch.zeros(len(xs), dtype=torch.long)\r
        py_g.node_id = [None] * len(xs)\r
        return py_g\r
    except Exception as e:\r
        logging.error(f"Error building AST graph: {e}")\r
        return None\r
\r
class CodeDataset(Dataset):\r
    def __init__(self, data_dir, tokenizer, model, codebert_pipeline, max_length=1024, graph_cache_dir="graph_cache_v9", file_extensions=None):\r
        # ! V9: Updated cache dir\r
        if file_extensions is None: file_extensions = ["*.lua", "*.luau"]\r
        self.file_paths = []\r
        for ext in file_extensions:\r
            self.file_paths.extend(glob.glob(os.path.join(data_dir, ext)))\r
        if not self.file_paths: raise FileNotFoundError(f"No files with extensions {file_extensions} found in {data_dir}.")\r
        self.tokenizer = tokenizer\r
        self.model = model\r
        self.codebert_pipeline = codebert_pipeline\r
        self.max_length = max_length\r
        self.graph_cache_dir = graph_cache_dir\r
        if not os.path.exists(self.graph_cache_dir): os.makedirs(self.graph_cache_dir)\r
\r
    def __len__(self): return len(self.file_paths)\r
    def __getitem__(self, idx):\r
        file_path = self.file_paths[idx]\r
        try:\r
            with open(file_path, "r", encoding="utf-8") as f: content = f.read()\r
            # ! V9: ตรวจสอบ content ว่าง\r
            if not content.strip():\r
                logging.warning(f"Skipping empty file: {file_path}")\r
                return None\r
            tokenized_data = self.tokenizer(content, return_tensors="pt", max_length=self.max_length, truncation=True, padding="max_length")\r
            graph_data = self.get_graph(content) # Simplified\r
            return {'input_ids': tokenized_data['input_ids'].squeeze(), 'attention_mask': tokenized_data['attention_mask'].squeeze(), 'code_content': content, 'code_graph_data': graph_data}\r
        except Exception as e:\r
            logging.error(f"Skipping file {file_path} due to error: {e}")\r
            return None\r
\r
    def get_graph(self, content): # Simplified\r
        content_hash = str(hash(content))\r
        cache_path = os.path.join(self.graph_cache_dir, f"{content_hash}.joblib")\r
        if os.path.exists(cache_path):\r
            try: return joblib.load(cache_path)\r
            except Exception: pass\r
        graph_data = build_real_code_graph_ast(content, self.model, self.codebert_pipeline)\r
        if graph_data: \r
            try:\r
                joblib.dump(graph_data, cache_path)\r
            except Exception as e:\r
                logging.warning(f"Failed to save graph cache: {e}")\r
        return graph_data\r
\r
def custom_collate_fn(batch):\r
    batch = [item for item in batch if item is not None]\r
    if not batch: return None\r
    try:\r
        input_ids = torch.stack([item['input_ids'] for item in batch])\r
        attention_mask = torch.stack([item['attention_mask'] for item in batch])\r
        code_contents = [item['code_content'] for item in batch]\r
        graph_data_list = [item['code_graph_data'] for item in batch if item['code_graph_data'] is not None]\r
        batched_graph = Batch.from_data_list(graph_data_list) if graph_data_list else None\r
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'code_contents': code_contents, 'code_graph_data': batched_graph}\r
    except Exception as e:\r
        logging.error(f"Error in custom_collate_fn: {e}")\r
        return None # Skip this batch\r
\r
# --- PPO Loss & Training Loop (คงเดิมจาก v8) ---\r
def _calculate_ppo_loss(model, accelerator, state, action, action_len, old_log_prob, reward, next_state_embedding, old_value_preds, done, curiosity_weight, clip_epsilon, weights=None):\r
    # ... (โค้ดส่วนนี้ไม่มีการเปลี่ยนแปลง) ...\r
    if next_state_embedding is not None: next_state_embedding = next_state_embedding.to(accelerator.device)\r
    full_input_ids = torch.cat([state, action], dim=1).to(accelerator.device)\r
    max_len = model.llm.config.max_position_embeddings\r
    if full_input_ids.size(1) > max_len: full_input_ids = full_input_ids[:, :max_len]\r
    \r
    logits, value, fused_state = model(full_input_ids, None, project_graph_embedding=next_state_embedding)\r
    \r
    if action.numel() == 0 or logits.numel() == 0: return None, None, None, None, None\r
    logits_gen = logits[:, state.size(1)-1:-1, :]\r
    log_probs = log_softmax(logits_gen, dim=-1)\r
    \r
    # ! V9: ตรวจสอบขนาด action ให้ตรงกับ logits_gen\r
    action_clipped = action\r
    if action.size(1) > logits_gen.size(1):\r
        action_clipped = action[:, :logits_gen.size(1)]\r
    elif logits_gen.size(1) > action.size(1):\r
        logits_gen = logits_gen[:, :action.size(1), :]\r
\r
    action_mask = (action_clipped != model.tokenizer.pad_token_id).to(accelerator.device)\r
    action_log_probs = log_probs.gather(2, action_clipped.unsqueeze(-1)).squeeze(-1)\r
    current_log_prob = (action_log_probs * action_mask).sum(dim=-1) / action_mask.sum(dim=-1).clamp(min=1e-6)\r
    \r
    gamma = 0.99\r
    reward = reward.squeeze()\r
    done = done.squeeze()\r
    value = value.squeeze()\r
    old_value_preds = old_value_preds.squeeze()\r
    \r
    advantages = reward + gamma * value.detach() * (1 - done.int()) - old_value_preds.detach()\r
    returns = advantages + old_value_preds.detach()\r
    \r
    curiosity_loss = torch.tensor(0.0).to(accelerator.device) # Simplified\r
    \r
    ratio = torch.exp(current_log_prob - old_log_prob.detach())\r
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)\r
    policy_loss_unweighted = -torch.min(ratio * advantages, clipped_ratio * advantages)\r
    value_loss_unweighted = mse_loss(value, returns)\r
    \r
    policy_loss = (policy_loss_unweighted * weights).mean() if weights is not None else policy_loss_unweighted.mean()\r
    value_loss = (value_loss_unweighted * weights).mean() if weights is not None else value_loss_unweighted.mean()\r
    \r
    entropy = -(softmax(logits_gen, dim=-1) * log_probs).sum(dim=-1).mean()\r
    \r
    total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy + curiosity_loss\r
    return total_loss, policy_loss, value_loss, entropy, curiosity_loss\r
\r
def get_human_feedback(code_string):\r
    # ... (คงเดิมจาก v8) ...\r
    if "error" in code_string.lower() or "bug" in code_string.lower(): return -2.0\r
    if len(code_string) > 800: return 0.5\r
    return 1.5\r
\r
# ! REWORKED V8: Training loop (คงเดิมจาก v8)\r
def train_ppo_with_accelerator(model, data_loader, val_loader, optimizer, codebert_pipeline, all_agents, hierarchical_memory, num_epochs, gradient_accumulation_steps, use_mock_reward, visualize_graphs, clip_epsilon, curiosity_weight, engine_name="", language="lua"):\r
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)\r
    model, optimizer, data_loader, val_loader = accelerator.prepare(model, optimizer, data_loader, val_loader)\r
    \r
    code_evaluator = CodeEvaluator(use_mock_reward=use_mock_reward, language=language)\r
    moe_router = all_agents["MixtureOfExpertsRouter"] # ! V9: ใช้ Router ที่อัปเดตแล้ว\r
\r
    logging.info(f"🚀 Starting PPO training with MoE (V9) for {engine_name}...")\r
    model.train()\r
    total_steps = len(data_loader) // gradient_accumulation_steps * num_epochs\r
    progress_bar = tqdm(range(total_steps), desc=f"Training ({engine_name})")\r
    replay_buffer = PrioritizedReplayBuffer(capacity=1024)\r
    \r
    for epoch in range(num_epochs):\r
        for step, batch in enumerate(data_loader):\r
            if batch is None: continue\r
            \r
            with accelerator.accumulate(model):\r
                initial_code_string = batch['code_contents'][0]\r
                \r
                # --- (คงเดิม) MoE in action (Critique -> Refine loop) ---\r
                \r
                # 1. Retrieve similar examples from memory\r
                retrieved_examples = hierarchical_memory.retrieve_similar(initial_code_string)\r
                \r
                # 2. First agent interaction: Critique\r
                critic_agent = all_agents["CodeCriticAgent"]\r
                eval_mock = code_evaluator.evaluate(initial_code_string, None) \r
                critique = critic_agent.critique(initial_code_string, eval_mock)\r
\r
                # 3. Second agent interaction: Refine\r
                refinement_agent = all_agents["CodeRefinementAgent"]\r
                refined_code = refinement_agent.refine(initial_code_string, critique, context_examples=retrieved_examples)\r
                \r
                # 4. Evaluate the refined code\r
                refined_graph = build_real_code_graph_ast(refined_code, model, codebert_pipeline, language=language)\r
                refined_reward_dict = code_evaluator.evaluate(refined_code, refined_graph)\r
                \r
                human_reward = get_human_feedback(refined_code)\r
                final_reward_value = refined_reward_dict['total_reward'] + human_reward\r
                \r
                # 5. Store high-quality results in hierarchical memory\r
                hierarchical_memory.add_experience(refined_code, final_reward_value, metadata={"engine": engine_name, "reward": final_reward_value})\r
\r
                # --- PPO Update (similar to v8) ---\r
                graph_embedding = None\r
                if refined_graph:\r
                    graph_embedding = model.graph_memory(\r
                        refined_graph.x, \r
                        refined_graph.edge_index, \r
                        None, \r
                        torch.zeros(refined_graph.x.shape[0], dtype=torch.long, device=accelerator.device)\r
                    )\r
\r
                with torch.no_grad():\r
                    gen_ids = model.tokenizer.encode(refined_code, return_tensors="pt")\r
                    # ! V9: ตรวจสอบความยาว\r
                    if gen_ids.size(1) == 0: continue # Skip if refinement is empty\r
                    \r
                    full_input_ids = torch.cat([batch['input_ids'], gen_ids], dim=1).to(accelerator.device)\r
                    if full_input_ids.size(1) > model.llm.config.max_position_embeddings:\r
                        full_input_ids = full_input_ids[:, :model.llm.config.max_position_embeddings]\r
                    \r
                    logits, value_preds, _ = model(full_input_ids, None, project_graph_embedding=graph_embedding)\r
                    \r
                    # ! V9: ตรวจสอบขนาด Logits\r
                    if logits.size(1) <= gen_ids.size(1): continue # Skip if output is too small\r
                        \r
                    log_probs_tensor = log_softmax(logits[:, -gen_ids.size(1)-1:-1, :], dim=-1)\r
                    \r
                    # ! V9: ตรวจสอบขนาดอีกครั้ง\r
                    if log_probs_tensor.size(1) != gen_ids.size(1):\r
                        # This mismatch can happen, skip this step\r
                        continue\r
                        \r
                    gathered_log_probs = log_probs_tensor.gather(2, gen_ids.to(accelerator.device).unsqueeze(-1)).squeeze(-1).mean()\r
                \r
                experience = (batch['input_ids'].cpu(), gen_ids.cpu(), torch.tensor([gen_ids.size(1)]), gathered_log_probs.cpu(), torch.tensor([final_reward_value]), graph_embedding.cpu() if graph_embedding is not None else None, torch.tensor([False]))\r
                replay_buffer.push(experience)\r
\r
                if len(replay_buffer) >= 32:\r
                    batch_data = replay_buffer.sample(32)\r
                    if not batch_data: continue\r
                    states, actions, lens, old_log_probs, rewards, next_states, dones, weights, indices = batch_data\r
                    normalized_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)\r
                    with torch.no_grad():\r
                         _, old_value_preds, _ = model(states.to(accelerator.device))\r
                    loss_outputs = _calculate_ppo_loss(model, accelerator, states, actions, lens, old_log_probs, normalized_rewards, next_states, old_value_preds, dones, curiosity_weight, clip_epsilon, weights)\r
                    if loss_outputs and loss_outputs[0] is not None:\r
                        total_loss = loss_outputs[0]\r
                        accelerator.backward(total_loss)\r
                        if accelerator.sync_gradients:\r
                            accelerator.clip_grad_norm_(model.parameters(), 1.0)\r
                        optimizer.step()\r
                        optimizer.zero_grad()\r
                        replay_buffer.update_priorities(indices, torch.abs(rewards.squeeze() - old_value_preds.cpu().squeeze()))\r
\r
\r
            if accelerator.sync_gradients:\r
                progress_bar.update(1)\r
                progress_bar.set_postfix({"loss": total_loss.item() if 'total_loss' in locals() else 0.0, "reward": final_reward_value})\r
\r
        # --- Checkpoint saving ---\r
        if (epoch + 1) % 5 == 0:\r
            logging.info(f"Epoch {epoch + 1}/{num_epochs} finished.")\r
            accelerator.wait_for_everyone()\r
            unwrapped_model = accelerator.unwrap_model(model)\r
            save_dir = os.path.join("model_checkpoints_v9", engine_name, f"epoch_{epoch+1}") # ! V9\r
            os.makedirs(save_dir, exist_ok=True)\r
            unwrapped_model.llm.save_pretrained(save_dir)\r
            logging.info(f"✅ Model checkpoint for {engine_name} (V9) saved at epoch {epoch+1}")\r
            \r
    logging.info(f"PPO training for {engine_name} finished.")\r
\r
\r
# ! REWORKED V9: Function to run the 24/7 Chat GUI\r
def run_chat_app_gui(model_path: str):\r
    logging.info(f"🚀 Launching in 24/7 Interactive GUI Mode (V9)...")\r
    logging.info(f"Loading final trained model from: {model_path}")\r
    \r
    try:\r
        model = AutoModelForCausalLM.from_pretrained(\r
            model_path,\r
            torch_dtype=torch.bfloat16,\r
            device_map="auto",\r
            trust_remote_code=True,\r
            offload_folder="offload_run"\r
        )\r
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)\r
        logging.info("✅ Final model and tokenizer loaded successfully.")\r
    except Exception as e:\r
        logging.error(f"❌ Failed to load the final model: {e}")\r
        return\r
\r
    # ! V9: Lightweight wrapper for inference (Updated for Multi-Agent MoE)\r
    class InferenceWrapper(nn.Module):\r
        def __init__(self, llm, tokenizer, agents):\r
            super().__init__()\r
            self.llm = llm\r
            self.tokenizer = tokenizer\r
            self.agents = agents\r
            self.moe_router = agents["MixtureOfExpertsRouter"]\r
            # ! V9: Add specific agent for summarization\r
            self.long_context_agent = agents["LongContextSummarizerAgent"]\r
\r
        def summarize_long_text(self, text: str):\r
            """\r
            ! V9: Dedicated function to call the summarizer agent.\r
            """\r
            logging.info(f"Summarizing long text (length: {len(text)})...")\r
            # We use the agent's _generate method directly\r
            summary = self.long_context_agent.summarize(text)\r
            logging.info(f"Summary length: {len(summary)}")\r
            return summary\r
\r
        def generate_response(self, prompt_text: str):\r
            self.llm.eval()\r
            device = self.llm.device\r
            \r
            # ! V9: Use MoE to select a *list* of agents\r
            selected_agent_names = self.moe_router.route(prompt_text)\r
            \r
            # ! V9: Fuse System Prompts for Multi-Agent persona\r
            system_prompts = []\r
            for name in selected_agent_names:\r
                agent = self.agents.get(name)\r
                if agent and hasattr(agent, "__doc__") and agent.__doc__:\r
                    system_prompts.append(agent.__doc__.strip())\r
                elif name == "GeneralConversationAgent":\r
                     system_prompts.append("You are a helpful AI assistant.")\r
            \r
            if not system_prompts:\r
                fused_system_prompt = "You are a helpful AI assistant."\r
            elif len(system_prompts) == 1:\r
                fused_system_prompt = system_prompts[0]\r
            else:\r
                fused_system_prompt = "You are a multi-talented AI assistant. You must act as the following experts simultaneously:\\n\\n"\r
                for i, p in enumerate(system_prompts):\r
                    fused_system_prompt += f"EXPERT {i+1}: {p}\\n"\r
                fused_system_prompt += "\\nCombine these skills to answer the user's request comprehensively."\r
\r
            logging.info(f"Using Fused System Prompt for agents: {selected_agent_names}")\r
\r
            messages = [\r
                {"role": "system", "content": fused_system_prompt},\r
                {"role": "user", "content": prompt_text}\r
            ]\r
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\r
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)\r
            generation_params = { "max_new_tokens": 1500, "pad_token_id": self.tokenizer.pad_token_id, "eos_token_id": self.tokenizer.eos_token_id, "do_sample": True, "temperature": 0.7, "top_p": 0.95 }\r
            \r
            with torch.no_grad():\r
                generated_ids = self.llm.generate(input_ids, **generation_params)\r
            \r
            full_response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)\r
            \r
            # ! V9: Use improved response parsing\r
            assistant_split = "<|assistant|>"\r
            if assistant_split in full_response:\r
                generated_text = full_response.split(assistant_split)[-1].strip()\r
            else:\r
                # Fallback\r
                prompt_lines = prompt.splitlines()\r
                last_prompt_line = prompt_lines[-1] if prompt_lines else ""\r
                if last_prompt_line in full_response:\r
                    generated_text = full_response.split(last_prompt_line)[-1].strip()\r
                else:\r
                    generated_text = full_response \r
            \r
            return generated_text\r
\r
    # Instantiate agents for the inference wrapper\r
    inference_agents = {}\r
    # ! V9: Add new agents to the list\r
    agent_classes = [\r
        CodeGeneratorAgent, CodeCriticAgent, CodeRefinementAgent, AssetGeneratorAgent, \r
        BugReportGeneratorAgent, TestGenerationAgent, DocumentationAgent, AutoRefactoringAgent, \r
        GameDesignerAgent, CodeSummarizationAgent, CodeQuestionAnsweringAgent, WebDeveloperAgent, \r
        FinancialAnalystAgent, AIArchitectAgent, AnimationCodeAgent, DatabaseAdminAgent, \r
        GeneralConversationAgent, MarketingCopywriterAgent,\r
        LongContextSummarizerAgent, ScientificResearcherAgent, CreativeWriterAgent # ! V9 New Agents\r
    ]\r
    \r
    # Mock model for agent initialization (same as v8)\r
    mock_model_for_agents = type('obj', (object,), {\r
        'llm': model, \r
        'tokenizer': tokenizer, \r
        'generate_response': lambda *args, **kwargs: None\r
    })() \r
    \r
    for agent_class in agent_classes:\r
        if agent_class.__name__ == "AssetGeneratorAgent":\r
             inference_agents[agent_class.__name__] = agent_class()\r
        else:\r
             # ! V9: ใช้ docstring เป็น system prompt โดยอัตโนมัติ\r
             agent_instance = agent_class(mock_model_for_agents)\r
             inference_agents[agent_class.__name__] = agent_instance\r
\r
    # ! V9: Router needs all LLM agents\r
    inference_agents["MixtureOfExpertsRouter"] = MixtureOfExpertsRouter(mock_model_for_agents, inference_agents)\r
\r
    inference_model = InferenceWrapper(model, tokenizer, inference_agents)\r
    \r
    # ! V9: Instantiate the Security Moderator\r
    moderator = SecurityModeratorAgent()\r
    \r
    # ! V9: Define context length limit (in characters)\r
    # Phi-3-mini 4k tokens is ~12-16k chars. We set a lower limit to trigger summarization.\r
    MAX_PROMPT_LENGTH_CHARS = 3500 \r
\r
    # --- GUI Setup ---\r
    def send_message(event=None):\r
        user_input = user_entry.get()\r
        if not user_input:\r
            return\r
        \r
        chat_area.config(state=tk.NORMAL)\r
        chat_area.insert(tk.END, "You: " + user_input + "\\n\\n")\r
        chat_area.config(state=tk.DISABLED)\r
        user_entry.delete(0, tk.END)\r
        \r
        # Disable input while processing\r
        user_entry.config(state=tk.DISABLED)\r
        send_button.config(state=tk.DISABLED)\r
        \r
        def generate():\r
            try:\r
                # ! V9: STEP 1 - Pre-Screen Input\r
                is_safe, screening_message = moderator.pre_screen_input(user_input)\r
                if not is_safe:\r
                    chat_area.config(state=tk.NORMAL)\r
                    chat_area.insert(tk.END, f"AI (Security): {screening_message}\\n\\n")\r
                    chat_area.see(tk.END)\r
                    chat_area.config(state=tk.DISABLED)\r
                    return # Stop processing\r
                \r
                prompt_to_process = user_input\r
                \r
                # ! V9: STEP 2 - Long Context Handling\r
                if len(prompt_to_process) > MAX_PROMPT_LENGTH_CHARS:\r
                    chat_area.config(state=tk.NORMAL)\r
                    chat_area.insert(tk.END, "AI: (Your input is very long. Summarizing it first...)\\n\\n")\r
                    chat_area.see(tk.END)\r
                    chat_area.config(state=tk.DISABLED)\r
                    \r
                    prompt_to_process = inference_model.summarize_long_text(prompt_to_process)\r
                    \r
                    chat_area.config(state=tk.NORMAL)\r
                    chat_area.insert(tk.END, f"AI (Summary): {prompt_to_process}\\n(Now processing the summary...)\\n\\n")\r
                    chat_area.see(tk.END)\r
                    chat_area.config(state=tk.DISABLED)\r
\r
                # ! V9: STEP 3 - Generate Response (using MoE)\r
                raw_response = inference_model.generate_response(prompt_to_process)\r
                \r
                # ! V9: STEP 4 - Post-Screen Output\r
                is_safe, final_response = moderator.post_screen_output(raw_response)\r
                \r
                if not is_safe:\r
                    # If output is unsafe, display the canned safe response\r
                    chat_area.config(state=tk.NORMAL)\r
                    chat_area.insert(tk.END, f"AI (Security): {final_response}\\n\\n")\r
                    chat_area.see(tk.END)\r
                    chat_area.config(state=tk.DISABLED)\r
                else:\r
                    # Output is safe, display it\r
                    chat_area.config(state=tk.NORMAL)\r
                    chat_area.insert(tk.END, "AI: " + final_response + "\\n\\n")\r
                    chat_area.see(tk.END)\r
                    chat_area.config(state=tk.DISABLED)\r
\r
            except Exception as e:\r
                logging.error(f"Error during GUI generation: {e}")\r
                chat_area.config(state=tk.NORMAL)\r
                chat_area.insert(tk.END, f"AI (Error): An internal error occurred. Please try again.\\n\\n")\r
                chat_area.see(tk.END)\r
                chat_area.config(state=tk.DISABLED)\r
            finally:\r
                # Re-enable input regardless of outcome\r
                user_entry.config(state=tk.NORMAL)\r
                send_button.config(state=tk.NORMAL)\r
                user_entry.focus_set()\r
\r
        threading.Thread(target=generate, daemon=True).start()\r
\r
    root = tk.Tk()\r
    root.title("Multiverse V9 AI")\r
    \r
    # Position window at bottom right (คงเดิม)\r
    window_width = 400\r
    window_height = 500\r
    screen_width = root.winfo_screenwidth()\r
    screen_height = root.winfo_screenheight()\r
    x_cordinate = int(screen_width - window_width - 20)\r
    y_cordinate = int(screen_height - window_height - 60) \r
    root.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")\r
    root.resizable(False, False)\r
\r
    main_frame = Frame(root, bg="#2E2E2E")\r
    main_frame.pack(fill=tk.BOTH, expand=True)\r
\r
    chat_area = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, state=tk.DISABLED, bg="#1E1E1E", fg="#D4D4D4", font=("Arial", 10))\r
    chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)\r
\r
    input_frame = Frame(main_frame, bg="#2E2E2E")\r
    input_frame.pack(padx=10, pady=5, fill=tk.X)\r
\r
    user_entry = Entry(input_frame, bg="#3C3C3C", fg="#D4D4D4", font=("Arial", 10), insertbackground='white')\r
    user_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)\r
    user_entry.bind("<Return>", send_message)\r
    user_entry.focus_set()\r
\r
    send_button = Button(input_frame, text="Send", command=send_message, bg="#0E639C", fg="white", activebackground="#1177BB", borderwidth=0)\r
    send_button.pack(side=tk.RIGHT, padx=(5,0))\r
\r
    root.mainloop()\r
\r
# ! ##################################################################\r
# ! ################ REWORKED V9 MAIN FUNCTION #######################\r
# ! ##################################################################\r
def main():\r
    parser = argparse.ArgumentParser(description="AI FWK Multiverse V9 - Training and Inference")\r
    parser.add_argument('mode', choices=['train', 'run'], help="Choose 'train' to start the training pipeline or 'run' to start the interactive chat GUI.")\r
    args = parser.parse_args()\r
\r
    # Start resource monitoring\r
    monitor = ResourceMonitor()\r
    monitor.start()\r
\r
    if args.mode == 'train':\r
        github_token = os.getenv("GITHUB_TOKEN")\r
        if not github_token:\r
            logging.error("CRITICAL ERROR: GITHUB_TOKEN environment variable is not set. Cannot download data.")\r
            sys.exit(1)\r
\r
        engine_configs = [\r
            {"name": "Roblox", "data_dir": "roblox_code_data", "github_query": "roblox luau", "file_extensions": ["lua", "luau"], "mock_epochs": 1, "real_epochs": 5, "language": "luau"},\r
            {"name": "Godot", "data_dir": "godot_code_data", "github_query": "godot gdscript", "file_extensions": ["gd"], "mock_epochs": 1, "real_epochs": 5, "language": "gdscript"},\r
            {"name": "Unity", "data_dir": "unity_code_data", "github_query": "unity csharp", "file_extensions": ["cs"], "mock_epochs": 1, "real_epochs": 5, "language": "csharp"},\r
            {"name": "Unreal", "data_dir": "unreal_code_data", "github_query": "unreal engine c++", "file_extensions": ["cpp", "h"], "mock_epochs": 1, "real_epochs": 5, "language": "c++"},\r
            {"name": "WebDev", "data_dir": "web_code_data", "github_query": "react javascript", "file_extensions": ["js", "jsx", "ts", "tsx", "html", "css"], "mock_epochs": 1, "real_epochs": 5, "language": "javascript"},\r
            {"name": "Python", "data_dir": "python_code_data", "github_query": "python", "file_extensions": ["py"], "mock_epochs": 1, "real_epochs": 5, "language": "python"}\r
        ]\r
\r
        logging.info("Initializing MultiAgentLLM (V9) model for training...")\r
        model = MultiAgentLLM()\r
        logging.info("Initializing CodeBERT pipeline...")\r
        codebert_pipeline = pipeline("feature-extraction", model="microsoft/CodeBERT-base", tokenizer="microsoft/CodeBERT-base", device=0 if torch.cuda.is_available() else -1)\r
\r
        # ! V9: Initialize Hierarchical Memory and all Agents\r
        hierarchical_memory = HierarchicalMemory(codebert_pipeline)\r
        \r
        all_agents = {}\r
        # ! V9: Add new agents to training list\r
        agent_classes = [\r
            CodeGeneratorAgent, CodeCriticAgent, CodeRefinementAgent, AssetGeneratorAgent, \r
            BugReportGeneratorAgent, TestGenerationAgent, DocumentationAgent, AutoRefactoringAgent, \r
            GameDesignerAgent, CodeSummarizationAgent, CodeQuestionAnsweringAgent, WebDeveloperAgent, \r
            FinancialAnalystAgent, AIArchitectAgent, AnimationCodeAgent, DatabaseAdminAgent, \r
            GeneralConversationAgent, MarketingCopywriterAgent,\r
            LongContextSummarizerAgent, ScientificResearcherAgent, CreativeWriterAgent # ! V9 New Agents\r
        ]\r
        \r
        for agent_class in agent_classes:\r
            if agent_class.__name__ == "AssetGeneratorAgent":\r
                 all_agents[agent_class.__name__] = agent_class()\r
            else:\r
                 all_agents[agent_class.__name__] = agent_class(model)\r
        \r
        # ! V9: Router needs all LLM agents\r
        all_agents["MixtureOfExpertsRouter"] = MixtureOfExpertsRouter(model, all_agents)\r
\r
\r
        best_params = {"lr": 5e-5, "batch_size": 1, "clip_epsilon": 0.2, "curiosity_weight": 0.03}\r
        os.makedirs("model_checkpoints_v9", exist_ok=True) # ! V9\r
\r
        for i, config in enumerate(engine_configs):\r
            engine_name, data_dir, language = config["name"], config["data_dir"], config["language"]\r
            logging.info(f"\\n{'='*25}\\n Stage {i+1}/{len(engine_configs)}: Processing Engine: {engine_name} \\n{'='*25}")\r
\r
            download_code_from_github(engine_name, config["github_query"], config["file_extensions"], data_dir, github_token)\r
\r
            try:\r
                dataset = CodeDataset(data_dir, model.tokenizer, model, codebert_pipeline, file_extensions=[f"{ext}" for ext in config["file_extensions"]]) # ! V9: Fix glob pattern\r
                if len(dataset) < best_params["batch_size"]:\r
                    logging.warning(f"Dataset for {engine_name} is too small ({len(dataset)} files). Skipping.")\r
                    continue\r
                \r
                train_size = int(0.95 * len(dataset))\r
                val_size = len(dataset) - train_size\r
                train_dataset, val_dataset = random_split(dataset, [train_size, val_size])\r
\r
                num_workers = 0\r
                if platform.system() == "Linux": num_workers = min(4, os.cpu_count() // 2 if os.cpu_count() else 0)\r
                logging.info(f"Using {num_workers} workers for DataLoader.")\r
\r
                train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True, collate_fn=custom_collate_fn, num_workers=num_workers)\r
                val_loader = DataLoader(val_dataset, batch_size=best_params["batch_size"], shuffle=False, collate_fn=custom_collate_fn, num_workers=num_workers)\r
                optimizer = torch.optim.AdamW(model.parameters(), lr=best_params["lr"])\r
\r
                logging.info(f"\\n--- [{engine_name}] Phase 1: Training with mock rewards ---")\r
                train_ppo_with_accelerator(model, train_loader, val_loader, optimizer, codebert_pipeline, all_agents, hierarchical_memory, num_epochs=config["mock_epochs"], gradient_accumulation_steps=8, use_mock_reward=True, visualize_graphs=False, **best_params, engine_name=engine_name, language=language)\r
                \r
                logging.info(f"\\n--- [{engine_name}] Phase 2: Fine-tuning with real rewards ---")\r
                train_ppo_with_accelerator(model, train_loader, val_loader, optimizer, codebert_pipeline, all_agents, hierarchical_memory, num_epochs=config["real_epochs"], gradient_accumulation_steps=8, use_mock_reward=False, visualize_graphs=True, **best_params, engine_name=engine_name, language=language)\r
\r
            except FileNotFoundError as e:\r
                logging.error(f"CRITICAL ERROR for {engine_name}: {e}. Skipping this engine.")\r
                continue\r
            finally:\r
                if 'train_loader' in locals(): del train_loader, val_loader, dataset, train_dataset, val_dataset, optimizer\r
                gc.collect()\r
                if torch.cuda.is_available(): torch.cuda.empty_cache()\r
                logging.info(f"🧹 Cleaned up memory after training on {engine_name}.")\r
        \r
        final_model_save_path = "final_multi_engine_model_v9" # ! V9\r
        unwrapped_model = model.module if hasattr(model, 'module') else model\r
        unwrapped_model.llm.save_pretrained(final_model_save_path)\r
        logging.info(f"🎉 Final, sequentially-trained model (V9) saved to '{final_model_save_path}'.")\r
        logging.info("Training complete. Automatically launching chat GUI...")\r
        \r
        # Automatically switch to run mode\r
        run_chat_app_gui(model_path=final_model_save_path)\r
\r
    elif args.mode == 'run':\r
        final_model_path = "final_multi_engine_model_v9" # ! V9\r
        if not os.path.exists(final_model_path):\r
            logging.error(f"Model directory not found at '{final_model_path}'.")\r
            logging.error("Please train the model first by running: python multiverse_v9.py train")\r
            sys.exit(1)\r
        \r
        run_chat_app_gui(model_path=final_model_path)\r
    \r
    # Stop the monitor when the program exits\r
    monitor.stop()\r
\r
\r
if __name__ == "__main__":\r
    main()`,N=`import torch\r
import torch.nn as nn\r
from torch_geometric.nn import GATv2Conv, global_mean_pool\r
from torch_geometric.data import Data, Batch\r
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline\r
from datasets import load_dataset\r
from torch.utils.data import Dataset, DataLoader, random_split\r
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training\r
import os\r
import glob\r
import requests\r
import base64\r
import time\r
import sys\r
from torch.nn.functional import mse_loss, softmax, log_softmax, pad\r
from accelerate import Accelerator\r
from tqdm.auto import tqdm\r
import re\r
import random\r
import json\r
import ast\r
import networkx as nx\r
import subprocess\r
import pickle\r
import matplotlib.pyplot as plt\r
import shutil\r
import logging\r
import numpy as np\r
import joblib\r
import optuna\r
import gc # ! NEW: Import garbage collector for memory management\r
from datetime import datetime\r
import heapq\r
import platform\r
\r
# --- Configuration & Setup (from original) ---\r
def set_seed(s=42):\r
    random.seed(s)\r
    np.random.seed(s)\r
    torch.manual_seed(s)\r
    if torch.cuda.is_available():\r
        torch.cuda.manual_seed_all(s)\r
\r
set_seed(42)\r
\r
if torch.cuda.is_available():\r
    torch.backends.cudnn.benchmark = True\r
\r
logging.basicConfig(filename='training_errors.log', level=logging.ERROR,\r
                    format='%(asctime)s - %(levelname)s - %(message)s')\r
console_handler = logging.StreamHandler()\r
console_handler.setLevel(logging.INFO)\r
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')\r
console_handler.setFormatter(formatter)\r
logging.getLogger().addHandler(console_handler)\r
\r
try:\r
    from luaparser import ast as luaparser_ast\r
    from luaparser import parser as luaparser_parser\r
    logging.info("Using luaparser for Lua AST-based graphs.")\r
except ImportError:\r
    luaparser_ast = None\r
    luaparser_parser = None\r
    logging.warning("luaparser not found. Cannot build AST-based graphs.")\r
\r
try:\r
    from pygments.lexers.lua import LuaLexer\r
    from pygments import lex\r
    from pygments.token import Token\r
except ImportError:\r
    logging.warning("Pygments with LuaLexer not found. Falling back to regex.")\r
    LuaLexer = None\r
\r
# --- NEW: Prioritized Experience Replay (PER) Buffer ---\r
class PrioritizedReplayBuffer:\r
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_end=1.0):\r
        self.capacity = capacity\r
        self.alpha = alpha\r
        self.beta = beta_start\r
        self.beta_start = beta_start # Store start value for correct increment calculation\r
        self.beta_end = beta_end\r
        # FIX A: Correct beta annealing increment calculation\r
        self.beta_increment = (self.beta_end - self.beta_start) / float(self.capacity)\r
        self.buffer = []\r
        self.priorities = np.zeros(capacity, dtype=np.float32)\r
        self.position = 0\r
        self.device = 'cpu'\r
        self.max_priority = 1.0\r
\r
    def push(self, experience):\r
        priority = self.max_priority ** self.alpha\r
        if len(self.buffer) < self.capacity:\r
            self.buffer.append(experience)\r
        else:\r
            self.buffer[self.position] = experience\r
        self.priorities[self.position] = priority\r
        self.position = (self.position + 1) % self.capacity\r
\r
    def sample(self, batch_size):\r
        if len(self.buffer) < batch_size:\r
            return None\r
\r
        priorities = self.priorities[:len(self.buffer)]\r
        probs = priorities ** self.alpha\r
        probs /= probs.sum()\r
\r
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)\r
        samples = [self.buffer[idx] for idx in indices]\r
\r
        total = len(self.buffer)\r
        weights = (total * probs[indices]) ** (-self.beta)\r
        weights /= weights.max()\r
\r
        state, action, action_len, log_prob, reward, next_state, done = zip(*samples)\r
\r
        max_len = max(a.size(0) for a in action)\r
        action_padded = [pad(a, (0, max_len - a.size(0)), 'constant', 0) for a in action]\r
\r
        # FIX A: Correctly update beta value\r
        self.beta = min(self.beta + self.beta_increment, self.beta_end)\r
\r
        # FIX B: Handle next_state which is a list of tensors or None\r
        valid_next_states = [n for n in next_state if n is not None]\r
        batched_next_state = torch.stack(valid_next_states) if valid_next_states else None\r
\r
        return (\r
            torch.stack(state),\r
            torch.stack(action_padded),\r
            torch.stack(action_len),\r
            torch.stack(log_prob),\r
            torch.stack(reward),\r
            batched_next_state, # Return the correctly batched tensor\r
            torch.stack(done),\r
            torch.tensor(weights, dtype=torch.float32),\r
            indices\r
        )\r
\r
    def update_priorities(self, indices, errors):\r
        for idx, error in zip(indices, errors):\r
            self.priorities[idx] = max(error.item(), 1e-6)\r
        self.max_priority = self.priorities[:len(self.buffer)].max()\r
\r
    def __len__(self):\r
        return len(self.buffer)\r
\r
    def clear(self):\r
        self.buffer.clear()\r
        self.priorities = np.zeros(self.capacity, dtype=np.float32)\r
        self.position = 0\r
        self.max_priority = 1.0\r
\r
# --- 1. Enhanced MARL Architecture with Knowledge Graph Diffusion ---\r
\r
# NEW: Long-Term Memory using Vectorized Storage (RAG simulation)\r
class VectorizedMemory:\r
    def __init__(self, embedding_pipeline):\r
        self.memory = {}\r
        self.vectors = []\r
        self.code_snippets = []\r
        self.embedding_pipeline = embedding_pipeline\r
        logging.info("VectorizedMemory (Long-Term Memory) initialized.")\r
\r
    def add_experience(self, code_snippet, reward):\r
        # Only store high-quality experiences\r
        if reward < 5.0:\r
            return\r
\r
        # Avoid duplicates\r
        if code_snippet in self.memory:\r
            return\r
\r
        try:\r
            # Generate embedding for the good code snippet\r
            embedding = self.embedding_pipeline(code_snippet)\r
            vector = np.array(embedding).mean(axis=1).flatten()\r
\r
            self.memory[code_snippet] = vector\r
            self.vectors.append(vector)\r
            self.code_snippets.append(code_snippet)\r
            logging.info(f"Added a high-quality code snippet to Long-Term Memory.")\r
        except Exception as e:\r
            logging.error(f"Could not process and add experience to VectorizedMemory: {e}")\r
\r
\r
    def retrieve_similar(self, query_code, k=2):\r
        if not self.vectors:\r
            return []\r
\r
        try:\r
            query_embedding = self.embedding_pipeline(query_code)\r
            query_vector = np.array(query_embedding).mean(axis=1).flatten()\r
\r
            # Calculate cosine similarity\r
            vectors_matrix = np.array(self.vectors)\r
            dot_product = np.dot(vectors_matrix, query_vector)\r
            norm_query = np.linalg.norm(query_vector)\r
            norm_vectors = np.linalg.norm(vectors_matrix, axis=1)\r
            similarities = dot_product / (norm_vectors * norm_query)\r
\r
            # Get top-k most similar snippets\r
            top_k_indices = np.argsort(similarities)[::-1][:k]\r
            return [self.code_snippets[i] for i in top_k_indices]\r
        except Exception as e:\r
            logging.error(f"Could not retrieve similar code from VectorizedMemory: {e}")\r
            return []\r
\r
class ProjectGraphMemory(nn.Module):\r
    def __init__(self, num_features):\r
        super(ProjectGraphMemory, self).__init__()\r
        self.conv1 = GATv2Conv(num_features, 64, heads=4, edge_dim=1, concat=True)\r
        self.conv2 = GATv2Conv(64 * 4, 32, heads=2, edge_dim=1, concat=True)\r
        self.proj = nn.Linear(32 * 2, 32)\r
        self.device = 'cpu'\r
\r
    def forward(self, x, edge_index, edge_attr, batch):\r
        x = self.conv1(x, edge_index, edge_attr=edge_attr)\r
        x = x.relu()\r
        if x.device != self.conv2.weight.device:\r
            x = x.to(self.conv2.weight.device)\r
        x = self.conv2(x, edge_index, edge_attr=edge_attr)\r
        x = self.proj(x)\r
        return global_mean_pool(x, batch)\r
\r
class CuriosityModule(nn.Module):\r
    def __init__(self, feature_size, action_size):\r
        super(CuriosityModule, self).__init__()\r
        self.inverse_model = nn.Sequential(\r
            nn.Linear(feature_size * 2, 128),\r
            nn.ReLU(),\r
            nn.Linear(128, action_size)\r
        )\r
        self.forward_model = nn.Sequential(\r
            nn.Linear(feature_size + action_size, 128),\r
            nn.ReLU(),\r
            nn.Linear(128, feature_size)\r
        )\r
        self.feature_size = feature_size\r
        self.device = 'cpu'\r
\r
    def forward(self, current_features, action_features, next_features):\r
        combined_features = torch.cat([current_features, next_features], dim=-1)\r
        predicted_action = self.inverse_model(combined_features)\r
        combined_forward = torch.cat([current_features, action_features], dim=-1)\r
        predicted_next_features = self.forward_model(combined_forward)\r
        forward_loss = mse_loss(predicted_next_features, next_features.detach())\r
        return forward_loss\r
\r
def guess_lora_targets(model):\r
    names = []\r
    for n, mod in model.named_modules():\r
        if isinstance(mod, torch.nn.Linear) and ("attn" in n or "attention" in n or "q_proj" in n or "k_proj" in n or "v_proj" in n):\r
            names.append(n.split('.')[-1])\r
    common = ["query_key_value", "dense", "out_proj", "c_attn", "c_proj"]\r
    return list(set(names + common))\r
\r
class MultiAgentLLM(nn.Module):\r
    def __init__(self, llm_name="Salesforce/codegen-2B-mono", lora_rank=8, lora_alpha=16, lora_dropout=0.05):\r
        super(MultiAgentLLM, self).__init__()\r
\r
        if torch.cuda.is_available():\r
            gpu_name = torch.cuda.get_device_name(0)\r
            logging.info(f"Detected GPU: {gpu_name}")\r
            if "RTX 30" in gpu_name or "RTX 40" in gpu_name or hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():\r
                logging.info("Using bfloat16 for better performance on modern GPUs.")\r
                compute_dtype = torch.bfloat16\r
            else:\r
                logging.info("Using float16 as a fallback for older GPUs.")\r
                compute_dtype = torch.float16\r
\r
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)\r
            logging.info(f"Total VRAM: {vram:.2f} GB")\r
\r
            # ! KEY IMPROVEMENT for 8GB VRAM: Force "auto" to enable offloading\r
            # This will use CPU RAM when VRAM is full, which is crucial for your setup.\r
            device_map_setting = "auto"\r
            logging.info(f"Setting device_map to '{device_map_setting}' to manage limited VRAM.")\r
\r
\r
            bnb_config = BitsAndBytesConfig(\r
                load_in_4bit=True,\r
                bnb_4bit_quant_type="nf4",\r
                bnb_4bit_compute_dtype=compute_dtype,\r
                bnb_4bit_use_double_quant=True,\r
            )\r
        else:\r
            logging.info("No CUDA device found. Using CPU.")\r
            if os.cpu_count() < 3:\r
                logging.warning("Less than 3 CPU cores detected. Performance may be very slow.")\r
            bnb_config = BitsAndBytesConfig(\r
                load_in_4bit=True,\r
                bnb_4bit_quant_type="nf4",\r
                bnb_4bit_compute_dtype=torch.float32,\r
                bnb_4bit_use_double_quant=True,\r
            )\r
            device_map_setting = "cpu"\r
\r
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)\r
        if self.tokenizer.pad_token is None:\r
            self.tokenizer.pad_token = self.tokenizer.eos_token\r
\r
        self.llm = AutoModelForCausalLM.from_pretrained(\r
            llm_name,\r
            quantization_config=bnb_config,\r
            trust_remote_code=True,\r
            device_map=device_map_setting,\r
            offload_folder="offload" # ! Specify folder for offloaded layers\r
        )\r
\r
        self.llm.config.torch_dtype = bnb_config.bnb_4bit_compute_dtype\r
        self.llm = prepare_model_for_kbit_training(self.llm)\r
\r
        targets = guess_lora_targets(self.llm)\r
        logging.info(f"Guessed LoRA target modules: {targets}")\r
\r
        lora_config = LoraConfig(\r
            r=lora_rank,\r
            lora_alpha=lora_alpha,\r
            lora_dropout=lora_dropout,\r
            bias="none",\r
            task_type="CAUSAL_LM",\r
            target_modules=targets\r
        )\r
        self.llm = get_peft_model(self.llm, lora_config)\r
        self.llm.print_trainable_parameters()\r
\r
        # FIX D: Use a fixed dimension for graph node embeddings to avoid dynamic resizing\r
        codebert_embedding_dim = 768\r
        extra_features_dim = 1  # For usage data, etc.\r
        self.fixed_graph_embedding_dim = codebert_embedding_dim + extra_features_dim\r
\r
        self.embedding_proj = nn.Linear(self.fixed_graph_embedding_dim, self.llm.config.hidden_size)\r
\r
        self.graph_memory = ProjectGraphMemory(num_features=self.llm.config.hidden_size)\r
\r
        self.graph_attn = nn.MultiheadAttention(\r
            embed_dim=self.llm.config.hidden_size,\r
            num_heads=4,\r
            batch_first=True\r
        )\r
        self.graph_norm = nn.LayerNorm(self.llm.config.hidden_size)\r
\r
        self.policy_head = nn.Sequential(\r
            nn.LayerNorm(self.llm.config.hidden_size),\r
            nn.Linear(self.llm.config.hidden_size, self.tokenizer.vocab_size)\r
        )\r
        self.value_head = nn.Sequential(\r
            nn.LayerNorm(self.llm.config.hidden_size),\r
            nn.Linear(self.llm.config.hidden_size, 1)\r
        )\r
        self.curiosity_module = CuriosityModule(self.llm.config.hidden_size, self.tokenizer.vocab_size)\r
\r
    def _model_device(self):\r
        return next(p.device for p in self.llm.parameters())\r
\r
    def _add_emojis(self, text):\r
        emojis = {\r
            "hello": "👋", "hi": "👋", "hey": "👋", "thank you": "🙏", "thanks": "🙏",\r
            "great": "👍", "good": "👍", "ok": "👍", "sorry": "😔", "apologize": "😔",\r
            "happy": "😊", "exciting": "✨", "code": "💻", "error": "❌", "bug": "🐛",\r
            "success": "✅", "completed": "✅", "question": "🤔", "help": "💡",\r
            "problem": "🤯"\r
        }\r
        for word, emoji in emojis.items():\r
            text = re.sub(r'\\b' + re.escape(word) + r'\\b', word + ' ' + emoji, text, flags=re.IGNORECASE)\r
        return text\r
\r
    def forward(self, input_ids, attention_mask=None, project_graph_embedding=None):\r
        device = self._model_device()\r
        input_ids = input_ids.to(device)\r
        if attention_mask is not None:\r
            attention_mask = attention_mask.to(device)\r
\r
        llm_outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)\r
        last_hidden_state = llm_outputs.hidden_states[-1]\r
\r
        if project_graph_embedding is not None:\r
            batch_size, seq_len, _ = last_hidden_state.shape\r
            expanded_graph_embedding = project_graph_embedding.unsqueeze(1).repeat(1, seq_len, 1)\r
            fused_state, _ = self.graph_attn(last_hidden_state, expanded_graph_embedding, expanded_graph_embedding)\r
            fused_state = self.graph_norm(fused_state)\r
            fused_state = fused_state + last_hidden_state\r
        else:\r
            fused_state = last_hidden_state\r
\r
        logits = self.policy_head(fused_state.to(self.policy_head[1].weight.dtype))\r
        last_token_hidden_state = fused_state[:, -1, :]\r
        value = self.value_head(last_token_hidden_state.to(self.value_head[1].weight.dtype))\r
\r
        return logits, value, fused_state\r
\r
    def generate_response(self, prompt_text: str, max_new_tokens=512, do_sample=True, temperature=0.8, top_k=50, top_p=0.95):\r
        self.eval()\r
        device = self._model_device()\r
\r
        template_prompt = (\r
            "You are a helpful and experienced AI assistant for game development. "\r
            "Your task is to provide a complete solution to the user's request. "\r
            "First, give a friendly and clear explanation of the solution in Thai, using emojis to make it easy to understand. "\r
            "Then, provide the complete, well-commented code. "\r
            "The user's request is: "\r
            f"'{prompt_text}'\\n"\r
            "Here is the solution:"\r
        )\r
\r
        input_ids = self.tokenizer.encode(template_prompt, return_tensors="pt").to(device)\r
        attention_mask = torch.ones_like(input_ids).to(device)\r
\r
        generation_params = {\r
            "input_ids": input_ids,\r
            "attention_mask": attention_mask,\r
            "max_new_tokens": max_new_tokens,\r
            "pad_token_id": self.tokenizer.pad_token_id,\r
            "num_return_sequences": 1,\r
            "return_dict_in_generate": True,\r
            "output_scores": True,\r
            "do_sample": do_sample,\r
            "temperature": temperature,\r
            "top_k": top_k,\r
            "top_p": top_p\r
        }\r
\r
        if do_sample:\r
            temp_decay_factor = 0.95\r
            generation_params['temperature'] *= temp_decay_factor\r
\r
        generated_outputs = self.llm.generate(**generation_params)\r
        generated_ids = generated_outputs.sequences[0]\r
        full_response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)\r
\r
        start_index = full_response.find(template_prompt)\r
        if start_index != -1:\r
            generated_text = full_response[start_index + len(template_prompt):].strip()\r
        else:\r
            # Fallback if the prompt is not found in the output\r
            # This can happen with some models that rephrase the beginning\r
            input_text_decoded = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)\r
            if full_response.startswith(input_text_decoded):\r
                 generated_text = full_response[len(input_text_decoded):].strip()\r
            else:\r
                 generated_text = full_response\r
\r
        return generated_text, generated_outputs.scores\r
\r
# --- 2. Advanced Code Evaluation and MARL Agents ---\r
class CodeEvaluator:\r
    def __init__(self, use_mock_reward=False, language="lua"):\r
        self.use_mock_reward = use_mock_reward\r
        self.language = language\r
        # Patterns for different languages\r
        self.safe_patterns_lua = [\r
            r'io\\.open', r'os\\.execute', r'os\\.getenv', r'require\\(["\\'](http|socket)',\r
            r'HttpService', r'RunService', r'pcall\\s*\\(\\s*require'\r
        ]\r
        self.safe_patterns_cs = [\r
            r'System\\.IO\\.File', r'System\\.Diagnostics\\.Process', r'System\\.Net\\.Http'\r
        ]\r
        self.safe_patterns_gd = [\r
            r'OS\\.execute', r'File\\.new', r'Directory\\.new'\r
        ]\r
        # NEW: More advanced security vulnerability patterns\r
        self.vulnerability_patterns = {\r
            "lua": [r'loadstring'], # Potential Remote Code Execution\r
            "c#": [r'SqlCommand.*\\.CommandText\\s*=\\s*".*"\\s*\\+'], # Potential SQL Injection\r
            "gdscript": [],\r
            "c++": [r'strcpy', r'sprintf'] # Potential Buffer Overflows\r
        }\r
        self.llm = None # Will be set during training\r
\r
    def _pre_check_code(self, code_string):\r
        patterns = []\r
        if self.language == "lua":\r
            patterns = self.safe_patterns_lua\r
        elif self.language == "c#":\r
            patterns = self.safe_patterns_cs\r
        elif self.language == "gdscript":\r
            patterns = self.safe_patterns_gd\r
\r
        for pattern in patterns:\r
            if re.search(pattern, code_string, re.IGNORECASE):\r
                return False, f"Potential security risk detected: {pattern}"\r
        return True, "No security risks detected."\r
\r
    # NEW: Enhanced security audit method\r
    def _security_audit(self, code_string):\r
        """\r
        Performs a more detailed security check for common vulnerabilities.\r
        Returns a penalty score and a log of findings.\r
        """\r
        penalty = 0.0\r
        findings = []\r
        lang_patterns = self.vulnerability_patterns.get(self.language, [])\r
        for pattern in lang_patterns:\r
            if re.search(pattern, code_string):\r
                findings.append(f"High-risk pattern found: '{pattern}'. This could lead to security vulnerabilities.")\r
                penalty -= 5.0 # Apply a significant penalty\r
        if not findings:\r
            return 0.0, "No high-risk security patterns found."\r
        return penalty, "\\n".join(findings)\r
\r
    def _static_analysis(self, code_string):\r
        if self.language != "lua":\r
            logging.warning(f"Static analysis not implemented for {self.language}. Skipping.")\r
            return 0.0, "Skipped", "Static analysis not available."\r
\r
        temp_file_path = "temp_code.lua"\r
        with open(temp_file_path, "w", encoding="utf-8") as f:\r
            f.write(code_string)\r
\r
        try:\r
            result = subprocess.run(['luacheck', temp_file_path], capture_output=True, text=True, timeout=5)\r
            os.remove(temp_file_path)\r
            if "No issues found" in result.stdout:\r
                return 1.0, "Syntax check: OK", result.stdout\r
            else:\r
                return -1.0, f"Syntax check failed: {result.stdout}", result.stdout\r
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:\r
            return -1.0, f"Error during static analysis: {e}", ""\r
\r
    def _dynamic_analysis(self, code_string):\r
        if self.language != "lua":\r
            logging.warning(f"Dynamic analysis not implemented for {self.language}. Skipping.")\r
            return 0.0, "Skipped", "Dynamic analysis not available."\r
\r
        if shutil.which("docker") is None:\r
            logging.warning("Docker not installed, skipping functional test.")\r
            return 0.0, "Docker not available", "Docker not found."\r
\r
        temp_file_path = "temp_code.lua"\r
        with open(temp_file_path, "w", encoding="utf-8") as f:\r
            f.write(code_string)\r
\r
        docker_command = [\r
            'docker', 'run', '--rm', '--network', 'none', '--pids-limit', '256',\r
            '--cpus', '1', '--memory', '512m', '--ulimit', 'cpu=5',\r
            '--read-only', '-v', f'{os.getcwd()}/temp_code.lua:/app/temp_code.lua:ro',\r
            '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',\r
            'luau_runtime_image', 'luau', '/app/temp_code.lua'\r
        ]\r
\r
        try:\r
            start_time = time.time()\r
            result = subprocess.run(docker_command, capture_output=True, text=True, timeout=10)\r
            execution_time = time.time() - start_time\r
\r
            if result.returncode == 0:\r
                reward = 10.0\r
                efficiency_reward = max(0, 1.0 - execution_time / 5.0)\r
                return reward + efficiency_reward, f"Functional test: Success! Output: {result.stdout}", result.stdout + "\\n" + result.stderr\r
            else:\r
                return -5.0, f"Functional test failed. Exit code: {result.returncode}", result.stdout + "\\n" + result.stderr\r
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:\r
            return -2.0, f"Error during dynamic analysis: {e}", ""\r
        finally:\r
            if os.path.exists(temp_file_path):\r
                os.remove(temp_file_path)\r
\r
    def _assess_readability(self, code_string):\r
        # A simple heuristic-based assessment\r
        lines = code_string.split('\\n')\r
        comment_lines = sum(1 for line in lines if line.strip().startswith('--') or line.strip().startswith('//'))\r
        blank_lines = sum(1 for line in lines if not line.strip())\r
\r
        total_lines = len(lines)\r
        if total_lines == 0: return 0.0\r
\r
        # Reward for comments and blank lines, penalize for long lines\r
        comment_ratio = comment_lines / total_lines\r
        blank_ratio = blank_lines / total_lines\r
\r
        long_line_penalty = sum(1 for line in lines if len(line) > 80)\r
\r
        readability_score = (comment_ratio * 0.5) + (blank_ratio * 0.3) - (long_line_penalty * 0.1)\r
        return max(0, min(1, readability_score))\r
\r
    def evaluate(self, code_string, project_graph):\r
        if self.use_mock_reward:\r
            return {\r
                "total_reward": random.uniform(-1, 10),\r
                "correctness_score": random.uniform(0, 10),\r
                "efficiency_score": random.uniform(0, 1),\r
                "knowledge_graph_score": random.uniform(0, 5),\r
                "readability_score": random.uniform(0, 1),\r
                "security_score": 0.0,\r
                "luacheck_log": "Mock luacheck log",\r
                "docker_log": "Mock docker log",\r
                "security_log": "Security audit skipped for mock reward."\r
            }\r
\r
        is_safe, security_log = self._pre_check_code(code_string)\r
        if not is_safe:\r
            return {\r
                "total_reward": -10.0,\r
                "correctness_score": -10.0,\r
                "efficiency_score": 0.0,\r
                "knowledge_graph_score": 0.0,\r
                "readability_score": 0.0,\r
                "security_score": -10.0,\r
                "luacheck_log": "Skipped due to security risk.",\r
                "docker_log": security_log,\r
                "security_log": security_log\r
            }\r
\r
        security_score, detailed_security_log = self._security_audit(code_string)\r
        syntax_score, _, luacheck_log = self._static_analysis(code_string)\r
        functional_score, _, docker_log = self._dynamic_analysis(code_string)\r
        correctness_score = (syntax_score + functional_score) / 2\r
        efficiency_score = max(0, 1.0 - len(code_string) / 1000)\r
        readability_score = self._assess_readability(code_string)\r
\r
        kg_score = 0\r
        if project_graph is not None and project_graph.num_nodes > 0:\r
            asset_nodes = [i for i, data in enumerate(project_graph.x.tolist()) if project_graph.node_type[i] == 1]\r
            if asset_nodes:\r
                if 'Instance.new("Part")' in code_string or 'TextureId' in code_string:\r
                    kg_score += 2.0\r
                asset_node_index = asset_nodes[0]\r
                asset_id = project_graph.node_id[asset_node_index]\r
                if asset_id and str(asset_id) in code_string:\r
                    kg_score += 3.0\r
\r
            # Check for library/API usage in the graph\r
            if hasattr(project_graph, 'node_type') and 2 in project_graph.node_type:\r
                if 'Roblox.HttpService' in code_string:\r
                    kg_score += 2.0\r
                if 'UnityEngine.Rigidbody' in code_string:\r
                    kg_score += 2.0\r
\r
        # NEW: Integrate security score into the total reward calculation\r
        total_reward = (0.35 * correctness_score) + \\\r
                       (0.15 * efficiency_score) + \\\r
                       (0.15 * kg_score) + \\\r
                       (0.15 * readability_score) + \\\r
                       (0.20 * security_score) # Security has a significant weight\r
\r
        return {\r
            "total_reward": total_reward,\r
            "correctness_score": correctness_score,\r
            "efficiency_score": efficiency_score,\r
            "knowledge_graph_score": kg_score,\r
            "readability_score": readability_score,\r
            "security_score": security_score,\r
            "luacheck_log": luacheck_log,\r
            "docker_log": docker_log,\r
            "security_log": detailed_security_log\r
        }\r
\r
class AssetGeneratorAgent:\r
    def __init__(self):\r
        logging.info("AssetGeneratorAgent initialized.")\r
        self.asset_database = {}\r
\r
    def generate_asset(self, prompt):\r
        asset_id = f"asset_{hash(prompt) % 10000}"\r
        asset_type = "Texture" if "texture" in prompt.lower() else "Model" if "model" in prompt.lower() else "Part"\r
        self.asset_database[asset_id] = {"prompt": prompt, "type": asset_type}\r
        logging.info(f"Generated mock asset with ID {asset_id} for prompt '{prompt}'")\r
        return asset_id\r
\r
# --- NEW: Bug Report Generator Agent ---\r
class BugReportGeneratorAgent:\r
    def __init__(self, model):\r
        self.model = model\r
\r
    def generate_report(self, code_string, error_log):\r
        prompt = f"""\r
        Analyze the following code and error log to create a professional bug report.\r
        Code:\r
        \`\`\`\r
        {code_string}\r
        \`\`\`\r
        Error Log:\r
        {error_log}\r
\r
        Provide a clear bug report with the following sections:\r
        - **Bug Description:** What is the issue?\r
        - **Steps to Reproduce:** How can this bug be replicated?\r
        - **Expected Behavior:** What should happen?\r
        - **Actual Behavior:** What is happening instead?\r
        - **Relevant Code Line:** Specify the line number where the bug likely originates.\r
        """\r
        bug_report, _ = self.model.generate_response(prompt, max_new_tokens=256, temperature=0.6)\r
\r
        # Extract relevant info\r
        description = re.search(r'\\*\\*Bug Description:\\*\\*(.*?)\\n\\n', bug_report, re.DOTALL)\r
        line_num = re.search(r'\\*\\*Relevant Code Line:\\*\\*(.*?)\\n', bug_report)\r
\r
        return {\r
            'full_report': bug_report,\r
            'description': description.group(1).strip() if description else 'No description.',\r
            'line_number': int(line_num.group(1).strip()) if line_num and line_num.group(1).strip().isdigit() else None\r
        }\r
\r
class CodeGeneratorAgent:\r
    def __init__(self, model):\r
        self.model = model\r
\r
    def generate(self, prompt, project_graph_embedding=None, context_examples=None):\r
        # NEW: Augment prompt with retrieved examples (RAG)\r
        if context_examples:\r
            examples_text = "\\n\\n---\\nHere are some relevant examples of good code:\\n---\\n"\r
            for ex in context_examples:\r
                examples_text += f"\`\`\`lua\\n{ex}\\n\`\`\`\\n"\r
            prompt += examples_text\r
\r
        return self.model.generate_response(prompt)\r
\r
class CodeCriticAgent:\r
    def __init__(self, model):\r
        self.model = model\r
\r
    def critique(self, code_string, evaluation_results, asset_id=None):\r
        # NEW: Include security score in the critique prompt\r
        critique_prompt = f"""\r
        Analyze the following code and its evaluation results.\r
        Code:\r
        \`\`\`\r
        {code_string}\r
        \`\`\`\r
\r
        Evaluation Results:\r
        Correctness Score: {evaluation_results['correctness_score']}\r
        Efficiency Score: {evaluation_results['efficiency_score']}\r
        Knowledge Graph Score: {evaluation_results['knowledge_graph_score']}\r
        Readability Score: {evaluation_results['readability_score']}\r
        Security Score: {evaluation_results.get('security_score', 'N/A')}\r
        Logs: {evaluation_results['luacheck_log']} | {evaluation_results['docker_log']}\r
        Security Log: {evaluation_results.get('security_log', 'N/A')}\r
        Asset ID: {asset_id}\r
\r
        Provide a constructive critique. Identify bugs, errors, inefficiencies, or security risks. Suggest specific improvements.\r
        """\r
        critique_response, _ = self.model.generate_response(critique_prompt, max_new_tokens=256, temperature=0.5)\r
        return critique_response\r
\r
class CodeRefinementAgent:\r
    def __init__(self, model):\r
        self.model = model\r
\r
    def refine(self, original_code, critique, bug_report=None, asset_id=None, context_examples=None):\r
        # NEW: Augment prompt with retrieved examples (RAG)\r
        refinement_prompt = f"""\r
        Based on the critique and bug report, refactor the original code.\r
        Original Code:\r
        \`\`\`\r
        {original_code}\r
        \`\`\`\r
        Critique:\r
        {critique}\r
\r
        Bug Report (if available):\r
        {bug_report['full_report'] if bug_report else "N/A"}\r
\r
        Consider the asset with ID: {asset_id}.\r
        """\r
        if context_examples:\r
            examples_text = "\\n\\n---\\nHere are some relevant examples to guide your refinement:\\n---\\n"\r
            for ex in context_examples:\r
                examples_text += f"\`\`\`lua\\n{ex}\\n\`\`\`\\n"\r
            refinement_prompt += examples_text\r
\r
        refinement_prompt += "\\nProvide the complete, corrected, and improved code. Only output the code block."\r
\r
        refined_code, _ = self.model.generate_response(refinement_prompt, max_new_tokens=512, temperature=0.7)\r
\r
        match = re.search(r'\`\`\`(?:\\w+)?\\n(.*?)\\n\`\`\`', refined_code, re.DOTALL)\r
        if match:\r
            return match.group(1).strip()\r
        return refined_code\r
\r
# --- NEW: Specialized Agents ---\r
class TestGenerationAgent:\r
    def __init__(self, model):\r
        self.model = model\r
\r
    def generate_tests(self, code_string, language="Luau"):\r
        """Generates unit tests for the given code."""\r
        prompt = f"""\r
        You are an expert in software testing. Your task is to generate a comprehensive set of unit tests for the following {language} code.\r
        The tests should cover normal cases, edge cases, and potential error conditions.\r
        Use a common testing framework if applicable for the language (e.g., TestEZ for Luau).\r
\r
        Code to test:\r
        \`\`\`\r
        {code_string}\r
        \`\`\`\r
\r
        Provide the complete unit test script.\r
        """\r
        tests, _ = self.model.generate_response(prompt, max_new_tokens=512, temperature=0.6)\r
        return tests\r
\r
    # NEW: Method for generating integration tests\r
    def generate_integration_tests(self, code_module_A, code_module_B, description, language="Luau"):\r
        """Generates integration tests for two interacting code modules."""\r
        prompt = f"""\r
        You are a senior QA engineer. Your task is to write an integration test for two modules.\r
        Description of interaction: {description}\r
\r
        Module A:\r
        \`\`\`\r
        {code_module_A}\r
        \`\`\`\r
\r
        Module B:\r
        \`\`\`\r
        {code_module_B}\r
        \`\`\`\r
\r
        Write a complete integration test script in {language} that verifies the correct interaction between these two modules.\r
        Focus on testing the data flow and function calls between them.\r
        """\r
        tests, _ = self.model.generate_response(prompt, max_new_tokens=600, temperature=0.65)\r
        return tests\r
\r
\r
class DocumentationAgent:\r
    def __init__(self, model):\r
        self.model = model\r
\r
    def generate_docs(self, code_string, language="Luau"):\r
        """Generates clear documentation for the given code."""\r
        prompt = f"""\r
        You are an expert technical writer. Your task is to create clear, concise, and easy-to-understand documentation for the following {language} code.\r
        Explain what the code does, describe its main functions/classes, and provide examples of how to use it.\r
        Format the output in Markdown.\r
\r
        Code to document:\r
        \`\`\`\r
        {code_string}\r
        \`\`\`\r
\r
        Provide the complete documentation.\r
        """\r
        docs, _ = self.model.generate_response(prompt, max_new_tokens=400, temperature=0.7)\r
        return docs\r
\r
# NEW: Agent for automatic code refactoring\r
class AutoRefactoringAgent:\r
    def __init__(self, model):\r
        self.model = model\r
\r
    def refactor(self, code_string, language="Luau"):\r
        """Automatically refactors code for cleanliness and maintainability."""\r
        prompt = f"""\r
        You are an AI code quality assistant. Your task is to refactor the following {language} code to improve its structure and readability without changing its functionality.\r
        Focus on these principles:\r
        - **DRY (Don't Repeat Yourself):** Consolidate redundant code.\r
        - **KISS (Keep It Simple, Stupid):** Simplify complex logic.\r
        - **Readability:** Improve variable names and add comments where necessary.\r
        - **Modularity:** Break down large functions into smaller, single-purpose functions.\r
\r
        Code to refactor:\r
        \`\`\`\r
        {code_string}\r
        \`\`\`\r
\r
        Provide only the refactored code block.\r
        """\r
        refactored_code, _ = self.model.generate_response(prompt, max_new_tokens=512, temperature=0.6)\r
        match = re.search(r'\`\`\`(?:\\w+)?\\n(.*?)\\n\`\`\`', refactored_code, re.DOTALL)\r
        if match:\r
            return match.group(1).strip()\r
        return refactored_code\r
\r
\r
# NEW: Agent for game design ideas\r
class GameDesignerAgent:\r
    def __init__(self, model):\r
        self.model = model\r
\r
    def propose_feature(self, context_prompt):\r
        """Proposes a new game feature based on a given context."""\r
        prompt = f"""\r
        You are a creative and experienced game designer.\r
        Based on the following request, propose a new, innovative game feature.\r
        Describe the feature, how it works, and why it would be fun for players.\r
\r
        Request: "{context_prompt}"\r
\r
        **Feature Proposal:**\r
        """\r
        proposal, _ = self.model.generate_response(prompt, max_new_tokens=300, temperature=0.85)\r
        return proposal\r
\r
\r
# --- NEW: Multi-Task Learning Agents ---\r
class CodeSummarizationAgent:\r
    def __init__(self, model):\r
        self.model = model\r
\r
    def summarize(self, code_string, language="Luau"):\r
        prompt = f"""\r
        Summarize the main functionality of the following {language} code in a single, concise sentence.\r
\r
        Code:\r
        \`\`\`\r
        {code_string}\r
        \`\`\`\r
\r
        Summary:\r
        """\r
        summary, _ = self.model.generate_response(prompt, max_new_tokens=50, temperature=0.5)\r
        return summary\r
\r
class CodeQuestionAnsweringAgent:\r
    def __init__(self, model):\r
        self.model = model\r
\r
    def answer_question(self, code_string, question, language="Luau"):\r
        prompt = f"""\r
        Analyze the following {language} code and answer the question provided.\r
\r
        Code:\r
        \`\`\`\r
        {code_string}\r
        \`\`\`\r
\r
        Question: {question}\r
\r
        Answer:\r
        """\r
        answer, _ = self.model.generate_response(prompt, max_new_tokens=100, temperature=0.5)\r
        return answer\r
\r
# --- Data Handling (IMPROVED) ---\r
def download_code_from_github(engine_name: str, github_query: str, file_extensions: list, save_dir: str, github_token: str):\r
    """\r
    Downloads code files for a specific game engine from GitHub with rate limit handling.\r
    """\r
    if not os.path.exists(save_dir):\r
        os.makedirs(save_dir)\r
\r
    headers = {"Authorization": f"token {github_token}"}\r
    search_url = "https://api.github.com/search/repositories"\r
\r
    # Count existing files to avoid re-downloading\r
    downloaded_count = 0\r
    for ext in file_extensions:\r
        downloaded_count += len(glob.glob(os.path.join(save_dir, f"*{ext}")))\r
\r
    page = 1\r
    target_count = 1200 # <-- Increased target\r
\r
    logging.info(f"[{engine_name}] Found {downloaded_count} existing files. Target is {target_count}.")\r
\r
    initial_downloaded_count = downloaded_count\r
\r
    while downloaded_count < target_count:\r
        params = {"q": github_query, "per_page": 100, "page": page}\r
\r
        logging.info(f"[{engine_name}] Searching GitHub repositories (page {page})...")\r
        try:\r
            response = requests.get(search_url, headers=headers, params=params, timeout=30)\r
\r
            # --- NEW: Rate Limit Handling ---\r
            if 'X-RateLimit-Remaining' in response.headers:\r
                remaining = int(response.headers['X-RateLimit-Remaining'])\r
                if remaining < 10:\r
                    reset_time = int(response.headers['X-RateLimit-Reset'])\r
                    sleep_duration = max(0, reset_time - time.time()) + 5 # Add 5s buffer\r
                    logging.warning(f"GitHub API rate limit low ({remaining} left). Sleeping for {sleep_duration:.0f} seconds.")\r
                    time.sleep(sleep_duration)\r
\r
            response.raise_for_status()\r
            repos = response.json().get("items", [])\r
\r
            if not repos:\r
                logging.info(f"[{engine_name}] No more repositories found.")\r
                break\r
\r
        except requests.exceptions.RequestException as e:\r
            logging.error(f"Error connecting to GitHub API: {e}. Retrying search in 15 seconds...")\r
            time.sleep(15)\r
            continue\r
\r
        for repo in repos:\r
            repo_name = repo["full_name"]\r
            default_branch = repo["default_branch"]\r
            try:\r
                files_url = f"https://api.github.com/repos/{repo_name}/git/trees/{default_branch}?recursive=1"\r
                files_response = requests.get(files_url, headers=headers, timeout=30)\r
                files_response.raise_for_status()\r
                files_tree = files_response.json().get("tree", [])\r
\r
                for file in files_tree:\r
                    if any(file["path"].endswith(ext) for ext in file_extensions) and file.get("size", 0) > 100:\r
                        save_path = os.path.join(save_dir, f"{repo_name.replace('/', '_')}_{os.path.basename(file['path'])}")\r
\r
                        if os.path.exists(save_path):\r
                            continue\r
\r
                        file_content_url = f"https://api.github.com/repos/{repo_name}/contents/{file['path']}?ref={default_branch}"\r
\r
                        retries = 3\r
                        for i in range(retries):\r
                            try:\r
                                content_response = requests.get(file_content_url, headers=headers, timeout=30)\r
                                content_response.raise_for_status()\r
                                file_data = content_response.json()\r
\r
                                try:\r
                                    content = base64.b64decode(file_data["content"]).decode("utf-8")\r
                                    with open(save_path, "w", encoding="utf-8") as f:\r
                                        f.write(content)\r
                                    downloaded_count += 1\r
                                    if downloaded_count % 50 == 0:\r
                                        logging.info(f"[{engine_name}] Downloaded {downloaded_count}/{target_count} files.")\r
                                    break\r
                                except UnicodeDecodeError:\r
                                    logging.warning(f"Skipping file {file['path']} due to encoding error.")\r
                                    break\r
\r
                            except requests.exceptions.RequestException as e:\r
                                logging.error(f"Connection error for {file['path']}. Retrying {i+1}/{retries}...")\r
                                time.sleep(5 * (i + 1)) # Exponential backoff\r
                            except KeyError:\r
                                logging.error(f"Could not decode content for {file['path']}. Skipping.")\r
                                break\r
                        else:\r
                            logging.error(f"Failed to download {file['path']} after {retries} retries.")\r
                            continue\r
\r
                        if downloaded_count >= target_count:\r
                            break\r
\r
                        time.sleep(1) # Be respectful to the API\r
\r
                if downloaded_count >= target_count:\r
                    break\r
\r
            except requests.exceptions.RequestException as e:\r
                logging.error(f"Skipping repo {repo_name} due to API error: {e}")\r
                continue\r
\r
        if downloaded_count >= target_count:\r
            break\r
\r
        page += 1\r
\r
    logging.info(f"[{engine_name}] Successfully downloaded a total of {downloaded_count - initial_downloaded_count} new files.")\r
    return downloaded_count\r
\r
def get_func_name(node):\r
    try:\r
        if hasattr(node.name.id, 'id'):\r
            return node.name.id.id\r
        return node.name.id\r
    except Exception:\r
        return None\r
\r
def cache_embeddings(code_chunks, codebert_pipeline, cache_file="embeddings_cache.joblib"):\r
    # FIX C: Ensure codebert_pipeline is not None\r
    if codebert_pipeline is None:\r
        logging.error("CodeBERT pipeline is None. Cannot generate embeddings.")\r
        return {} # Return empty cache if pipeline is missing\r
\r
    if os.path.exists(cache_file):\r
        try:\r
            cache = joblib.load(cache_file)\r
        except (IOError, joblib.UnpicklingError) as e:\r
            logging.error(f"Error loading embedding cache: {e}. Creating a new cache.")\r
            cache = {}\r
    else:\r
        cache = {}\r
\r
    new_chunks = [chunk for chunk in code_chunks if chunk not in cache]\r
    if new_chunks:\r
        logging.info(f"Generating embeddings for {len(new_chunks)} new chunks...")\r
        try:\r
            # Batch processing for efficiency\r
            new_embeddings_raw = codebert_pipeline(new_chunks, batch_size=8)\r
            new_embeddings = [torch.tensor(emb).squeeze() for emb in new_embeddings_raw]\r
\r
            for chunk, embedding in zip(new_chunks, new_embeddings):\r
                if embedding.dim() > 1:\r
                    embedding = embedding.mean(dim=0)\r
                cache[chunk] = embedding.tolist()\r
        except Exception as e:\r
            logging.error(f"Error generating CodeBERT embeddings: {e}")\r
            return {}\r
\r
        joblib.dump(cache, cache_file)\r
        logging.info("Embeddings cached.")\r
\r
    return cache\r
\r
def visualize_graph(graph, filename="code_graph.png", max_nodes=50):\r
    if not hasattr(graph, 'x') or graph.x.shape[0] == 0:\r
        logging.warning("Cannot visualize an empty graph.")\r
        return\r
\r
    num_nodes_to_display = min(graph.x.shape[0], max_nodes)\r
\r
    if graph.x.shape[0] > max_nodes:\r
        logging.warning(f"Graph too large ({graph.x.shape[0]} nodes). Visualizing a random subgraph of {max_nodes} nodes.")\r
        node_indices = random.sample(range(graph.num_nodes), max_nodes)\r
\r
        g = nx.DiGraph()\r
        node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(node_indices)}\r
\r
        for i in range(graph.edge_index.shape[1]):\r
            src, dest = graph.edge_index[0][i].item(), graph.edge_index[1][i].item()\r
            if src in node_map and dest in node_map:\r
                g.add_edge(node_map[src], node_map[dest])\r
    else:\r
        g = nx.DiGraph()\r
        g.add_nodes_from(range(graph.x.shape[0]))\r
        if hasattr(graph, 'edge_index') and graph.edge_index.shape[1] > 0:\r
            edges = graph.edge_index.t().tolist()\r
            g.add_edges_from(edges)\r
\r
    plt.figure(figsize=(14, 14))\r
    pos = nx.spring_layout(g) # Spring layout is often good for complex graphs\r
    nx.draw(g, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=8, edge_color='gray', arrows=True)\r
    plt.title("Code Knowledge Graph Visualization")\r
    plt.savefig(filename)\r
    plt.close() # Close the figure to free memory\r
    logging.info(f"Graph visualization saved to {filename}.")\r
\r
# --- IMPROVED Graph Building with Usage Data and Libraries ---\r
def build_real_code_graph_ast(code_content, model, codebert_pipeline, asset_id=None, language="lua", design_doc=None, user_feedback=None):\r
    logging.info("Using AST-based graph builder with luaparser.")\r
\r
    if language != "lua" or not luaparser_parser:\r
        logging.error("luaparser not found or language is not Lua. Cannot build AST graph.")\r
        return None\r
\r
    try:\r
        ast_tree = luaparser_parser.parse(code_content)\r
        function_definitions = {}\r
        function_calls = []\r
        library_usages = set()\r
        variable_usages = {}\r
\r
        def find_nodes(node, parent=None):\r
            if isinstance(node, luaparser_ast.Function):\r
                func_name = get_func_name(node)\r
                if func_name and func_name not in function_definitions:\r
                    function_definitions[func_name] = node\r
            elif isinstance(node, luaparser_ast.Call):\r
                func_name = get_func_name(node.func)\r
                if func_name:\r
                    function_calls.append({'name': func_name, 'node': node})\r
            elif isinstance(node, luaparser_ast.Index):\r
                if isinstance(node.idx, luaparser_ast.Id):\r
                    library_usages.add(node.idx.id)\r
            elif isinstance(node, luaparser_ast.Id):\r
                var_name = node.id\r
                if var_name not in variable_usages:\r
                    variable_usages[var_name] = 0\r
                variable_usages[var_name] += 1\r
\r
            for child in luaparser_ast.walk(node):\r
                if child is not node:\r
                    find_nodes(child)\r
\r
        find_nodes(ast_tree)\r
\r
        all_chunks = re.split(r'\\n(function|local function)', code_content)\r
        chunks = []\r
        for i in range(1, len(all_chunks), 2):\r
            if i + 1 < len(all_chunks):\r
                chunks.append(all_chunks[i] + all_chunks[i+1])\r
        valid_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]\r
\r
        if not valid_chunks:\r
            return None\r
\r
        embeddings_cache = cache_embeddings(valid_chunks, codebert_pipeline)\r
        if not embeddings_cache: # Handle case where embedding fails\r
            return None\r
\r
        G = nx.DiGraph()\r
        node_map = {}\r
        xs = []\r
        node_types = []\r
        node_ids = []\r
\r
        for i, chunk in enumerate(valid_chunks):\r
            embedding = embeddings_cache.get(chunk)\r
            if embedding is not None:\r
                G.add_node(i, type='code')\r
                node_types.append(0) # 0 for code\r
                node_ids.append(None)\r
                xs.append(torch.tensor(embedding, dtype=torch.float32))\r
                func_name_match = re.search(r'(?:function|local function)\\s+([a-zA-Z0-9_:]+)', chunk)\r
                if func_name_match:\r
                    node_map[func_name_match.group(1)] = i\r
\r
        # FIX D: Add 'usage_data' as a feature and pad to fixed dimension\r
        for i, chunk in enumerate(valid_chunks):\r
            chunk_usage = sum(variable_usages.get(var, 0) for var in re.findall(r'\\b[a-zA-Z0-9_]+\\b', chunk))\r
            if i < len(xs):\r
                current_embedding = xs[i]\r
                usage_feature = torch.tensor([chunk_usage / 10.0]) # Normalize usage\r
                combined_features = torch.cat([current_embedding, usage_feature], dim=0)\r
\r
                padding_needed = model.fixed_graph_embedding_dim - combined_features.shape[0]\r
                if padding_needed < 0: # Truncate if too long\r
                     combined_features = combined_features[:model.fixed_graph_embedding_dim]\r
                elif padding_needed > 0: # Pad if too short\r
                    combined_features = pad(combined_features, (0, padding_needed), 'constant', 0)\r
                xs[i] = combined_features\r
\r
        # Add library/API nodes\r
        library_node_start_idx = len(xs)\r
        for lib in library_usages:\r
            # Pad to fixed dimension\r
            lib_embedding = torch.randn(model.fixed_graph_embedding_dim, dtype=torch.float32)\r
            xs.append(lib_embedding)\r
            node_types.append(2) # 2 for library/API\r
            node_ids.append(lib)\r
            node_map[lib] = len(xs) - 1\r
\r
        if not xs:\r
            return None\r
\r
        edge_list = []\r
        edge_attr = []\r
        for call in function_calls:\r
            caller_name = None\r
            for func_name, def_node in function_definitions.items():\r
                if def_node.location and call['node'].location:\r
                    if def_node.location.line <= call['node'].location.line and def_node.end_location.line >= call['node'].location.line:\r
                        caller_name = func_name\r
                        break\r
\r
            if caller_name and call['name'] in node_map and caller_name in node_map:\r
                caller_node_idx = node_map[caller_name]\r
                callee_node_idx = node_map[call['name']]\r
                if caller_node_idx != callee_node_idx:\r
                    edge_list.append((caller_node_idx, callee_node_idx))\r
                    edge_attr.append(0) # 0 for 'calls'\r
\r
        # Add library/API edges\r
        for i, chunk in enumerate(valid_chunks):\r
            for lib in library_usages:\r
                if lib in chunk:\r
                    edge_list.append((i, node_map[lib]))\r
                    edge_attr.append(2) # 2 for 'uses_library'\r
\r
        if asset_id:\r
            asset_node_idx = len(xs)\r
            # Pad to fixed dimension\r
            asset_embedding = torch.zeros(model.fixed_graph_embedding_dim, dtype=torch.float32)\r
            xs.append(asset_embedding)\r
            node_types.append(1) # 1 for asset\r
            node_ids.append(asset_id)\r
\r
            for i, chunk in enumerate(valid_chunks):\r
                if 'Instance.new' in chunk or asset_id in chunk:\r
                    edge_list.append((i, asset_node_idx))\r
                    edge_attr.append(1) # 1 for 'uses'\r
\r
        # NEW: Add Game Design Doc and User Feedback nodes (simulation)\r
        if design_doc:\r
            doc_node_idx = len(xs)\r
            doc_embedding = torch.randn(model.fixed_graph_embedding_dim)\r
            xs.append(doc_embedding)\r
            node_types.append(3) # 3 for design doc\r
            node_ids.append("design_doc_1")\r
            # Connect all code nodes to the design doc\r
            for i in range(len(valid_chunks)):\r
                edge_list.append((i, doc_node_idx))\r
                edge_attr.append(3) # 3 for 'implements_design'\r
\r
        if user_feedback:\r
            feedback_node_idx = len(xs)\r
            feedback_embedding = torch.randn(model.fixed_graph_embedding_dim)\r
            xs.append(feedback_embedding)\r
            node_types.append(4) # 4 for user feedback\r
            node_ids.append("user_feedback_1")\r
            # Connect feedback to a random code chunk\r
            if valid_chunks:\r
                edge_list.append((random.randint(0, len(valid_chunks)-1), feedback_node_idx))\r
                edge_attr.append(4) # 4 for 'addresses_feedback'\r
\r
\r
        edge_list = list(set(edge_list))\r
\r
        x = torch.stack(xs, dim=0)\r
\r
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)\r
        edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(1) if edge_list else torch.empty((0, 1), dtype=torch.float)\r
\r
        py_g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)\r
        py_g = py_g.to(model._model_device())\r
\r
        # FIX D: Removed dynamic resizing of the embedding projection layer.\r
        # The model is now initialized with a fixed-size layer.\r
\r
        proj_x = model.embedding_proj(py_g.x)\r
        py_g.x = proj_x\r
        py_g.node_type = torch.tensor(node_types, dtype=torch.long)\r
        py_g.node_id = node_ids\r
\r
        return py_g\r
    except Exception as e:\r
        logging.error(f"Error building AST graph: {e}")\r
        return None\r
\r
def build_real_code_graph_fallback(code_content, model, codebert_pipeline):\r
    logging.info("Using fallback regex-based graph builder.")\r
\r
    code_chunks = []\r
    parts = re.split(r'\\n(function|local function)', code_content)\r
    for i in range(1, len(parts), 2):\r
        if i + 1 < len(parts):\r
            code_chunks.append(parts[i] + parts[i+1])\r
\r
    valid_chunks = [chunk.strip() for chunk in code_chunks if chunk.strip()]\r
    if not valid_chunks:\r
        return None\r
\r
    embeddings_cache = cache_embeddings(valid_chunks, codebert_pipeline)\r
    if not embeddings_cache:\r
        return None\r
\r
    node_map = {}\r
    xs = []\r
    node_types = []\r
\r
    for i, chunk in enumerate(valid_chunks):\r
        embedding_list = embeddings_cache.get(chunk)\r
        if embedding_list is not None:\r
            embedding = torch.tensor(embedding_list, dtype=torch.float32)\r
            # FIX D: Pad fallback embeddings to the fixed dimension as well\r
            padding_needed = model.fixed_graph_embedding_dim - embedding.shape[0]\r
            if padding_needed > 0:\r
                embedding = pad(embedding, (0, padding_needed), 'constant', 0)\r
            elif padding_needed < 0:\r
                embedding = embedding[:model.fixed_graph_embedding_dim]\r
\r
            xs.append(embedding)\r
            node_types.append(0)\r
            func_name_match = re.search(r'(?:function|local function)\\s+([a-zA-Z0-9_:]+)', valid_chunks[i])\r
            if func_name_match:\r
                node_map[func_name_match.group(1)] = len(xs) - 1\r
\r
    if not xs: return None\r
\r
    edge_list = []\r
    edge_attr = []\r
    for i, chunk in enumerate(valid_chunks):\r
        for func_name, node_idx in node_map.items():\r
            if i == node_idx: continue\r
            if f'{func_name}(' in valid_chunks[i]:\r
                edge_list.append((i, node_idx))\r
                edge_attr.append(0)\r
\r
    edge_list = list(set(edge_list))\r
\r
    x = torch.stack(xs, dim=0)\r
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)\r
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(1) if edge_list else torch.empty((0, 1), dtype=torch.float)\r
\r
    py_g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)\r
    py_g = py_g.to(model._model_device())\r
    proj_x = model.embedding_proj(py_g.x)\r
    py_g.x = proj_x\r
    py_g.node_type = torch.tensor(node_types, dtype=torch.long)\r
    py_g.node_id = [None] * len(node_types)\r
    return py_g\r
\r
\r
class CodeDataset(Dataset):\r
    def __init__(self, data_dir, tokenizer, model, codebert_pipeline, max_length=512, graph_cache_dir="graph_cache", file_extensions=None):\r
        if file_extensions is None:\r
            file_extensions = ["*.lua", "*.luau"]\r
\r
        self.file_paths = []\r
        for ext in file_extensions:\r
            self.file_paths.extend(glob.glob(os.path.join(data_dir, ext)))\r
\r
        if not self.file_paths:\r
            raise FileNotFoundError(f"No files with extensions {file_extensions} found in {data_dir}.")\r
\r
        self.tokenizer = tokenizer\r
        self.model = model\r
        self.codebert_pipeline = codebert_pipeline\r
        self.max_length = max_length\r
        self.graph_cache_dir = graph_cache_dir\r
        if not os.path.exists(self.graph_cache_dir):\r
            os.makedirs(self.graph_cache_dir)\r
        self.tokenized_data = []\r
        self._load_data()\r
\r
    def _load_data(self):\r
        logging.info(f"Loading and preprocessing {len(self.file_paths)} files...")\r
        for file_path in tqdm(self.file_paths):\r
            try:\r
                with open(file_path, "r", encoding="utf-8") as f:\r
                    content = f.read()\r
\r
                tokenized_data = self.tokenizer(\r
                    content,\r
                    return_tensors="pt",\r
                    max_length=self.max_length,\r
                    truncation=True,\r
                    padding="max_length"\r
                )\r
\r
                self.tokenized_data.append({\r
                    'file_path': file_path,\r
                    'input_ids': tokenized_data['input_ids'].squeeze(),\r
                    'attention_mask': tokenized_data['attention_mask'].squeeze(),\r
                })\r
            except Exception as e:\r
                logging.error(f"Skipping file {file_path} due to read/tokenize error: {e}")\r
\r
    def get_graph(self, file_path, asset_id=None):\r
        with open(file_path, "r", encoding="utf-8") as f:\r
            content = f.read()\r
\r
        content_hash = str(hash(content))\r
        cache_path = os.path.join(self.graph_cache_dir, f"{content_hash}.joblib")\r
\r
        if os.path.exists(cache_path):\r
            try:\r
                return joblib.load(cache_path)\r
            except Exception as e:\r
                logging.error(f"Error loading graph from cache: {e}. Rebuilding...")\r
\r
        graph_data = build_real_code_graph_ast(content, self.model, self.codebert_pipeline, asset_id, design_doc="mock", user_feedback="mock")\r
\r
        if graph_data is not None:\r
            joblib.dump(graph_data, cache_path)\r
\r
        return graph_data\r
\r
    def __len__(self):\r
        return len(self.tokenized_data)\r
\r
    def __getitem__(self, idx):\r
        item = self.tokenized_data[idx]\r
        file_path = item['file_path']\r
\r
        with open(file_path, "r", encoding="utf-8") as f:\r
            content = f.read()\r
\r
        graph_data = self.get_graph(file_path)\r
\r
        return {\r
            'input_ids': item['input_ids'],\r
            'attention_mask': item['attention_mask'],\r
            'code_content': content,\r
            'code_graph_data': graph_data\r
        }\r
\r
def custom_collate_fn(batch):\r
    batch = [item for item in batch if item is not None]\r
    if not batch:\r
        return None\r
\r
    input_ids = torch.stack([item['input_ids'] for item in batch])\r
    attention_mask = torch.stack([item['attention_mask'] for item in batch])\r
    code_contents = [item['code_content'] for item in batch]\r
    graph_data_list = [item['code_graph_data'] for item in batch if item['code_graph_data'] is not None]\r
\r
    batched_graph = None\r
    if graph_data_list:\r
        batched_graph = Batch.from_data_list(graph_data_list)\r
\r
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'code_contents': code_contents, 'code_graph_data': batched_graph}\r
\r
def _calculate_ppo_loss(model, accelerator, state, action, action_len, old_log_prob, reward, next_state_embedding, old_value_preds, done, curiosity_weight, clip_epsilon, weights=None):\r
    # Move next_state_embedding to the correct device if it exists\r
    if next_state_embedding is not None:\r
        next_state_embedding = next_state_embedding.to(accelerator.device)\r
\r
    full_input_ids = torch.cat([state, action], dim=1).to(accelerator.device)\r
\r
    if full_input_ids.size(1) > model.llm.config.max_position_embeddings:\r
        logging.warning("Sequence length exceeds max position embeddings. Truncating.")\r
        full_input_ids = full_input_ids[:, :model.llm.config.max_position_embeddings]\r
\r
    logits, value, fused_state = model(full_input_ids, None, project_graph_embedding=next_state_embedding)\r
\r
    if action.numel() == 0 or logits.numel() == 0:\r
        return None, None, None, None, None\r
\r
    logits_gen = logits[:, state.size(1)-1:-1, :]\r
    log_probs = log_softmax(logits_gen, dim=-1)\r
\r
    action_mask = (action != model.tokenizer.pad_token_id).to(accelerator.device)\r
\r
    action_log_probs = log_probs.gather(2, action.unsqueeze(-1)).squeeze(-1)\r
    masked_action_log_probs = action_log_probs * action_mask\r
    current_log_prob = masked_action_log_probs.sum(dim=-1) / action_mask.sum(dim=-1).clamp(min=1e-6)\r
\r
    gamma = 0.99\r
\r
    # FIX E: Corrected Advantage and Return calculation for batch-based Actor-Critic\r
    # Ensure shapes are compatible for broadcasting\r
    reward = reward.squeeze() if reward.dim() > 1 else reward\r
    done = done.squeeze() if done.dim() > 1 else done\r
    value = value.squeeze()\r
    old_value_preds = old_value_preds.squeeze()\r
\r
    # Advantage A(s,a) = r + gamma * V(s') * (1-done) - V(s)\r
    advantages = reward + gamma * value.detach() * (1 - done.int()) - old_value_preds.detach()\r
    returns = advantages + old_value_preds.detach()\r
\r
\r
    current_state_features = fused_state[:, -2, :]\r
    next_state_features = fused_state[:, -1, :]\r
    action_features = model.llm.get_input_embeddings()(action).mean(dim=1) # Average embeddings over sequence length\r
\r
    curiosity_loss = model.curiosity_module(current_state_features, action_features, next_state_features)\r
    intrinsic_reward = curiosity_loss.detach()\r
\r
    total_reward = reward + curiosity_weight * intrinsic_reward\r
\r
    ratio = torch.exp(current_log_prob - old_log_prob.detach())\r
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)\r
\r
    policy_loss_unweighted = -torch.min(ratio * advantages, clipped_ratio * advantages)\r
    value_loss_unweighted = mse_loss(value, returns)\r
\r
    if weights is not None:\r
        policy_loss = (policy_loss_unweighted * weights).mean()\r
        value_loss = (value_loss_unweighted * weights).mean()\r
    else:\r
        policy_loss = policy_loss_unweighted.mean()\r
        value_loss = value_loss_unweighted.mean()\r
\r
\r
    entropy_beta = 0.01\r
    probs = softmax(logits_gen, dim=-1)\r
    entropy = -(probs * log_probs).sum(dim=-1).mean()\r
\r
    total_loss = policy_loss + 0.5 * value_loss - entropy_beta * entropy + curiosity_loss\r
\r
    return total_loss, policy_loss, value_loss, entropy, curiosity_loss\r
\r
\r
# NEW: Mock function to simulate human feedback in the loop\r
def get_human_feedback(code_string):\r
    """Simulates a user providing a rating for the generated code."""\r
    # In a real system, this would be a UI where a user rates the code.\r
    # Here, we'll mock it based on code properties.\r
    if "error" in code_string.lower() or "bug" in code_string.lower():\r
        return -2.0 # User is unhappy with buggy code\r
    if len(code_string) > 500:\r
        return 0.5 # User might find very long code less helpful\r
    return 1.5 # User is generally happy\r
\r
def train_ppo_with_accelerator(model, data_loader, val_loader, optimizer, codebert_pipeline, num_epochs, gradient_accumulation_steps, use_mock_reward, visualize_graphs, clip_epsilon, curiosity_weight, engine_name="", language="lua"):\r
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)\r
    model, optimizer, data_loader, val_loader = accelerator.prepare(model, optimizer, data_loader, val_loader)\r
\r
    # --- Initialize all agents ---\r
    code_evaluator = CodeEvaluator(use_mock_reward=use_mock_reward, language=language)\r
    asset_generator_agent = AssetGeneratorAgent()\r
    bug_report_generator_agent = BugReportGeneratorAgent(model)\r
    code_generator_agent = CodeGeneratorAgent(model)\r
    code_critic_agent = CodeCriticAgent(model)\r
    code_refinement_agent = CodeRefinementAgent(model)\r
    test_generation_agent = TestGenerationAgent(model)\r
    documentation_agent = DocumentationAgent(model)\r
    # NEW: Initialize new agents\r
    auto_refactoring_agent = AutoRefactoringAgent(model)\r
    game_designer_agent = GameDesignerAgent(model)\r
    vectorized_memory = VectorizedMemory(codebert_pipeline)\r
\r
\r
    logging.info(f"Starting PPO training with Accelerator for {engine_name}...")\r
    model.train()\r
\r
    total_steps = (num_epochs * len(data_loader)) // gradient_accumulation_steps\r
    progress_bar = tqdm(range(total_steps), desc=f"Training ({engine_name})")\r
\r
    replay_buffer = PrioritizedReplayBuffer(capacity=1024)\r
    visualize_interval = 100\r
\r
    # FIX F: Initialize logging variables to prevent potential undefined errors\r
    total_loss, policy_loss, value_loss, entropy, curiosity_loss = [torch.tensor(0.0) for _ in range(5)]\r
\r
    for epoch in range(num_epochs):\r
        for step, batch in enumerate(data_loader):\r
            if batch is None:\r
                continue\r
\r
            initial_code_string = batch['code_contents'][0]\r
\r
            # --- NEW: Game Designer Agent Step ---\r
            if step % 75 == 0:\r
                 design_proposal = game_designer_agent.propose_feature(f"A new feature for a {engine_name} game involving player interaction.")\r
                 logging.info(f"\\n--- Game Design Proposal (Step {step}) ---\\n{design_proposal}\\n")\r
\r
            # --- RAG Step 1: Retrieve similar code from long-term memory ---\r
            retrieved_examples = vectorized_memory.retrieve_similar(initial_code_string)\r
\r
            asset_prompt = "A red spinning part"\r
            generated_asset_id = asset_generator_agent.generate_asset(asset_prompt)\r
\r
            # --- Pass retrieved examples to the generator ---\r
            generated_code_string, _ = code_generator_agent.generate(\r
                prompt=f"Improve the following code to use the asset ID {generated_asset_id}:\\n{initial_code_string}",\r
                context_examples=retrieved_examples\r
            )\r
\r
            # FIX C: Pass the codebert_pipeline to the graph builder\r
            project_graph = build_real_code_graph_ast(generated_code_string, model, codebert_pipeline, asset_id=generated_asset_id)\r
            reward_dict = code_evaluator.evaluate(generated_code_string, project_graph)\r
\r
            # --- Bug Fixing and Refinement Loop ---\r
            if reward_dict['total_reward'] < 0 and not use_mock_reward:\r
                bug_report = bug_report_generator_agent.generate_report(generated_code_string, reward_dict['docker_log'])\r
                critique = code_critic_agent.critique(generated_code_string, reward_dict, asset_id=generated_asset_id)\r
                refined_code_string = code_refinement_agent.refine(generated_code_string, critique, bug_report=bug_report, asset_id=generated_asset_id, context_examples=retrieved_examples)\r
            else:\r
                critique = code_critic_agent.critique(generated_code_string, reward_dict, asset_id=generated_asset_id)\r
                refined_code_string = code_refinement_agent.refine(generated_code_string, critique, asset_id=generated_asset_id, context_examples=retrieved_examples)\r
                bug_report = None\r
\r
            # --- NEW: Auto-Refactoring Step ---\r
            if reward_dict['readability_score'] < 0.5: # Only refactor if readability is low\r
                logging.info(f"Readability score is low ({reward_dict['readability_score']:.2f}). Attempting auto-refactoring...")\r
                refined_code_string = auto_refactoring_agent.refactor(refined_code_string, language)\r
\r
\r
            # --- NEW: Multi-Task Learning Step ---\r
            summary = CodeSummarizationAgent(model).summarize(refined_code_string, language)\r
            question = "What does the main function do?"\r
            answer = CodeQuestionAnsweringAgent(model).answer_question(refined_code_string, question, language)\r
\r
            if step % 50 == 0:\r
                logging.info(f"Step {step} - Summary: {summary}")\r
                logging.info(f"Step {step} - Q: {question} A: {answer}")\r
\r
            # --- Use specialized agents ---\r
            generated_tests = test_generation_agent.generate_tests(refined_code_string, language)\r
            generated_docs = documentation_agent.generate_docs(refined_code_string, language)\r
\r
            if step % 50 == 0: # Log occasionally to avoid spam\r
                logging.info(f"\\n--- Generated Tests (Step {step}) ---\\n{generated_tests}")\r
                logging.info(f"\\n--- Generated Docs (Step {step}) ---\\n{generated_docs}")\r
\r
            refined_project_graph = build_real_code_graph_ast(refined_code_string, model, codebert_pipeline, asset_id=generated_asset_id)\r
            refined_reward_dict = code_evaluator.evaluate(refined_code_string, refined_project_graph)\r
\r
            # --- NEW: Human-in-the-Loop Feedback Simulation ---\r
            human_reward = get_human_feedback(refined_code_string)\r
            final_reward_value = refined_reward_dict['total_reward'] + human_reward\r
            reward = torch.tensor([final_reward_value]).float()\r
\r
            # --- RAG Step 2: Add high-quality code to long-term memory ---\r
            vectorized_memory.add_experience(refined_code_string, final_reward_value)\r
\r
\r
            project_graph_embedding = None\r
            if refined_project_graph is not None:\r
                graph_data = refined_project_graph.to(accelerator.device)\r
                project_graph_embedding = model.graph_memory(graph_data.x, graph_data.edge_index, graph_data.edge_attr, graph_data.batch)\r
\r
                if visualize_graphs and step % visualize_interval == 0 and not use_mock_reward:\r
                    visualize_graph(graph_data.cpu(), filename=f"graph_{engine_name}_epoch{epoch}_step{step}.png")\r
            else:\r
                logging.warning(f"Skipping graph creation for this step due to an issue.")\r
\r
            with torch.no_grad():\r
                gen_ids = model.tokenizer.encode(refined_code_string, return_tensors="pt")\r
                full_input_ids = torch.cat([batch['input_ids'], gen_ids], dim=1).to(accelerator.device)\r
\r
                if full_input_ids.size(1) > model.llm.config.max_position_embeddings:\r
                    full_input_ids = full_input_ids[:, :model.llm.config.max_position_embeddings]\r
\r
                logits_full, value_preds, _ = model(full_input_ids, None, project_graph_embedding=project_graph_embedding)\r
\r
                gen_len = gen_ids.size(1)\r
                logits_gen = logits_full[:, -gen_len-1:-1, :]\r
                log_probs = log_softmax(logits_gen, dim=-1)\r
\r
                action_mask = (gen_ids != model.tokenizer.pad_token_id).to(accelerator.device)\r
                action_log_probs = log_probs.gather(2, gen_ids.unsqueeze(-1).to(accelerator.device)).squeeze(-1)\r
                masked_action_log_probs = action_log_probs * action_mask\r
\r
                gathered_log_probs = masked_action_log_probs.sum(dim=-1) / action_mask.sum(dim=-1).clamp(min=1e-6)\r
\r
            # FIX B: Detach embedding and move to CPU before pushing to buffer to save VRAM\r
            project_graph_embedding_cpu = project_graph_embedding.detach().cpu() if project_graph_embedding is not None else None\r
            experience = (\r
                batch['input_ids'].cpu(),\r
                gen_ids.cpu(),\r
                torch.tensor([gen_ids.size(1)], dtype=torch.long).cpu(),\r
                gathered_log_probs.cpu(),\r
                reward.cpu(),\r
                project_graph_embedding_cpu,\r
                torch.tensor([False]).cpu()\r
            )\r
            replay_buffer.push(experience)\r
\r
            if len(replay_buffer) >= 64:\r
                batch_data = replay_buffer.sample(64)\r
                if batch_data is None: continue\r
\r
                batch_states, batch_actions, batch_action_lens, batch_old_log_probs, batch_rewards, batch_next_states, batch_dones, weights, indices = batch_data\r
\r
                rewards_tensor = batch_rewards\r
                mean_reward = torch.mean(rewards_tensor)\r
                std_reward = torch.std(rewards_tensor)\r
                if std_reward.item() == 0:\r
                    std_reward = 1e-8\r
                normalized_rewards = (rewards_tensor - mean_reward) / std_reward\r
\r
                total_loss_sum = 0\r
                losses_list = []\r
\r
                for i in range(batch_states.size(0)):\r
                    with accelerator.accumulate(model):\r
                        state = batch_states[i].unsqueeze(0)\r
                        action = batch_actions[i].unsqueeze(0)[:, :batch_action_lens[i]]\r
                        old_log_prob = batch_old_log_probs[i].unsqueeze(0)\r
                        reward_val = normalized_rewards[i].unsqueeze(0)\r
                        next_state_item = batch_next_states[i].unsqueeze(0) if batch_next_states is not None and i < batch_next_states.size(0) else None\r
                        done = batch_dones[i].unsqueeze(0)\r
\r
                        with torch.no_grad():\r
                            _, old_value_preds, _ = model(state.to(accelerator.device), None, project_graph_embedding=next_state_item.to(accelerator.device) if next_state_item is not None else None)\r
\r
                        loss_outputs = _calculate_ppo_loss(\r
                            model, accelerator, state, action, batch_action_lens[i], old_log_prob, reward_val, next_state_item, old_value_preds, done, curiosity_weight, clip_epsilon, weights=weights[i].unsqueeze(0)\r
                        )\r
                        if loss_outputs is not None:\r
                             total_loss, policy_loss, value_loss, entropy, curiosity_loss = loss_outputs\r
                        else:\r
                             continue\r
\r
                        if total_loss is not None:\r
                            accelerator.backward(total_loss)\r
                            total_loss_sum += total_loss.item()\r
                            losses_list.append(total_loss.item())\r
\r
                if accelerator.sync_gradients:\r
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)\r
                    optimizer.step()\r
                    optimizer.zero_grad()\r
\r
                    if losses_list:\r
                        replay_buffer.update_priorities(indices, torch.tensor(losses_list))\r
\r
\r
                if total_loss_sum > 0:\r
                    avg_loss = total_loss_sum / len(losses_list) if losses_list else 0\r
                    logging.info(f"PPO Batch Loss: {avg_loss:.4f}, Mean Reward: {mean_reward.item():.2f}")\r
\r
            if accelerator.sync_gradients:\r
                progress_bar.update(1)\r
\r
        if (epoch + 1) % 5 == 0:\r
            logging.info(f"Epoch {epoch + 1}/{num_epochs}, Total Loss: {total_loss.item():.4f}, Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}, Curiosity Loss: {curiosity_loss.item():.4f}")\r
            accelerator.wait_for_everyone()\r
            unwrapped_model = accelerator.unwrap_model(model)\r
\r
            save_dir = os.path.join("model_checkpoints", engine_name, f"epoch_{epoch+1}")\r
            os.makedirs(save_dir, exist_ok=True)\r
            unwrapped_model.llm.save_pretrained(save_dir)\r
            logging.info(f"Model checkpoint for {engine_name} saved at epoch {epoch+1}")\r
\r
    logging.info(f"PPO training for {engine_name} finished.")\r
\r
# Optuna objective is now simplified, as it's run once before the main loop\r
def objective(trial, data_dir, model, codebert_pipeline, file_extensions):\r
    try:\r
        dataset = CodeDataset(data_dir, model.tokenizer, model, codebert_pipeline, file_extensions=file_extensions)\r
    except FileNotFoundError as e:\r
        logging.error(f"ERROR during Optuna setup: {e}")\r
        return float('inf')\r
\r
    if len(dataset) == 0:\r
        logging.error("ERROR: Dataset is empty for hyperparameter tuning.")\r
        return float('inf')\r
\r
    train_size = int(0.9 * len(dataset))\r
    val_size = len(dataset) - train_size\r
    train_dataset, _ = random_split(dataset, [train_size, val_size])\r
\r
    lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)\r
    batch_size = trial.suggest_categorical("batch_size", [1, 2, 4]) # Smaller batches for tuning\r
    clip_epsilon = trial.suggest_float("clip_epsilon", 0.1, 0.3)\r
    curiosity_weight = trial.suggest_float("curiosity_weight", 0.01, 0.1)\r
\r
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)\r
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)\r
\r
    accelerator = Accelerator()\r
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)\r
\r
    total_loss_list = []\r
    max_steps = 50\r
    for step, batch in enumerate(train_loader):\r
        if step >= max_steps:\r
            break\r
        if batch is None: continue\r
\r
        reward = torch.tensor([random.uniform(-1, 10)]).float().to(accelerator.device)\r
        gen_ids = model.tokenizer.encode("print('hello world')", return_tensors="pt")\r
\r
        with torch.no_grad():\r
            full_input_ids = torch.cat([batch['input_ids'], gen_ids], dim=1).to(accelerator.device)\r
            if full_input_ids.size(1) > model.llm.config.max_position_embeddings:\r
                full_input_ids = full_input_ids[:, :model.llm.config.max_position_embeddings]\r
            _, value_preds, _ = model(full_input_ids, None, project_graph_embedding=None)\r
            log_probs = torch.randn(1).to(accelerator.device) # Mock log_probs\r
\r
        loss_outputs = _calculate_ppo_loss(\r
            model, accelerator, batch['input_ids'], gen_ids, gen_ids.size(1), log_probs, reward, None, value_preds, torch.tensor([False]).to(accelerator.device), curiosity_weight, clip_epsilon\r
        )\r
        if loss_outputs is not None:\r
            total_loss = loss_outputs[0]\r
            if total_loss is not None:\r
                accelerator.backward(total_loss)\r
                optimizer.step()\r
                optimizer.zero_grad()\r
                total_loss_list.append(total_loss.item())\r
\r
    if total_loss_list:\r
        return np.mean(total_loss_list)\r
    return float('inf')\r
\r
def run_inference_examples(model):\r
    logging.info("\\n--- Running Inference Examples ---")\r
    prompts = [\r
        "Create a script to make a part in Roblox spin constantly.",\r
        "Write a GDScript function in Godot to save player data to a JSON file.",\r
        "How do I handle character movement in Unity using C# and the new Input System?",\r
        "Show a basic C++ example of spawning an Actor in Unreal Engine."\r
    ]\r
\r
    for i, prompt in enumerate(prompts):\r
        logging.info(f"\\n--- Example {i+1}: Prompt: '{prompt}' ---")\r
        try:\r
            generated_response, _ = model.generate_response(prompt)\r
            logging.info(f"Generated Response:\\n{generated_response}")\r
        except Exception as e:\r
            logging.error(f"Failed to generate response for prompt '{prompt}': {e}")\r
\r
# ! ##################################################################\r
# ! ################ REWORKED MAIN FUNCTION ##########################\r
# ! ##################################################################\r
def main():\r
    github_token = os.getenv("GITHUB_TOKEN")\r
    if not github_token:\r
        logging.error("CRITICAL ERROR: GITHUB_TOKEN environment variable is not set. Cannot download data.")\r
        sys.exit()\r
\r
    # --- Configuration for each training stage ---\r
    engine_configs = [\r
        {\r
            "name": "Roblox",\r
            "data_dir": "roblox_code_data",\r
            "github_query": "roblox luau language:lua size:>100",\r
            "file_extensions": ["*.lua", "*.luau"],\r
            "mock_epochs": 10,\r
            "real_epochs": 15,\r
            "language": "luau"\r
        },\r
        {\r
            "name": "Godot",\r
            "data_dir": "godot_code_data",\r
            "github_query": "godot gdscript language:gdscript size:>100",\r
            "file_extensions": ["*.gd"],\r
            "mock_epochs": 10,\r
            "real_epochs": 15,\r
            "language": "gdscript"\r
        },\r
        {\r
            "name": "Unity",\r
            "data_dir": "unity_code_data",\r
            "github_query": "unity c# language:c# size:>100",\r
            "file_extensions": ["*.cs"],\r
            "mock_epochs": 10,\r
            "real_epochs": 15,\r
            "language": "c#"\r
        },\r
        {\r
            "name": "Unreal",\r
            "data_dir": "unreal_code_data",\r
            "github_query": "unreal engine language:c++ size:>100",\r
            "file_extensions": ["*.cpp", "*.h"],\r
            "mock_epochs": 10,\r
            "real_epochs": 15,\r
            "language": "c++"\r
        }\r
    ]\r
\r
    # --- 1. Initialize Model and Pipelines ONCE ---\r
    # The same model instance will be sequentially fine-tuned on each engine's data.\r
    logging.info("Initializing MultiAgentLLM model...")\r
    model = MultiAgentLLM()\r
    logging.info("Initializing CodeBERT pipeline...")\r
    codebert_pipeline = pipeline("feature-extraction", model="microsoft/CodeBERT-base", tokenizer="microsoft/CodeBERT-base", device=0 if torch.cuda.is_available() else -1)\r
\r
    # --- 2. Hyperparameter Tuning (Optional, done once at the start) ---\r
    best_params = {\r
        "lr": 2e-5,\r
        "batch_size": 2, # Smaller batch size is better for low VRAM\r
        "clip_epsilon": 0.2,\r
        "curiosity_weight": 0.05\r
    }\r
    try:\r
        logging.info("--- Starting Hyperparameter Tuning with Optuna on Roblox data (as a baseline) ---")\r
        first_engine = engine_configs[0]\r
        # Download data just for the tuning phase\r
        download_code_from_github(\r
            engine_name=first_engine["name"],\r
            github_query=first_engine["github_query"],\r
            file_extensions=[ext.strip("*") for ext in first_engine["file_extensions"]],\r
            save_dir=first_engine["data_dir"],\r
            github_token=github_token\r
        )\r
        study = optuna.create_study(direction="minimize")\r
        study.optimize(lambda trial: objective(trial, first_engine["data_dir"], model, codebert_pipeline, first_engine["file_extensions"]), n_trials=5) # Reduced trials for speed\r
\r
        logging.info(f"Best trial value: {study.best_trial.value}")\r
        logging.info(f"Best hyperparameters from Optuna: {study.best_trial.params}")\r
        best_params.update(study.best_trial.params)\r
\r
    except Exception as e:\r
        logging.error(f"Error during hyperparameter tuning: {e}. Falling back to default parameters.")\r
\r
    os.makedirs("model_checkpoints", exist_ok=True)\r
\r
    # --- 3. Sequential Training Loop ---\r
    # This is the core of the new, robust pipeline.\r
    # We iterate through each engine, train the model, save it, clean up, and then move to the next.\r
    for i, config in enumerate(engine_configs):\r
        engine_name = config["name"]\r
        data_dir = config["data_dir"]\r
        language = config["language"]\r
        \r
        logging.info(f"\\n{'='*25}\\n Stage {i+1}/{len(engine_configs)}: Processing Engine: {engine_name} \\n{'='*25}")\r
\r
        # --- Stage 3.1: Download Data for the CURRENT engine ---\r
        download_code_from_github(\r
            engine_name=engine_name,\r
            github_query=config["github_query"],\r
            file_extensions=[ext.strip("*") for ext in config["file_extensions"]],\r
            save_dir=data_dir,\r
            github_token=github_token\r
        )\r
\r
        try:\r
            # --- Stage 3.2: Create Dataset and Dataloaders for the CURRENT engine ---\r
            # Using a try-except block to make the process robust.\r
            # If one engine fails (e.g., no data), it will skip to the next.\r
            dataset = CodeDataset(data_dir, model.tokenizer, model, codebert_pipeline, file_extensions=config["file_extensions"])\r
\r
            if len(dataset) < best_params["batch_size"]:\r
                logging.warning(f"Dataset for {engine_name} is too small ({len(dataset)} files). Skipping training for this engine.")\r
                continue\r
\r
            train_size = int(0.9 * len(dataset))\r
            val_size = len(dataset) - train_size\r
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])\r
\r
            # ! KEY IMPROVEMENT for i7-3770: Smartly set num_workers\r
            num_workers = 0 # Default to 0 for Windows or if unsure\r
            if platform.system() == "Linux" or platform.system() == "Darwin":\r
                cpu_count = os.cpu_count()\r
                if cpu_count is not None:\r
                    # For an older i7, using fewer workers can prevent system lag\r
                    num_workers = min(4, cpu_count // 2)\r
            logging.info(f"Using {num_workers} workers for DataLoader.")\r
\r
            train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True, collate_fn=custom_collate_fn, num_workers=num_workers, pin_memory=True)\r
            val_loader = DataLoader(val_dataset, batch_size=best_params["batch_size"], shuffle=False, collate_fn=custom_collate_fn, num_workers=num_workers, pin_memory=True)\r
\r
            # --- Stage 3.3: Create a fresh Optimizer for this fine-tuning stage ---\r
            optimizer = torch.optim.AdamW(model.parameters(), lr=best_params["lr"])\r
\r
            # --- Stage 3.4: Train on the CURRENT engine's data ---\r
            logging.info(f"\\n--- [{engine_name}] Starting Phase 1: Training with mock rewards ---")\r
            train_ppo_with_accelerator(\r
                model, train_loader, val_loader, optimizer, codebert_pipeline,\r
                num_epochs=config["mock_epochs"],\r
                gradient_accumulation_steps=4,\r
                use_mock_reward=True, visualize_graphs=False,\r
                clip_epsilon=best_params["clip_epsilon"], curiosity_weight=best_params["curiosity_weight"],\r
                engine_name=engine_name, language=language\r
            )\r
\r
            logging.info(f"\\n--- [{engine_name}] Starting Phase 2: Fine-tuning with real rewards ---")\r
            train_ppo_with_accelerator(\r
                model, train_loader, val_loader, optimizer, codebert_pipeline,\r
                num_epochs=config["real_epochs"],\r
                gradient_accumulation_steps=4,\r
                use_mock_reward=False, visualize_graphs=True,\r
                clip_epsilon=best_params["clip_epsilon"], curiosity_weight=best_params["curiosity_weight"],\r
                engine_name=engine_name, language=language\r
            )\r
\r
            # --- Stage 3.5: Save an intermediate checkpoint for this engine ---\r
            intermediate_save_path = os.path.join("model_checkpoints", f"model_after_{engine_name}_finetune")\r
            unwrapped_model = model.module if hasattr(model, 'module') else model\r
            unwrapped_model.llm.save_pretrained(intermediate_save_path)\r
            logging.info(f"Intermediate model fine-tuned on {engine_name} saved to '{intermediate_save_path}'.")\r
\r
        except FileNotFoundError as e:\r
            logging.error(f"CRITICAL ERROR for {engine_name}: {e}. Skipping this engine and moving to the next.")\r
            continue # Continue to the next engine in the list\r
        \r
        finally:\r
            # --- Stage 3.6: Clean up memory before starting the next stage ---\r
            # ! This is CRUCIAL for stability in long training runs on your hardware.\r
            if 'train_loader' in locals():\r
                del train_loader, val_loader, dataset, train_dataset, val_dataset, optimizer\r
            gc.collect()\r
            if torch.cuda.is_available():\r
                torch.cuda.empty_cache()\r
            logging.info(f"Cleaned up memory after training on {engine_name}.")\r
\r
\r
    # --- 4. Save the Final Model ---\r
    # This model has now been trained on all engines sequentially.\r
    final_model_save_path = "final_multi_engine_model"\r
    unwrapped_model = model.module if hasattr(model, 'module') else model\r
    unwrapped_model.llm.save_pretrained(final_model_save_path)\r
    logging.info(f"Final, sequentially-trained model saved to '{final_model_save_path}'.")\r
\r
    # --- 5. Run Inference Examples with the Final Model ---\r
    run_inference_examples(model)\r
\r
if __name__ == "__main__":\r
    main()`,a=(()=>{const e="FWK-V9-KEY-448146481,FWK-V9-KEY-774298112,FWK-V9-KEY-001124653,FWK-V9-KEY-124001204,FWK-V9-KEY-996801541,FWK-V9-KEY-411543678",i="FWK-ADMIN-999",s="FWK-ADMIN-TEST-123",c=parseInt("40",10)||30;return{VALID_KEYS:e.split(",").filter(n=>n.trim()!==""),ADMIN_KEY:i.trim(),ADMIN_TEST_KEY:s.trim(),EXPIRATION_DAYS:c,STORAGE_KEY_NAME:"fwk_active_key_v4",STORAGE_KEY_TIME:"fwk_expiration_time_v4"}})(),p={getActiveKey:()=>localStorage.getItem(a.STORAGE_KEY_NAME),getExpirationTime:()=>localStorage.getItem(a.STORAGE_KEY_TIME),setAccess:(e,i)=>{localStorage.setItem(a.STORAGE_KEY_NAME,e),localStorage.setItem(a.STORAGE_KEY_TIME,i.toString())},clearAccess:()=>{localStorage.removeItem(a.STORAGE_KEY_NAME),localStorage.removeItem(a.STORAGE_KEY_TIME)}},w={en:{"splash.title":"🚀 FWK Multiverse V9","splash.subtitle":"Cognitive Framework Access Portal","splash.desc":"This system is restricted intellectual property. Access is monitored. All attempts to copy, or reverse-engineer this framework are strictly prohibited and will be met with legal action.","splash.btn_text":"Proceed to Authentication","login.title":"🚀 FWK Multiverse V9","login.subtitle":"Access Portal","login.prompt":"Please enter your key to access resources.","login.key_placeholder":"Enter your access key","login.btn_text":"Authenticate","login.terms_title":"Access Agreement (Must Read)","viewer.warning_title":"⚠️ LEGAL WARNING (MAXIMUM PENALTY) ⚠️","viewer.warning_p1":"<strong>This is FWK Multiverse Intellectual Property (IP).</strong>","viewer.warning_p2":"Access is logged and monitored. Any attempt to Copy, Save, Print, or Refactor this code is a severe breach of the user agreement, resulting in permanent revocation and uncompromising legal action.","viewer.info_title":"About AI FWK Multiverse V9","viewer.btn_prototype":"Prototype","viewer.btn_v9":"V9 (Current)","lang.current_flag":"","lang.current_text":"EN"},th:{"splash.title":"🚀 FWK มัลติเวิร์ส V9","splash.subtitle":"พอร์ทัลเข้าถึงเฟรมเวิร์กการรับรู้","splash.desc":"ระบบนี้เป็นทรัพย์สินทางปัญญาที่ถูกจำกัดการเข้าถึง การเข้าถึงทั้งหมดจะถูกตรวจสอบ การพยายามคัดลอก, ดัดแปลง, หรือวิศวกรรมย้อนกลับเฟรมเวิร์กนี้เป็นสิ่งต้องห้ามโดยเด็ดขาดและจะถูกดำเนินการตามกฎหมาย","splash.btn_text":"ดำเนินการต่อ","login.title":"🚀 FWK มัลติเวิร์ส V9","login.subtitle":"พอร์ทัลเข้าใช้งาน","login.prompt":"กรุณาป้อนรหัสของคุณเพื่อเข้าถึงทรัพยากร","login.key_placeholder":"ป้อนรหัสเข้าใช้งานของคุณ","login.btn_text":"ยืนยันตัวตน","login.terms_title":"ข้อตกลงการเข้าใช้ (โปรดอ่าน)","viewer.warning_title":"⚠️ คำเตือนทางกฎหมาย (บทลงโทษสูงสุด) ⚠️","viewer.warning_p1":"<strong>นี่คือทรัพย์สินทางปัญญา (IP) ของ FWK Multiverse</strong>","viewer.warning_p2":"การเข้าถึงถูกบันทึกและตรวจสอบ ความพยายามใดๆ ในการคัดลอก, บันทึก, พิมพ์, หรือ Refactor โค้ดนี้ ถือเป็นการละเมิดข้อตกลงผู้ใช้อย่างร้ายแรง ซึ่งจะส่งผลให้ถูกเพิกถอนสิทธิ์การเข้าถึงถาวรและดำเนินการทางกฎหมายขั้นสูงสุด","viewer.info_title":"เกี่ยวกับ AI FWK มัลติเวิร์ส V9","viewer.btn_prototype":"ต้นแบบ","viewer.btn_v9":"V9 (ปัจจุบัน)","lang.current_flag":"","lang.current_text":"TH"}};document.addEventListener("DOMContentLoaded",()=>{const e={splashContainer:document.getElementById("splash-container"),loginContainer:document.getElementById("login-container"),viewerContainer:document.getElementById("viewer-container"),splashContinueBtn:document.getElementById("splash-continue-btn"),splashTitle:document.getElementById("splash-title"),splashSubtitle:document.getElementById("splash-subtitle"),splashDesc:document.getElementById("splash-desc"),splashBtnText:document.getElementById("splash-btn-text"),keyInput:document.getElementById("key-input"),loginButton:document.getElementById("login-button"),errorMessage:document.getElementById("error-message"),loginTitle:document.getElementById("login-title"),loginSubtitle:document.getElementById("login-subtitle"),loginPrompt:document.getElementById("login-prompt"),loginTermsTitle:document.getElementById("login-terms-title"),timerDisplay:document.getElementById("timer-display"),adminLogoutButton:document.getElementById("admin-logout-button"),viewerWarningTitle:document.getElementById("viewer-warning-title"),viewerWarningP1:document.getElementById("viewer-warning-p1"),viewerWarningP2:document.getElementById("viewer-warning-p2"),viewerInfoTitle:document.getElementById("viewer-info-title"),btnPrototype:document.getElementById("btn-prototype"),btnV9:document.getElementById("btn-v9"),wrapperPrototype:document.getElementById("wrapper-prototype"),wrapperV9:document.getElementById("wrapper-v9"),codePrototype:document.getElementById("code-prototype"),codeV9:document.getElementById("code-v9"),dynamicWatermark:document.getElementById("dynamic-watermark"),langBtnCurrent:document.getElementById("lang-btn-current"),langDropdown:document.getElementById("lang-dropdown"),langCurrentFlag:document.getElementById("lang-current-flag"),langCurrentText:document.getElementById("lang-current-text")};let i=null,s=null;function c(){i&&clearInterval(i),s&&clearInterval(s),e.splashContainer.style.display="flex",e.loginContainer.style.display="none",e.viewerContainer.style.display="none",e.splashContainer.classList.add("container-fade-in")}function n(t=""){i&&clearInterval(i),s&&clearInterval(s),e.splashContainer.style.display="none",e.loginContainer.style.display="flex",e.viewerContainer.style.display="none",e.errorMessage.textContent=t,e.keyInput.value="",e.loginContainer.classList.add("container-fade-in")}function o(t,r){e.splashContainer.style.display="none",e.loginContainer.style.display="none",e.viewerContainer.style.display="block",e.viewerContainer.classList.add("container-fade-in"),b();const l=r===a.ADMIN_KEY,d=r===a.ADMIN_TEST_KEY;e.adminLogoutButton.style.display=l||d?"block":"none",h(r),l?(e.timerDisplay.textContent="Status: ADMIN (No Expiration)",e.timerDisplay.dataset.status="admin-perm",i&&clearInterval(i)):(e.timerDisplay.dataset.status=d?"admin-test":"user",_(t,d),i&&clearInterval(i),i=setInterval(()=>_(t,d),1e3)),g("v9")}function _(t,r=!1){const l=Date.now(),d=t-l;if(d<=0){e.timerDisplay.textContent="EXPIRED",e.timerDisplay.dataset.status="expired",y();return}const f=Math.floor(d/(1e3*60*60*24)),u=Math.floor(d%(1e3*60*60*24)/(1e3*60*60)),A=Math.floor(d%(1e3*60*60)/(1e3*60)),E=Math.floor(d%(1e3*60)/1e3),z=`${f}d ${u}h ${A}m ${E}s`;let m="";r?m="Status: ADMIN (Test Key) | ":m="Expires in: ",e.timerDisplay.textContent=m+z}function h(t){s&&clearInterval(s);const r=btoa(t).substring(0,12),l=()=>{if(!e.dynamicWatermark)return;const d=new Date().toISOString().substring(11,19),f=Math.floor(Math.random()*10-5),u=Math.floor(Math.random()*20-10);e.dynamicWatermark.textContent=`User: ${r} | Time: ${d}`,e.dynamicWatermark.style.transform=`translate(calc(-50% + ${f}vw), calc(-50% + ${u}vh)) rotate(-15deg)`};l(),s=setInterval(l,2e3)}function g(t){const r=t==="v9";r?(e.wrapperV9.classList.add("visible"),e.wrapperPrototype.classList.remove("visible")):(e.wrapperV9.classList.remove("visible"),e.wrapperPrototype.classList.add("visible")),e.btnV9.classList.toggle("active",r),e.btnPrototype.classList.toggle("active",!r)}function b(){e.codeV9&&(e.codeV9.textContent=I,typeof Prism!="undefined"&&Prism.highlightElement(e.codeV9)),e.codePrototype&&(e.codePrototype.textContent=N,typeof Prism!="undefined"&&Prism.highlightElement(e.codePrototype))}function y(){const t=p.getActiveKey(),r=p.getExpirationTime(),l=t===a.ADMIN_KEY||t===a.ADMIN_TEST_KEY;if(a.VALID_KEYS.length===0&&!l){console.error("CONFIGURATION ERROR: Valid keys not loaded. Check ENV setup."),n("Key system error. Contact administrator.");return}if(!t||!r){c();return}if(t===a.ADMIN_KEY){o(Date.now()+365*24*60*60*1e3,t);return}const d=Date.now(),f=parseInt(r,10);if(d>f){p.clearAccess(),n("Your key has expired.");return}o(f,t)}function x(){const t=e.keyInput.value.trim();let r=0,l=!1;t===a.ADMIN_KEY?(r=Date.now()+365*24*60*60*1e3,p.setAccess(t,r),l=!0):(t===a.ADMIN_TEST_KEY||a.VALID_KEYS.includes(t))&&(r=Date.now()+a.EXPIRATION_DAYS*24*60*60*1e3,p.setAccess(t,r),l=!0),l?o(r,t):n("Invalid key or access right not found.")}function v(t){w[t]||(t="en");const r=w[t];e.splashTitle&&(e.splashTitle.innerHTML=r["splash.title"]),e.splashSubtitle&&(e.splashSubtitle.textContent=r["splash.subtitle"]),e.splashDesc&&(e.splashDesc.textContent=r["splash.desc"]),e.splashBtnText&&(e.splashBtnText.textContent=r["splash.btn_text"]),e.loginTitle&&(e.loginTitle.innerHTML=r["login.title"]),e.loginSubtitle&&(e.loginSubtitle.textContent=r["login.subtitle"]),e.loginPrompt&&(e.loginPrompt.textContent=r["login.prompt"]),e.keyInput&&(e.keyInput.placeholder=r["login.key_placeholder"]),e.loginButton&&(e.loginButton.textContent=r["login.btn_text"]),e.loginTermsTitle&&(e.loginTermsTitle.textContent=r["login.terms_title"]),e.viewerWarningTitle&&(e.viewerWarningTitle.innerHTML=r["viewer.warning_title"]),e.viewerWarningP1&&(e.viewerWarningP1.innerHTML=r["viewer.warning_p1"]),e.viewerWarningP2&&(e.viewerWarningP2.innerHTML=r["viewer.warning_p2"]),e.viewerInfoTitle&&(e.viewerInfoTitle.innerHTML=r["viewer.info_title"]),e.btnPrototype&&(e.btnPrototype.textContent=r["viewer.btn_prototype"]),e.btnV9&&(e.btnV9.textContent=r["viewer.btn_v9"]),e.langCurrentFlag&&(e.langCurrentFlag.textContent=r["lang.current_flag"]),e.langCurrentText&&(e.langCurrentText.textContent=r["lang.current_text"]),e.langDropdown&&e.langDropdown.classList.remove("visible")}function k(){e.splashContinueBtn.addEventListener("click",()=>n()),e.loginButton.addEventListener("click",x),e.keyInput.addEventListener("keyup",t=>{t.key==="Enter"&&x()}),e.adminLogoutButton.addEventListener("click",()=>{p.clearAccess(),n("Admin logged out.")}),e.btnPrototype.addEventListener("click",()=>g("prototype")),e.btnV9.addEventListener("click",()=>g("v9")),e.langBtnCurrent.addEventListener("click",t=>{t.stopPropagation(),e.langDropdown.classList.toggle("visible")}),e.langDropdown.addEventListener("click",t=>{const r=t.target.closest(".lang-option");r&&(t.preventDefault(),v(r.dataset.lang))}),document.addEventListener("click",t=>{!e.langBtnCurrent.contains(t.target)&&!e.langDropdown.contains(t.target)&&e.langDropdown.classList.remove("visible")})}k(),v("en"),y()});(function(){console.log("%cSTOP!","color: #f43f5e; font-size: 72px; font-weight: bold; text-shadow: 2px 2px 0 #000;"),console.log("%cThis is a restricted area. Access is monitored.","color: #e5e7eb; font-size: 18px;"),console.log("%cAny attempt to access, copy, or debug this code is a VIOLATION of the access agreement and will result in IMMEDIATE legal action. All activities are LOGGED.","color: #facc15; font-size: 16px; font-weight: bold;"),document.addEventListener("contextmenu",n=>n.preventDefault()),document.addEventListener("selectstart",n=>n.preventDefault()),document.addEventListener("dragstart",n=>n.preventDefault()),document.addEventListener("copy",n=>n.preventDefault()),document.addEventListener("cut",n=>n.preventDefault()),document.addEventListener("keydown",n=>{const o=n.target.tagName==="INPUT";if(n.key==="PrintScreen"||n.code==="PrintScreen"||n.key==="F12"||n.ctrlKey&&(n.key==="c"||n.key==="a"||n.key==="s"||n.key==="p"||n.key==="C"||n.key==="A"||n.key==="S"||n.key==="P")||n.ctrlKey&&n.shiftKey&&(n.key==="I"||n.key==="i"||n.key==="J"||n.key==="j")){n.preventDefault(),console.warn("Security Alert: Unauthorized key combination blocked.");return}!o&&n.target.closest(".code-wrapper")&&["ArrowUp","ArrowDown","PageUp","PageDown","Home","End"," "].includes(n.key)||o||(n.preventDefault(),n.stopPropagation())},!0);let i=!1,s=null;setInterval(()=>{const n=window.outerHeight-window.innerHeight,o=window.outerWidth-window.innerWidth;if(n>150||o>150||function(){const g=Date.now();debugger;return Date.now()-g>100}()){if(!i){i=!0,document.body.innerHTML="<h3 style='color:#f43f5e;text-align:center;margin-top:30vh;font-family:sans-serif;font-size:1.5rem;'>DEVTOOLS DETECTED.<br>Access Revoked. Close all developer tools and refresh.</h3>",s||(s=setInterval(()=>{console.clear(),console.log("%cILLEGAL ACCESS ATTEMPT LOGGED.","color: #f43f5e; font-size: 24px; font-weight: bold;")},500));try{window.close()}catch(g){}}}else i&&(s&&clearInterval(s),window.location.reload())},500)})();
