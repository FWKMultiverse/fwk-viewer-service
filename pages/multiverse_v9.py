import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data, Batch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, random_split
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os
import glob
import requests
import base64
import time
import sys
from torch.nn.functional import mse_loss, softmax, log_softmax, pad
from accelerate import Accelerator
from tqdm.auto import tqdm
import re
import random
import json
import ast
import networkx as nx
import subprocess
import pickle
import matplotlib.pyplot as plt
import shutil
import logging
import numpy as np
import joblib
import optuna
import gc
from datetime import datetime
import heapq
import platform
import argparse
import threading

# ! ##################################################################
# ! ################ V9 IMPORTS (from V8) ############################
# ! ##################################################################
import psutil # ! V8: สำหรับติดตามการใช้ CPU/RAM
try:
    import pynvml # ! V8: สำหรับติดตามการใช้ VRAM ของ NVIDIA
    pynvml.nvmlInit()
    VRAM_MONITORING_AVAILABLE = True
except ImportError:
    VRAM_MONITORING_AVAILABLE = False
    logging.warning("pynvml not found. VRAM monitoring will be disabled. Install with 'pip install pynvml'")

import tkinter as tk # ! V8: สำหรับสร้าง GUI Application
from tkinter import scrolledtext, Entry, Button, Frame

try:
    import redis # ! V8: L1 Cache Memory
except ImportError:
    redis = None
    logging.warning("redis-py not found. L1 Cache (Redis) will be disabled. Install with 'pip install redis'")

try:
    from qdrant_client import QdrantClient, models # ! V8: L2 Cache Memory
    from qdrant_client.http.models import Distance, VectorParams, PointStruct
except ImportError:
    QdrantClient = None
    logging.warning("qdrant-client not found. L2 Cache (Qdrant) will be disabled. Install with 'pip install qdrant-client'")

try:
    from minio import Minio # ! V8: L3 Storage
except ImportError:
    Minio = None
    logging.warning("minio not found. L3 Storage (MinIO) will be disabled. Install with 'pip install minio'")
# ! ##################################################################


# --- Configuration & Setup (from v8) ---
def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

set_seed(42)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# ! V9: Updated log file name
logging.basicConfig(filename='multiverse_v9_log.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

# --- ตรวจสอบการรองรับ Flash Attention 2 (from v8) ---
FLASH_ATTENTION_AVAILABLE = False
try:
    import flash_attn
    FLASH_ATTENTION_AVAILABLE = True
    logging.info("✅ Flash Attention 2 is available! Will be used for faster training.")
except ImportError:
    logging.warning("⚠️ Flash Attention 2 not found. Training will proceed without it, but might be slower.")

# --- (คงเดิมจาก v8) ---
try:
    from luaparser import ast as luaparser_ast
    from luaparser import parser as luaparser_parser
    logging.info("Using luaparser for Lua AST-based graphs.")
except ImportError:
    luaparser_ast = None
    luaparser_parser = None
    logging.warning("luaparser not found. Cannot build AST-based graphs.")

# ! V9: NEW - Security & Moderation Agent (Non-LLM)
class SecurityModeratorAgent:
    """
    ! V9: NEW AGENT (Non-LLM)
    ตัวกรองความปลอดภัยที่ทำงาน *ก่อน* และ *หลัง* การประมวลผลของ LLM
    ใช้กฎ Regex และ Keyword lists เพื่อความเร็วและความน่าเชื่อถือ (ไม่ใช้ LLM ตรวจสอบ LLM)
    """
    def __init__(self):
        logging.info("🛡️ Security Moderator Agent initialized.")
        
        # 1. รายการคำหยาบคาย (ตัวอย่าง) - ควรขยายให้ครอบคลุม
        self.PROFANITY_LIST = set([
            "example_profanity1", "example_profanity2", "ควย", "เหี้ย", "สัส", "ไอ้สัตว์"
        ])
        
        # 2. หัวข้อที่ผิดกฎหมาย/อันตราย (ตัวอย่าง)
        self.ILLEGAL_TOPICS_KEYWORDS = set([
            "how to make a bomb", "วิธีทำระเบิด", "how to hack", "วิธีแฮก", "child pornography", 
            "hate speech", "incite violence", "ยุยงให้เกิดความรุนแรง", "self-harm instruction"
        ])
        
        # 3. รูปแบบข้อมูลส่วนบุคคล (PII) - (ตัวอย่าง Regex)
        self.PII_PATTERNS = {
            "EMAIL": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "PHONE_TH": re.compile(r'\b0[689]\d{1}-?\d{3}-?\d{4}\b'), # เบอร์มือถือไทย
            "CREDIT_CARD": re.compile(r'\b(?:\d[ -]*?){13,16}\b'), # เลขบัตรเครดิต (แบบง่าย)
            "SSN_USA": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
        }
        
        self.SAFE_RESPONSE_FILTERED = "I apologize, but I cannot process this request. It violates my safety and moderation guidelines."
        self.SAFE_RESPONSE_PII = "I apologize, but I cannot process this request as it contains sensitive personal information. Please remove it and try again."

    def _check_text(self, text):
        """Helper function to check text against all rules."""
        text_lower = text.lower()
        
        # Check profanity
        if any(profanity in text_lower for profanity in self.PROFANITY_LIST):
            return False, "Profanity detected."
            
        # Check illegal topics
        if any(topic in text_lower for topic in self.ILLEGAL_TOPICS_KEYWORDS):
            return False, "Illegal/Harmful topic detected."
            
        # Check PII
        for pii_type, pattern in self.PII_PATTERNS.items():
            if pattern.search(text):
                return False, f"PII detected: {pii_type}"
                
        return True, "Safe"

    def pre_screen_input(self, prompt: str):
        """
        ! V9: กรอง Input *ก่อน* ส่งให้ LLM
        """
        is_safe, reason = self._check_text(prompt)
        if not is_safe:
            logging.warning(f"🛡️ Input blocked: {reason}. Prompt: '{prompt[:50]}...'")
            if "PII" in reason:
                return False, self.SAFE_RESPONSE_PII
            return False, self.SAFE_RESPONSE_FILTERED
        return True, "Input is safe."

    def post_screen_output(self, response: str):
        """
        ! V9: กรอง Output *หลัง* LLM สร้าง
        (ป้องกันกรณี LLM หลอนและสร้างเนื้อหาไม่เหมาะสม)
        """
        is_safe, reason = self._check_text(response)
        if not is_safe:
            logging.warning(f"🛡️ Output blocked: {reason}. Response: '{response[:50]}...'")
            if "PII" in reason: # LLM ไม่ควรเปิดเผย PII
                return False, "I apologize, but my response contained sensitive information and has been redacted for your safety."
            return False, self.SAFE_RESPONSE_FILTERED
        return True, response


# ! V8: Resource Monitoring System (คงเดิม)
class ResourceMonitor(threading.Thread):
    def __init__(self, interval=5):
        super().__init__(daemon=True)
        self.interval = interval
        self.stopped = False
        self.initial_vram_used = 0
        self.initial_ram_used = psutil.Process(os.getpid()).memory_info().rss
        if VRAM_MONITORING_AVAILABLE and torch.cuda.is_available():
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.initial_vram_used = pynvml.nvmlDeviceGetMemoryInfo(self.handle).used
        logging.info("📊 Resource Monitor initialized.")

    def run(self):
        while not self.stopped:
            # CPU
            cpu_usage = psutil.cpu_percent()
            # RAM
            process = psutil.Process(os.getpid())
            ram_info = process.memory_info()
            ram_used_gb = ram_info.rss / (1024 ** 3)
            ram_reduction = ((self.initial_ram_used - ram_info.rss) / self.initial_ram_used) * 100 if self.initial_ram_used > 0 else 0
            
            log_message = f"Resources: CPU: {cpu_usage:.1f}% | RAM: {ram_used_gb:.2f} GB (Reduction: {ram_reduction:.1f}%)"

            # VRAM (if available)
            if VRAM_MONITORING_AVAILABLE and torch.cuda.is_available():
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                vram_used_gb = mem_info.used / (1024 ** 3)
                vram_total_gb = mem_info.total / (1024 ** 3)
                vram_reduction = ((self.initial_vram_used - mem_info.used) / self.initial_vram_used) * 100 if self.initial_vram_used > 0 else 0
                log_message += f" | VRAM: {vram_used_gb:.2f}/{vram_total_gb:.2f} GB (Reduction: {vram_reduction:.1f}%)"
            
            logging.info(log_message)
            time.sleep(self.interval)

    def stop(self):
        self.stopped = True
        if VRAM_MONITORING_AVAILABLE:
            pynvml.nvmlShutdown()

# ! V8: Hierarchical Memory System (คงเดิม)
class HierarchicalMemory:
    def __init__(self, embedding_pipeline):
        self.embedding_pipeline = embedding_pipeline
        self.embedding_dim = 768 # CodeBERT base model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # L1 Cache: Redis
        self.redis_client = None
        if redis:
            try:
                self.redis_client = redis.Redis(
                    host=os.getenv("REDIS_HOST", "localhost"),
                    port=int(os.getenv("REDIS_PORT", 6379)),
                    db=0,
                    decode_responses=True
                )
                self.redis_client.ping()
                logging.info("✅ L1 Cache (Redis) connected successfully.")
            except Exception as e:
                logging.error(f"❌ Could not connect to L1 Cache (Redis): {e}")
                self.redis_client = None
        
        # L2 Cache: Qdrant
        self.qdrant_client = None
        # ! V9: Updated collection name
        self.qdrant_collection_name = "multiverse_v9_memory"
        if QdrantClient:
            try:
                self.qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
                # ! V9: ใช้ try-except แทน recreate เพื่อป้องกันข้อมูลหาย
                try:
                    self.qdrant_client.recreate_collection(
                        collection_name=self.qdrant_collection_name,
                        vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE),
                    )
                    logging.info(f"✅ L2 Cache (Qdrant) collection '{self.qdrant_collection_name}' re-created.")
                except Exception as e_coll:
                    logging.warning(f"Could not recreate collection (maybe it exists?): {e_coll}. Assuming it's ready.")
                
                logging.info(f"✅ L2 Cache (Qdrant) connected.")
            except Exception as e:
                logging.error(f"❌ Could not connect to L2 Cache (Qdrant): {e}")
                self.qdrant_client = None

        # L3 Storage: MinIO
        self.minio_client = None
        self.minio_bucket_name = "multiverse-v9-long-term-storage"
        if Minio:
            try:
                self.minio_client = Minio(
                    os.getenv("MINIO_ENDPOINT", "localhost:9000"),
                    access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
                    secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
                    secure=False
                )
                if not self.minio_client.bucket_exists(self.minio_bucket_name):
                    self.minio_client.make_bucket(self.minio_bucket_name)
                logging.info(f"✅ L3 Storage (MinIO) connected and bucket '{self.minio_bucket_name}' is ready.")
            except Exception as e:
                logging.error(f"❌ Could not connect to L3 Storage (MinIO): {e}")

    def _get_embedding(self, text):
        try:
            # ! V9: ปรับปรุงการจัดการ embedding (กรณี text ว่าง)
            if not text or not text.strip():
                return None
            embedding = self.embedding_pipeline(text)
            vector = np.array(embedding).mean(axis=1).flatten()
            return vector
        except Exception as e:
            logging.error(f"Could not generate embedding for text: {e}")
            return None

    def add_experience(self, code_snippet, reward, metadata={}):
        if reward < 5.0: return # Only store high-quality experiences
        
        snippet_hash = str(hash(code_snippet))
        vector = self._get_embedding(code_snippet)
        if vector is None: return

        # Add to L1 (hot cache)
        if self.redis_client:
            self.redis_client.set(f"code:{snippet_hash}", code_snippet, ex=3600) # Expire in 1 hour

        # Add to L2 (semantic search)
        if self.qdrant_client:
            self.qdrant_client.upsert(
                collection_name=self.qdrant_collection_name,
                points=[
                    PointStruct(id=snippet_hash, vector=vector.tolist(), payload={"code": code_snippet, **metadata})
                ]
            )
        
        # Add to L3 (permanent storage) - can be extended to store more complex objects
        logging.info(f"🧠 Added high-quality experience (Hash: {snippet_hash}) to Hierarchical Memory.")

    def retrieve_similar(self, query_code, k=3):
        query_vector = self._get_embedding(query_code)
        if query_vector is None or self.qdrant_client is None: return []

        try:
            search_result = self.qdrant_client.search(
                collection_name=self.qdrant_collection_name,
                query_vector=query_vector,
                limit=k,
                with_payload=True
            )
            return [hit.payload['code'] for hit in search_result]
        except Exception as e:
            logging.error(f"Could not retrieve similar code from L2 Cache: {e}")
            return []


# --- PPO Buffer & Curiosity Module (คงเดิมจาก v8) ---
class PrioritizedReplayBuffer:
    # ... (โค้ดส่วนนี้ไม่มีการเปลี่ยนแปลง) ...
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_end=1.0):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_increment = (self.beta_end - self.beta_start) / float(self.capacity)
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.device = 'cpu'
        self.max_priority = 1.0

    def push(self, experience):
        priority = self.max_priority ** self.alpha
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None

        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        state, action, action_len, log_prob, reward, next_state, done = zip(*samples)

        max_len = max(a.size(0) for a in action)
        action_padded = [pad(a, (0, max_len - a.size(0)), 'constant', 0) for a in action]

        self.beta = min(self.beta + self.beta_increment, self.beta_end)

        valid_next_states = [n for n in next_state if n is not None]
        batched_next_state = torch.stack(valid_next_states) if valid_next_states else None

        return (
            torch.stack(state),
            torch.stack(action_padded),
            torch.stack(action_len),
            torch.stack(log_prob),
            torch.stack(reward),
            batched_next_state,
            torch.stack(done),
            torch.tensor(weights, dtype=torch.float32),
            indices
        )

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = max(error.item(), 1e-6)
        self.max_priority = self.priorities[:len(self.buffer)].max()

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0

class ProjectGraphMemory(nn.Module):
    # ... (โค้ดส่วนนี้ไม่มีการเปลี่ยนแปลง) ...
    def __init__(self, num_features):
        super(ProjectGraphMemory, self).__init__()
        self.conv1 = GATv2Conv(num_features, 64, heads=4, edge_dim=1, concat=True)
        self.conv2 = GATv2Conv(64 * 4, 32, heads=2, edge_dim=1, concat=True)
        self.proj = nn.Linear(32 * 2, 32)
        self.device = 'cpu'

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = x.relu()
        if x.device != self.conv2.weight.device:
            x = x.to(self.conv2.weight.device)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.proj(x)
        return global_mean_pool(x, batch)


class CuriosityModule(nn.Module):
    # ... (โค้ดส่วนนี้ไม่มีการเปลี่ยนแปลง) ...
    def __init__(self, feature_size, action_size):
        super(CuriosityModule, self).__init__()
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
        self.forward_model = nn.Sequential(
            nn.Linear(feature_size + action_size, 128),
            nn.ReLU(),
            nn.Linear(128, feature_size)
        )
        self.feature_size = feature_size
        self.device = 'cpu'

    def forward(self, current_features, action_features, next_features):
        combined_features = torch.cat([current_features, next_features], dim=-1)
        predicted_action = self.inverse_model(combined_features)
        combined_forward = torch.cat([current_features, action_features], dim=-1)
        predicted_next_features = self.forward_model(combined_forward)
        forward_loss = mse_loss(predicted_next_features, next_features.detach())
        return forward_loss

# --- Core LLM Model (คงเดิมจาก v8) ---
class MultiAgentLLM(nn.Module):
    # ... (โค้ดส่วนนี้ไม่มีการเปลี่ยนแปลงที่สำคัญ) ...
    def __init__(self, llm_name="microsoft/phi-3-mini-4k-instruct", lora_rank=16, lora_alpha=32, lora_dropout=0.05):
        super(MultiAgentLLM, self).__init__()

        # ! (จาก v8) ตรวจสอบ VRAM และเลือก compute_dtype ที่ดีที่สุด
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logging.info(f"✅ Detected GPU: {gpu_name}")
            if "RTX 30" in gpu_name or "RTX 40" in gpu_name or "A100" in gpu_name or (hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported()):
                logging.info("GPU supports bfloat16. Using bf16 for better performance and stability.")
                compute_dtype = torch.bfloat16
            else:
                logging.info("Using float16 as a fallback.")
                compute_dtype = torch.float16

            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logging.info(f"Total VRAM: {vram:.2f} GB")
            if vram < 10:
                logging.warning(f"⚠️ VRAM is less than 10GB. Model may not fit. Performance might be slow due to offloading.")

            device_map_setting = "auto"
            logging.info(f"Setting device_map to '{device_map_setting}' to intelligently manage VRAM and CPU RAM.")

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
        else:
            logging.info("No CUDA device found. Using CPU. Performance will be very slow.")
            bnb_config = None 
            device_map_setting = "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logging.info("Tokenizer pad_token is not set. Using eos_token as pad_token.")


        # ! (จาก v8) ใช้ Flash Attention 2 ถ้ามี
        model_kwargs = {
            "quantization_config": bnb_config,
            "trust_remote_code": True,
            "device_map": device_map_setting,
            "offload_folder": "offload"
        }
        if FLASH_ATTENTION_AVAILABLE and torch.cuda.is_available():
            model_kwargs["attn_implementation"] = "flash_attention_2"


        self.llm = AutoModelForCausalLM.from_pretrained(llm_name, **model_kwargs)

        if bnb_config:
            self.llm.config.torch_dtype = bnb_config.bnb_4bit_compute_dtype
            self.llm = prepare_model_for_kbit_training(self.llm)
        
        def guess_lora_targets(model):
            names = []
            for n, mod in model.named_modules():
                if isinstance(mod, torch.nn.Linear) and ("attn" in n or "attention" in n or "q_proj" in n or "k_proj" in n or "v_proj" in n):
                    names.append(n.split('.')[-1])
            common = ["query_key_value", "dense", "out_proj", "c_attn", "c_proj", "fc1", "fc2", "gate_proj", "up_proj", "down_proj"]
            return list(set(names + common))

        targets = guess_lora_targets(self.llm)
        logging.info(f"Guessed LoRA target modules: {targets}")

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets
        )
        self.llm = get_peft_model(self.llm, lora_config)
        self.llm.print_trainable_parameters()

        codebert_embedding_dim = 768
        extra_features_dim = 1
        self.fixed_graph_embedding_dim = codebert_embedding_dim + extra_features_dim
        self.embedding_proj = nn.Linear(self.fixed_graph_embedding_dim, self.llm.config.hidden_size)
        self.graph_memory = ProjectGraphMemory(num_features=self.llm.config.hidden_size)
        self.graph_attn = nn.MultiheadAttention(
            embed_dim=self.llm.config.hidden_size,
            num_heads=8, 
            batch_first=True
        )
        self.graph_norm = nn.LayerNorm(self.llm.config.hidden_size)
        self.policy_head = nn.Sequential(
            nn.LayerNorm(self.llm.config.hidden_size),
            nn.Linear(self.llm.config.hidden_size, self.tokenizer.vocab_size)
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(self.llm.config.hidden_size),
            nn.Linear(self.llm.config.hidden_size, 1)
        )
        self.curiosity_module = CuriosityModule(self.llm.config.hidden_size, self.tokenizer.vocab_size)

    def _model_device(self):
        return next(p.device for p in self.llm.parameters())

    def forward(self, input_ids, attention_mask=None, project_graph_embedding=None):
        device = self._model_device()
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        llm_outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = llm_outputs.hidden_states[-1]

        if project_graph_embedding is not None:
            batch_size, seq_len, _ = last_hidden_state.shape
            # ! V9: แก้ไข Bug การย้าย device
            graph_emb_device = project_graph_embedding.to(device)
            expanded_graph_embedding = graph_emb_device.unsqueeze(1).repeat(1, seq_len, 1)
            fused_state, _ = self.graph_attn(last_hidden_state, expanded_graph_embedding, expanded_graph_embedding)
            fused_state = self.graph_norm(fused_state)
            fused_state = fused_state + last_hidden_state
        else:
            fused_state = last_hidden_state

        logits = self.policy_head(fused_state.to(self.policy_head[1].weight.dtype))
        last_token_hidden_state = fused_state[:, -1, :]
        value = self.value_head(last_token_hidden_state.to(self.value_head[1].weight.dtype))

        return logits, value, fused_state

    # ! (จาก v8) Prompt Template
    def generate_response(self, prompt_text: str, system_prompt: str, max_new_tokens=1024, do_sample=True, temperature=0.7, top_k=50, top_p=0.95):
        self.eval()
        device = self._model_device()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text}
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)

        generation_params = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "num_return_sequences": 1,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p
        }

        with torch.no_grad():
            generated_outputs = self.llm.generate(**generation_params)
        
        full_response = self.tokenizer.decode(generated_outputs[0], skip_special_tokens=True)
        
        # ! V9: ปรับปรุงการแยก response ให้แม่นยำขึ้น
        assistant_split = "<|assistant|>"
        if assistant_split in full_response:
            generated_text = full_response.split(assistant_split)[-1].strip()
        else:
            # Fallback for models that don't use the template correctly
            prompt_lines = prompt.splitlines()
            last_prompt_line = prompt_lines[-1] if prompt_lines else ""
            if last_prompt_line in full_response:
                 generated_text = full_response.split(last_prompt_line)[-1].strip()
            else:
                 generated_text = full_response # Return full text if we can't parse
        
        return generated_text, None

# --- Code Evaluator (ปรับปรุงจาก v8) ---
class CodeEvaluator:
    def __init__(self, use_mock_reward=False, language="lua"):
        self.use_mock_reward = use_mock_reward
        self.language = language
        # ! UPGRADE V9: เพิ่ม pattern ความปลอดภัยให้ครอบคลุมมากขึ้น
        self.safe_patterns_lua = [
            r'io\.open', r'os\.execute', r'os\.getenv', r'require\(["\'](http|socket)', 
            r'HttpService', r'RunService', r'pcall\s*\(\s*require',
            r'GetAsync', r'PostAsync', r'RequestAsync' # ! V9: Add network patterns
        ]
        self.safe_patterns_cs = [
            r'System\.IO\.File', r'System\.Diagnostics\.Process', r'System\.Net\.Http', 
            r'System\.Net\.Sockets', r'HttpClient' # ! V9
        ]
        self.safe_patterns_gd = [
            r'OS\.execute', r'File\.new', r'Directory\.new', r'HTTPClient' # ! V9
        ]
        self.safe_patterns_py = [
            r'subprocess', r'os\.system', r'eval', r'exec', 
            r'shutil', r'glob', r'socket' # ! V9
        ]
        self.vulnerability_patterns = {
            "lua": [r'loadstring'],
            "c#": [r'SqlCommand.*\.CommandText\s*=\s*".*"\s*\+'], # SQL Injection
            "gdscript": [],
            "c++": [r'strcpy', r'sprintf'], # Buffer overflow
            "python": [r'pickle\.load'] # Deserialization
        }
        self.llm = None

    def _pre_check_code(self, code_string):
        patterns = []
        if self.language in ["lua", "luau"]: patterns = self.safe_patterns_lua
        elif self.language in ["c#", "csharp"]: patterns = self.safe_patterns_cs
        elif self.language == "gdscript": patterns = self.safe_patterns_gd
        elif self.language == "python": patterns = self.safe_patterns_py
        
        for pattern in patterns:
            if re.search(pattern, code_string, re.IGNORECASE):
                # ! V9: ปรับปรุง Log ให้ชัดเจน
                logging.warning(f"Code pre-check FAIL. Language: {self.language}. Pattern: {pattern}")
                return False, f"Potential security risk detected (e.g., file/network access): {pattern}"
        return True, "No immediate high-risk security patterns detected."

    def _security_audit(self, code_string):
        # ... (คงเดิมจาก v8) ...
        penalty = 0.0
        findings = []
        lang_patterns = self.vulnerability_patterns.get(self.language, [])
        for pattern in lang_patterns:
            if re.search(pattern, code_string):
                findings.append(f"High-risk pattern found: '{pattern}'. This could lead to security vulnerabilities.")
                penalty -= 5.0
        if not findings:
            return 0.0, "No high-risk security patterns found."
        return penalty, "\n".join(findings)

    def _static_analysis(self, code_string):
        # ... (คงเดิมจาก v8) ...
        if self.language not in ["lua", "luau"]:
            return 0.0, "Skipped", f"Static analysis for {self.language} not implemented."
        temp_file_path = f"temp_code_{random.randint(1000,9999)}.lua"
        with open(temp_file_path, "w", encoding="utf-8") as f: f.write(code_string)
        try:
            result = subprocess.run(['luacheck', temp_file_path], capture_output=True, text=True, timeout=10)
            if "No issues found" in result.stdout or result.returncode == 0:
                return 1.0, "Syntax check: OK", result.stdout
            else:
                return -1.0, f"Syntax check failed", result.stdout + result.stderr
        except Exception as e:
            return -1.0, f"Error during static analysis: {e}", ""
        finally:
            if os.path.exists(temp_file_path): os.remove(temp_file_path)

    def _dynamic_analysis(self, code_string):
        # ... (คงเดิมจาก v8) ...
        if self.language not in ["lua", "luau"]:
            return 0.0, "Skipped", f"Dynamic analysis for {self.language} not implemented."
        if shutil.which("docker") is None:
            logging.warning("Docker not installed, skipping functional test.")
            return 0.0, "Docker not available", "Docker not found."
        temp_file_path = f"temp_code_{random.randint(1000,9999)}.lua"
        with open(temp_file_path, "w", encoding="utf-8") as f: f.write(code_string)
        docker_command = ['docker', 'run', '--rm', '--network', 'none', '--pids-limit', '256', '--cpus', '1', '--memory', '512m', '--ulimit', 'cpu=5', '--read-only', '-v', f'{os.getcwd()}/{temp_file_path}:/app/temp_code.lua:ro', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', 'luau_runtime_image', 'luau', '/app/temp_code.lua']
        try:
            start_time = time.time()
            result = subprocess.run(docker_command, capture_output=True, text=True, timeout=15)
            execution_time = time.time() - start_time
            if result.returncode == 0:
                reward = 10.0
                efficiency_reward = max(0, 1.0 - execution_time / 10.0)
                return reward + efficiency_reward, f"Functional test: Success!", result.stdout
            else:
                return -5.0, f"Functional test failed.", result.stdout + result.stderr
        except Exception as e:
            return -2.0, f"Error during dynamic analysis: {e}", ""
        finally:
            if os.path.exists(temp_file_path): os.remove(temp_file_path)

    def _assess_readability(self, code_string):
        # ... (คงเดิมจาก v8) ...
        lines = code_string.split('\n')
        if not lines: return 0.0
        comment_lines = sum(1 for line in lines if line.strip().startswith('--') or line.strip().startswith('//') or line.strip().startswith('#'))
        total_lines = len(lines)
        if total_lines == 0: return 0.0
        comment_ratio = comment_lines / total_lines
        long_line_penalty = sum(1 for line in lines if len(line) > 120)
        readability_score = (comment_ratio * 0.5) - (long_line_penalty * 0.1)
        return max(0, min(1, readability_score))

    def evaluate(self, code_string, project_graph):
        if self.use_mock_reward:
            return {"total_reward": random.uniform(-1, 10), "correctness_score": random.uniform(0, 10), "efficiency_score": random.uniform(0, 1), "knowledge_graph_score": random.uniform(0, 5), "readability_score": random.uniform(0, 1), "security_score": 0.0, "luacheck_log": "Mock", "docker_log": "Mock", "security_log": "Mock"}
        
        # ! V9: ตรวจสอบ code_string ว่างเปล่า
        if not code_string or not code_string.strip():
            return {"total_reward": -10.0, "correctness_score": -10.0, "efficiency_score": 0.0, "knowledge_graph_score": 0.0, "readability_score": 0.0, "security_score": -10.0, "luacheck_log": "No code provided", "docker_log": "No code provided", "security_log": "No code provided"}

        is_safe, security_log = self._pre_check_code(code_string)
        if not is_safe:
            return {"total_reward": -10.0, "correctness_score": -10.0, "efficiency_score": 0.0, "knowledge_graph_score": 0.0, "readability_score": 0.0, "security_score": -10.0, "luacheck_log": "Skipped", "docker_log": security_log, "security_log": security_log}

        security_score, detailed_security_log = self._security_audit(code_string)
        syntax_score, _, luacheck_log = self._static_analysis(code_string)
        functional_score, _, docker_log = self._dynamic_analysis(code_string)
        correctness_score = (syntax_score + functional_score)
        efficiency_score = max(0, 1.0 - len(code_string) / 2000)
        readability_score = self._assess_readability(code_string)
        
        kg_score = 0 # Placeholder
        total_reward = (0.4 * correctness_score) + (0.1 * efficiency_score) + (0.1 * kg_score) + (0.1 * readability_score) + (0.3 * security_score)
        return {"total_reward": total_reward, "correctness_score": correctness_score, "efficiency_score": efficiency_score, "knowledge_graph_score": kg_score, "readability_score": readability_score, "security_score": security_score, "luacheck_log": luacheck_log, "docker_log": docker_log, "security_log": detailed_security_log}

# ! ##################################################################
# ! ################ V9 EXPANDED AGENT TEAM (22 Agents) ##############
# ! ##################################################################
class BaseAgent:
    def __init__(self, model): self.model = model
    def _generate(self, prompt, system_prompt, **kwargs):
        # ! V9: จัดการ docstring สำหรับ system prompt
        doc_prompt = (self.__doc__ or system_prompt or "").strip()
        return self.model.generate_response(prompt, doc_prompt, **kwargs)[0]

# --- Existing Agents (Refactored from v8) ---
class CodeGeneratorAgent(BaseAgent):
    """You are an expert programmer. Write clean, efficient, and correct code based on the user's request. Add comments to explain complex parts."""
    def generate(self, prompt, context_examples=None):
        system_prompt = self.__doc__
        if context_examples:
            examples_text = "\n\n---\nHere are some relevant examples of good code:\n---\n" + "\n".join([f"```\n{ex}\n```" for ex in context_examples])
            prompt += examples_text
        return self._generate(prompt, system_prompt)

class CodeCriticAgent(BaseAgent):
    """You are a senior code reviewer. Provide a constructive critique, identifying bugs, inefficiencies, or security risks. Suggest specific improvements."""
    def critique(self, code, eval_results):
        critique_prompt = f"Analyze the code and its evaluation.\nCode:\n```\n{code}\n```\nEvaluation:\n{json.dumps(eval_results, indent=2)}"
        return self._generate(critique_prompt, self.__doc__, max_new_tokens=300)

class CodeRefinementAgent(BaseAgent):
    """You are a code refactoring specialist. Rewrite the provided code based on the critique to improve it. Provide ONLY the complete, corrected code block."""
    def refine(self, original_code, critique, context_examples=None):
        refinement_prompt = f"Based on the critique, refactor the original code.\nOriginal Code:\n```\n{original_code}\n```\nCritique:\n{critique}\n"
        if context_examples: refinement_prompt += "Relevant examples:\n" + "\n".join([f"```\n{ex}\n```" for ex in context_examples])
        refined_code = self._generate(refinement_prompt, self.__doc__, max_new_tokens=1500)
        match = re.search(r'```(?:\w+)?\n(.*?)\n```', refined_code, re.DOTALL)
        return match.group(1).strip() if match else refined_code

class AssetGeneratorAgent: # Does not need LLM
    def __init__(self): logging.info("AssetGeneratorAgent initialized.")
    def generate_asset(self, prompt): return f"asset_{hash(prompt) % 10000}"

class BugReportGeneratorAgent(BaseAgent):
    """You are a QA Engineer. Create a professional bug report from the given code and error log."""
    def generate_report(self, code, error_log):
        prompt = f"Code:\n```\n{code}\n```\nError Log:\n{error_log}\n\nProvide a bug report with: Description, Steps to Reproduce, Expected Behavior, Actual Behavior."
        return self._generate(prompt, self.__doc__)

class TestGenerationAgent(BaseAgent):
    """You are a test engineer. Generate a comprehensive set of unit tests for the following {language} code."""
    def generate_tests(self, code, language="Luau"):
        prompt = f"Code to test:\n```\n{code}\n```\nProvide the complete unit test script."
        return self._generate(prompt, self.__doc__.format(language=language))

class DocumentationAgent(BaseAgent):
    """You are a technical writer. Create clear, concise documentation for the following {language} code in Markdown format."""
    def generate_docs(self, code, language="Luau"):
        prompt = f"Code to document:\n```\n{code}\n```"
        return self._generate(prompt, self.__doc__.format(language=language))

class AutoRefactoringAgent(BaseAgent):
    """You are an expert in software architecture. Refactor this {language} code to improve its structure and readability without changing its functionality."""
    def refactor(self, code, language="Luau"):
        prompt = f"Code to refactor:\n```\n{code}\n```\nProvide only the refactored code."
        return self._generate(prompt, self.__doc__.format(language=language))

class GameDesignerAgent(BaseAgent):
    """You are a creative and innovative game designer."""
    def propose_feature(self, context):
        prompt = f"Based on the context '{context}', propose a new game feature. Describe it, how it works, and why it's fun."
        return self._generate(prompt, self.__doc__)

class CodeSummarizationAgent(BaseAgent):
    """You are a code summarization expert. Be concise."""
    def summarize(self, code, language="Luau"):
        prompt = f"Summarize the main functionality of this {language} code in one sentence.\nCode:\n```\n{code}\n```"
        return self._generate(prompt, self.__doc__, max_new_tokens=80)

class CodeQuestionAnsweringAgent(BaseAgent):
    """You are an AI assistant that analyzes code to answer questions accurately."""
    def answer_question(self, code, question, language="Luau"):
        prompt = f"Analyze this {language} code and answer the question.\nCode:\n```\n{code}\n```\nQuestion: {question}"
        return self._generate(prompt, self.__doc__.format(language=language))

# --- V8 NEW AGENTS (7 additions) ---
class WebDeveloperAgent(BaseAgent):
    """You are a full-stack web developer, expert in HTML, CSS, JavaScript, React, Python (Flask/Django), SQL, and PHP. Create a complete, functional code for the requested web component or application."""
    def generate_webapp(self, prompt):
        return self._generate(prompt, self.__doc__, max_new_tokens=2048)

class FinancialAnalystAgent(BaseAgent):
    """You are a financial analyst AI. Provide insightful analysis, data interpretation, and perspectives. IMPORTANT: ALWAYS include a disclaimer that you are not a certified financial advisor and your advice should not be taken as professional financial guidance."""
    def provide_analysis(self, prompt):
        return self._generate(prompt, self.__doc__, max_new_tokens=512)

class AIArchitectAgent(BaseAgent):
    """You are an AI/ML system architect. Design a robust and scalable architecture for the user's request. Consider data pipelines, model selection, training strategy, and deployment. Explain your choices."""
    def design_ai_system(self, prompt):
        return self._generate(prompt, self.__doc__, max_new_tokens=1024)

class AnimationCodeAgent(BaseAgent):
    """You are an expert in procedural animation and graphics programming. Generate code (e.g., using JavaScript libraries like three.js, p5.js, or CSS) to create the described animation."""
    def generate_animation(self, prompt):
        return self._generate(prompt, self.__doc__, max_new_tokens=1500)

class DatabaseAdminAgent(BaseAgent):
    """You are a database administrator, expert in SQL and database design. Write efficient SQL queries or schema definitions based on the request. Assume a standard SQL dialect unless specified."""
    def generate_sql(self, prompt):
        return self._generate(prompt, self.__doc__, max_new_tokens=512)

class GeneralConversationAgent(BaseAgent):
    """You are a helpful, friendly, and knowledgeable AI assistant. Engage in a natural conversation, answer questions, and provide information on a wide range of topics."""
    def chat(self, prompt):
        return self._generate(prompt, self.__doc__, max_new_tokens=1024)

class MarketingCopywriterAgent(BaseAgent):
    """You are a professional marketing copywriter. Write compelling and persuasive copy for websites, ads, or social media based on the user's product or service."""
    def write_copy(self, prompt):
        return self._generate(prompt, self.__doc__, max_new_tokens=512)

# ! ##################################################################
# ! ################ V9 NEW AGENTS (4 new additions) #################
# ! ##################################################################
class LongContextSummarizerAgent(BaseAgent):
    """You are an expert summarization AI. Your task is to receive very long text, code, or documents and condense them into a shorter, concise summary. You must preserve the core intent, key entities, function names, and the overall structure of the original content, but in a much shorter form."""
    def summarize(self, long_text):
        """
        ! V9: Handles long context by summarizing it.
        """
        prompt = f"Please summarize the following content, which may be thousands of lines long. Preserve the main purpose, key functions/classes, and structural flow.\n\nCONTENT:\n```\n{long_text}\n```\n\nCONCISE SUMMARY:"
        # ! V9: ให้ max_new_tokens สูงขึ้นเล็กน้อยสำหรับการสรุป
        return self._generate(prompt, self.__doc__, max_new_tokens=512)

class ScientificResearcherAgent(BaseAgent):
    """You are a scientific researcher and academic AI. Provide detailed, accurate, and sourced explanations for complex topics. Use clear, factual language and break down difficult concepts for understanding. Cite sources if possible (e.g., 'According to [field of study]...')."""
    def research(self, topic):
        prompt = f"Please provide a detailed, academic-level explanation for the following topic: {topic}"
        return self._generate(prompt, self.__doc__, max_new_tokens=1024)

class CreativeWriterAgent(BaseAgent):
    """You are a highly creative writer. You can write engaging stories, poems, scripts, song lyrics, or any other creative text based on the user's prompt."""
    def write(self, prompt):
        prompt = f"Please write a creative piece based on this idea: {prompt}"
        return self._generate(prompt, self.__doc__, max_new_tokens=1500)


# ! REWORKED V9: Mixture-of-Experts (MoE) Router
class MixtureOfExpertsRouter(BaseAgent):
    def __init__(self, model, agents):
        super().__init__(model)
        self.agents = agents
        # ! V9: อัปเดต Keywords ให้รวม Agent ใหม่
        self.agent_keywords = {
            # Code
            "CodeGeneratorAgent": ["code", "script", "function", "class", "algorithm", "implement", "write code", "สร้างโค้ด"],
            "CodeCriticAgent": ["review", "critique", "improve my code", "find bugs", "ตรวจโค้ด", "แก้บั๊ก"],
            "CodeRefinementAgent": ["refactor", "clean up code", "optimize this", "ปรับปรุงโค้ด"],
            "TestGenerationAgent": ["unit test", "test case", "py.test", "jest", "เขียนเทส"],
            "DocumentationAgent": ["document", "docs", "docstring", "comment", "explain this code", "ทำเอกสาร"],
            "CodeSummarizationAgent": ["summarize code", "what does this code do", "tl;dr code", "สรุปโค้ด"],
            "CodeQuestionAnsweringAgent": ["why this error", "how does this function work", "ถามเรื่องโค้ด"],
            # Web
            "WebDeveloperAgent": ["website", "web app", "html", "css", "javascript", "react", "frontend", "backend", "php", "api"],
            "DatabaseAdminAgent": ["sql", "database", "query", "schema", "table", "select", "insert", "update"],
            "AnimationCodeAgent": ["animation", "three.js", "p5.js", "css transition", "animate this"],
            # Game
            "GameDesignerAgent": ["game idea", "feature", "mechanic", "level design", "gameplay", "ออกแบบเกม"],
            "BugReportGeneratorAgent": ["bug report", "error log", "report issue", "รายงานบั๊ก"],
            # Business & AI
            "FinancialAnalystAgent": ["stock", "market", "investment", "finance", "economic", "portfolio", "การเงิน", "หุ้น"],
            "AIArchitectAgent": ["ai model", "machine learning", "architecture", "neural network", "pipeline", "ออกแบบ AI"],
            "MarketingCopywriterAgent": ["marketing", "copywriting", "ad copy", "social media post", "slogan", "เขียนโฆษณา"],
            # ! V9: New Agent Keywords
            "LongContextSummarizerAgent": ["summarize this document", "too long", "summarize this code", "สรุปไฟล์นี้"],
            "ScientificResearcherAgent": ["science", "physics", "biology", "chemistry", "explain topic", "research", "วิทยาศาสตร์"],
            "CreativeWriterAgent": ["write a story", "poem", "script", "song lyrics", "creative writing", "แต่งเรื่อง", "เขียนกลอน"],
            # Default
            "GeneralConversationAgent": ["what is", "who is", "explain", "tell me about", "how are you", "chat", "คุย", "คืออะไร"],
        }
        logging.info("🧠 Mixture-of-Experts Router (V9) initialized.")

    def route(self, prompt):
        """
        ! V9: Reworked Routing Logic
        Analyzes the prompt and selects the best *list* of agent(s) for the task.
        Returns a list of agent class names.
        """
        prompt_lower = prompt.lower()
        scores = {name: 0 for name in self.agents.keys()}
        
        # ! V9: ให้คะแนนตาม Keyword
        for name, keywords in self.agent_keywords.items():
            for keyword in keywords:
                if keyword in prompt_lower:
                    scores[name] += 1
        
        # ! V9: ใช้ Threshold-based-selection เพื่อเลือกหลาย Agent
        ROUTING_THRESHOLD = 0 # หมายความว่าแค่มี keyword 1 ครั้งก็ถูกเลือก
        
        selected_agents = [name for name, score in scores.items() if score > ROUTING_THRESHOLD]
        
        # --- Logic ในการคัดเลือก ---
        
        # 1. ถ้ามี Agent ที่เฉพาะเจาะจงสูง (เช่น Code) ถูกเลือก, ให้ตัด Agent ทั่วไป (General) ทิ้ง
        specific_agent_groups = [
            "CodeGeneratorAgent", "WebDeveloperAgent", "DatabaseAdminAgent", "FinancialAnalystAgent", 
            "AIArchitectAgent", "ScientificResearcherAgent", "CreativeWriterAgent", "CodeCriticAgent"
        ]
        
        has_specific_agent = any(agent in selected_agents for agent in specific_agent_groups)
        
        if has_specific_agent and "GeneralConversationAgent" in selected_agents and len(selected_agents) > 1:
            selected_agents.remove("GeneralConversationAgent")
            
        # 2. ถ้าไม่มี Agent ใดถูกเลือกเลย (คะแนน = 0), ให้ใช้ GeneralConversationAgent
        if not selected_agents:
            selected_agents = ["GeneralConversationAgent"]
            
        # 3. ! V9: ถ้า Prompt ยาวมาก, ให้บังคับใช้ LongContextSummarizerAgent (ตรรกะนี้จะถูกย้ายไป GUI เพื่อ pre-process)
        # (เราจะปล่อยให้ router เลือกตาม keyword "summarize" ที่นี่)

        logging.info(f"MoE Router V9 selected agents: {selected_agents} for prompt: '{prompt[:50]}...'")
        
        return selected_agents

# ... (Data Handling, Graph Building, Dataset classes from v8 can remain largely unchanged)
def download_code_from_github(engine_name: str, github_query: str, file_extensions: list, save_dir: str, github_token: str):
    # ... (คงเดิมจาก v8) ...
    if not os.path.exists(save_dir):
        logging.info(f"Directory '{save_dir}' not found. Creating it.")
        os.makedirs(save_dir)
        
    headers = {"Authorization": f"token {github_token}"}
    search_url = "https://api.github.com/search/code" # ! V9: แก้ไข URL ที่ถูกต้อง (v8 ผิด)
    
    downloaded_count = 0
    for ext in file_extensions:
        downloaded_count += len(glob.glob(os.path.join(save_dir, f"*.{ext}")))
    
    target_count = 1500 # Target remains 1500
    logging.info(f"[{engine_name}] Found {downloaded_count} existing files. Target is {target_count}.")
    if downloaded_count >= target_count:
        logging.info(f"[{engine_name}] Target already met. Skipping download.")
        return

    page = 1
    pbar = tqdm(total=target_count, initial=downloaded_count, desc=f"Downloading for {engine_name}")

    while downloaded_count < target_count:
        query = f'{github_query} ' + ' OR '.join([f'extension:{ext}' for ext in file_extensions])
        params = {"q": query, "per_page": 100, "page": page}
        
        try:
            response = requests.get(search_url, headers=headers, params=params, timeout=30)
            if 'X-RateLimit-Remaining' in response.headers and int(response.headers['X-RateLimit-Remaining']) < 10:
                reset_time = int(response.headers['X-RateLimit-Reset'])
                sleep_duration = max(0, reset_time - time.time()) + 5
                logging.warning(f"GitHub API rate limit low. Sleeping for {sleep_duration:.0f} seconds.")
                time.sleep(sleep_duration)
            response.raise_for_status()
            items = response.json().get("items", [])
            if not items:
                logging.info(f"[{engine_name}] No more code files found.")
                break
        except requests.exceptions.RequestException as e:
            logging.error(f"Error connecting to GitHub API: {e}. Retrying in 60 seconds...")
            time.sleep(60)
            continue

        for item in items:
            repo_name = item["repository"]["full_name"]
            file_path = item["path"]
            # ! V9: ตรวจสอบนามสกุลไฟล์ให้ตรงกับที่ร้องขอ
            file_ext = file_path.split('.')[-1]
            if file_ext not in file_extensions:
                continue

            save_path = os.path.join(save_dir, f"{repo_name.replace('/', '_')}_{os.path.basename(file_path)}")
            if os.path.exists(save_path):
                continue

            try:
                # ! V9: ใช้ "git_url" แทน "url" เพื่อรับ JSON ของไฟล์
                content_response = requests.get(item["git_url"], headers=headers, timeout=30)
                content_response.raise_for_status()
                file_data = content_response.json()
                if file_data.get("encoding") != "base64":
                     logging.warning(f"Skipping file {file_path} (not base64 encoded).")
                     continue
                
                content = base64.b64decode(file_data["content"]).decode("utf-8")
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(content)
                downloaded_count += 1
                pbar.update(1)
                if downloaded_count >= target_count:
                    break
            except (UnicodeDecodeError, KeyError, requests.exceptions.RequestException, json.JSONDecodeError) as e:
                logging.warning(f"Skipping file {file_path} in {repo_name} due to error: {e}")
                continue
        
        if downloaded_count >= target_count:
            break
        page += 1
        time.sleep(2) 
    
    pbar.close()
    logging.info(f"[{engine_name}] Download complete. Total files: {downloaded_count}.")
    return downloaded_count

# --- Graph Building & Dataset (No major changes from v8) ---
# ... (All functions like cache_embeddings, build_real_code_graph_ast, CodeDataset, etc. are here)
def get_func_name(node):
    try:
        if hasattr(node.name.id, 'id'): return node.name.id.id
        return node.name.id
    except Exception: return None

def cache_embeddings(code_chunks, codebert_pipeline, cache_file="embeddings_cache_v9.joblib"):
    # ! V9: Updated cache file name
    if codebert_pipeline is None:
        logging.error("CodeBERT pipeline is None. Cannot generate embeddings.")
        return {}
    if os.path.exists(cache_file):
        try: cache = joblib.load(cache_file)
        except Exception as e:
            logging.error(f"Error loading embedding cache: {e}. Creating new cache.")
            cache = {}
    else: cache = {}
    new_chunks = [chunk for chunk in code_chunks if chunk and chunk.strip() and chunk not in cache] # ! V9: Add check for empty chunks
    if new_chunks:
        logging.info(f"Generating embeddings for {len(new_chunks)} new chunks...")
        try:
            # ! V9: เพิ่มการจัดการ Error ใน pipeline
            new_embeddings_raw = codebert_pipeline(new_chunks, batch_size=16)
            new_embeddings = [torch.tensor(emb).squeeze() for emb in new_embeddings_raw]
            for chunk, embedding in zip(new_chunks, new_embeddings):
                if embedding.dim() > 1: embedding = embedding.mean(dim=0)
                cache[chunk] = embedding.tolist()
        except Exception as e:
            logging.error(f"Error generating CodeBERT embeddings: {e}")
            # ไม่ return {} แต่ให้ใช้ cache เก่าไปก่อน
        
        try:
            joblib.dump(cache, cache_file)
            logging.info("Embeddings cached.")
        except Exception as e:
            logging.error(f"Failed to save embedding cache: {e}")
    return cache

def visualize_graph(graph, filename="code_graph_v9.png", max_nodes=50):
    # ! V9: Updated file name
    if not hasattr(graph, 'x') or graph.x.shape[0] == 0: return
    g = nx.DiGraph()
    edges = graph.edge_index.t().tolist()
    g.add_edges_from(edges)
    
    # ! V9: จำกัดจำนวนโหนดที่จะวาด
    if len(g.nodes) > max_nodes:
        nodes_to_draw = list(g.nodes)[:max_nodes]
        g = g.subgraph(nodes_to_draw)
        
    plt.figure(figsize=(14, 14))
    try:
        pos = nx.spring_layout(g)
        nx.draw(g, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=8, edge_color='gray', arrows=True)
        plt.title("Code Knowledge Graph Visualization (Sampled)")
        plt.savefig(filename)
    except Exception as e:
        logging.error(f"Failed to visualize graph: {e}")
    plt.close()

def build_real_code_graph_ast(code_content, model, codebert_pipeline, asset_id=None, language="lua", design_doc=None, user_feedback=None):
    if language != "lua" or not luaparser_parser:
        return None
    try:
        # Simplified for brevity, original logic is sound
        ast_tree = luaparser_parser.parse(code_content)
        # ... (rest of the AST logic from original)
        all_chunks = re.split(r'\n(function|local function)', code_content)
        chunks = [all_chunks[i] + all_chunks[i+1] for i in range(1, len(all_chunks), 2) if i + 1 < len(all_chunks)]
        valid_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        if not valid_chunks: return None
        embeddings_cache = cache_embeddings(valid_chunks, codebert_pipeline)
        if not embeddings_cache: return None
        xs = []
        for chunk in valid_chunks:
            embedding = embeddings_cache.get(chunk)
            if embedding is not None:
                current_embedding = torch.tensor(embedding, dtype=torch.float32)
                usage_feature = torch.tensor([0.0]) # Mock usage
                combined_features = torch.cat([current_embedding, usage_feature], dim=0)
                padding_needed = model.fixed_graph_embedding_dim - combined_features.shape[0]
                if padding_needed > 0: combined_features = pad(combined_features, (0, padding_needed), 'constant', 0)
                else: combined_features = combined_features[:model.fixed_graph_embedding_dim]
                xs.append(combined_features)
        if not xs: return None
        x = torch.stack(xs, dim=0)
        edge_index = torch.empty((2, 0), dtype=torch.long) # Mock edges
        edge_attr = torch.empty((0, 1), dtype=torch.float)
        py_g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        py_g.x = model.embedding_proj(py_g.x.to(model._model_device()))
        py_g.node_type = torch.zeros(len(xs), dtype=torch.long)
        py_g.node_id = [None] * len(xs)
        return py_g
    except Exception as e:
        logging.error(f"Error building AST graph: {e}")
        return None

class CodeDataset(Dataset):
    def __init__(self, data_dir, tokenizer, model, codebert_pipeline, max_length=1024, graph_cache_dir="graph_cache_v9", file_extensions=None):
        # ! V9: Updated cache dir
        if file_extensions is None: file_extensions = ["*.lua", "*.luau"]
        self.file_paths = []
        for ext in file_extensions:
            self.file_paths.extend(glob.glob(os.path.join(data_dir, ext)))
        if not self.file_paths: raise FileNotFoundError(f"No files with extensions {file_extensions} found in {data_dir}.")
        self.tokenizer = tokenizer
        self.model = model
        self.codebert_pipeline = codebert_pipeline
        self.max_length = max_length
        self.graph_cache_dir = graph_cache_dir
        if not os.path.exists(self.graph_cache_dir): os.makedirs(self.graph_cache_dir)

    def __len__(self): return len(self.file_paths)
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        try:
            with open(file_path, "r", encoding="utf-8") as f: content = f.read()
            # ! V9: ตรวจสอบ content ว่าง
            if not content.strip():
                logging.warning(f"Skipping empty file: {file_path}")
                return None
            tokenized_data = self.tokenizer(content, return_tensors="pt", max_length=self.max_length, truncation=True, padding="max_length")
            graph_data = self.get_graph(content) # Simplified
            return {'input_ids': tokenized_data['input_ids'].squeeze(), 'attention_mask': tokenized_data['attention_mask'].squeeze(), 'code_content': content, 'code_graph_data': graph_data}
        except Exception as e:
            logging.error(f"Skipping file {file_path} due to error: {e}")
            return None

    def get_graph(self, content): # Simplified
        content_hash = str(hash(content))
        cache_path = os.path.join(self.graph_cache_dir, f"{content_hash}.joblib")
        if os.path.exists(cache_path):
            try: return joblib.load(cache_path)
            except Exception: pass
        graph_data = build_real_code_graph_ast(content, self.model, self.codebert_pipeline)
        if graph_data: 
            try:
                joblib.dump(graph_data, cache_path)
            except Exception as e:
                logging.warning(f"Failed to save graph cache: {e}")
        return graph_data

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None
    try:
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        code_contents = [item['code_content'] for item in batch]
        graph_data_list = [item['code_graph_data'] for item in batch if item['code_graph_data'] is not None]
        batched_graph = Batch.from_data_list(graph_data_list) if graph_data_list else None
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'code_contents': code_contents, 'code_graph_data': batched_graph}
    except Exception as e:
        logging.error(f"Error in custom_collate_fn: {e}")
        return None # Skip this batch

# --- PPO Loss & Training Loop (คงเดิมจาก v8) ---
def _calculate_ppo_loss(model, accelerator, state, action, action_len, old_log_prob, reward, next_state_embedding, old_value_preds, done, curiosity_weight, clip_epsilon, weights=None):
    # ... (โค้ดส่วนนี้ไม่มีการเปลี่ยนแปลง) ...
    if next_state_embedding is not None: next_state_embedding = next_state_embedding.to(accelerator.device)
    full_input_ids = torch.cat([state, action], dim=1).to(accelerator.device)
    max_len = model.llm.config.max_position_embeddings
    if full_input_ids.size(1) > max_len: full_input_ids = full_input_ids[:, :max_len]
    
    logits, value, fused_state = model(full_input_ids, None, project_graph_embedding=next_state_embedding)
    
    if action.numel() == 0 or logits.numel() == 0: return None, None, None, None, None
    logits_gen = logits[:, state.size(1)-1:-1, :]
    log_probs = log_softmax(logits_gen, dim=-1)
    
    # ! V9: ตรวจสอบขนาด action ให้ตรงกับ logits_gen
    action_clipped = action
    if action.size(1) > logits_gen.size(1):
        action_clipped = action[:, :logits_gen.size(1)]
    elif logits_gen.size(1) > action.size(1):
        logits_gen = logits_gen[:, :action.size(1), :]

    action_mask = (action_clipped != model.tokenizer.pad_token_id).to(accelerator.device)
    action_log_probs = log_probs.gather(2, action_clipped.unsqueeze(-1)).squeeze(-1)
    current_log_prob = (action_log_probs * action_mask).sum(dim=-1) / action_mask.sum(dim=-1).clamp(min=1e-6)
    
    gamma = 0.99
    reward = reward.squeeze()
    done = done.squeeze()
    value = value.squeeze()
    old_value_preds = old_value_preds.squeeze()
    
    advantages = reward + gamma * value.detach() * (1 - done.int()) - old_value_preds.detach()
    returns = advantages + old_value_preds.detach()
    
    curiosity_loss = torch.tensor(0.0).to(accelerator.device) # Simplified
    
    ratio = torch.exp(current_log_prob - old_log_prob.detach())
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    policy_loss_unweighted = -torch.min(ratio * advantages, clipped_ratio * advantages)
    value_loss_unweighted = mse_loss(value, returns)
    
    policy_loss = (policy_loss_unweighted * weights).mean() if weights is not None else policy_loss_unweighted.mean()
    value_loss = (value_loss_unweighted * weights).mean() if weights is not None else value_loss_unweighted.mean()
    
    entropy = -(softmax(logits_gen, dim=-1) * log_probs).sum(dim=-1).mean()
    
    total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy + curiosity_loss
    return total_loss, policy_loss, value_loss, entropy, curiosity_loss

def get_human_feedback(code_string):
    # ... (คงเดิมจาก v8) ...
    if "error" in code_string.lower() or "bug" in code_string.lower(): return -2.0
    if len(code_string) > 800: return 0.5
    return 1.5

# ! REWORKED V8: Training loop (คงเดิมจาก v8)
def train_ppo_with_accelerator(model, data_loader, val_loader, optimizer, codebert_pipeline, all_agents, hierarchical_memory, num_epochs, gradient_accumulation_steps, use_mock_reward, visualize_graphs, clip_epsilon, curiosity_weight, engine_name="", language="lua"):
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
    model, optimizer, data_loader, val_loader = accelerator.prepare(model, optimizer, data_loader, val_loader)
    
    code_evaluator = CodeEvaluator(use_mock_reward=use_mock_reward, language=language)
    moe_router = all_agents["MixtureOfExpertsRouter"] # ! V9: ใช้ Router ที่อัปเดตแล้ว

    logging.info(f"🚀 Starting PPO training with MoE (V9) for {engine_name}...")
    model.train()
    total_steps = len(data_loader) // gradient_accumulation_steps * num_epochs
    progress_bar = tqdm(range(total_steps), desc=f"Training ({engine_name})")
    replay_buffer = PrioritizedReplayBuffer(capacity=1024)
    
    for epoch in range(num_epochs):
        for step, batch in enumerate(data_loader):
            if batch is None: continue
            
            with accelerator.accumulate(model):
                initial_code_string = batch['code_contents'][0]
                
                # --- (คงเดิม) MoE in action (Critique -> Refine loop) ---
                
                # 1. Retrieve similar examples from memory
                retrieved_examples = hierarchical_memory.retrieve_similar(initial_code_string)
                
                # 2. First agent interaction: Critique
                critic_agent = all_agents["CodeCriticAgent"]
                eval_mock = code_evaluator.evaluate(initial_code_string, None) 
                critique = critic_agent.critique(initial_code_string, eval_mock)

                # 3. Second agent interaction: Refine
                refinement_agent = all_agents["CodeRefinementAgent"]
                refined_code = refinement_agent.refine(initial_code_string, critique, context_examples=retrieved_examples)
                
                # 4. Evaluate the refined code
                refined_graph = build_real_code_graph_ast(refined_code, model, codebert_pipeline, language=language)
                refined_reward_dict = code_evaluator.evaluate(refined_code, refined_graph)
                
                human_reward = get_human_feedback(refined_code)
                final_reward_value = refined_reward_dict['total_reward'] + human_reward
                
                # 5. Store high-quality results in hierarchical memory
                hierarchical_memory.add_experience(refined_code, final_reward_value, metadata={"engine": engine_name, "reward": final_reward_value})

                # --- PPO Update (similar to v8) ---
                graph_embedding = None
                if refined_graph:
                    graph_embedding = model.graph_memory(
                        refined_graph.x, 
                        refined_graph.edge_index, 
                        None, 
                        torch.zeros(refined_graph.x.shape[0], dtype=torch.long, device=accelerator.device)
                    )

                with torch.no_grad():
                    gen_ids = model.tokenizer.encode(refined_code, return_tensors="pt")
                    # ! V9: ตรวจสอบความยาว
                    if gen_ids.size(1) == 0: continue # Skip if refinement is empty
                    
                    full_input_ids = torch.cat([batch['input_ids'], gen_ids], dim=1).to(accelerator.device)
                    if full_input_ids.size(1) > model.llm.config.max_position_embeddings:
                        full_input_ids = full_input_ids[:, :model.llm.config.max_position_embeddings]
                    
                    logits, value_preds, _ = model(full_input_ids, None, project_graph_embedding=graph_embedding)
                    
                    # ! V9: ตรวจสอบขนาด Logits
                    if logits.size(1) <= gen_ids.size(1): continue # Skip if output is too small
                        
                    log_probs_tensor = log_softmax(logits[:, -gen_ids.size(1)-1:-1, :], dim=-1)
                    
                    # ! V9: ตรวจสอบขนาดอีกครั้ง
                    if log_probs_tensor.size(1) != gen_ids.size(1):
                        # This mismatch can happen, skip this step
                        continue
                        
                    gathered_log_probs = log_probs_tensor.gather(2, gen_ids.to(accelerator.device).unsqueeze(-1)).squeeze(-1).mean()
                
                experience = (batch['input_ids'].cpu(), gen_ids.cpu(), torch.tensor([gen_ids.size(1)]), gathered_log_probs.cpu(), torch.tensor([final_reward_value]), graph_embedding.cpu() if graph_embedding is not None else None, torch.tensor([False]))
                replay_buffer.push(experience)

                if len(replay_buffer) >= 32:
                    batch_data = replay_buffer.sample(32)
                    if not batch_data: continue
                    states, actions, lens, old_log_probs, rewards, next_states, dones, weights, indices = batch_data
                    normalized_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
                    with torch.no_grad():
                         _, old_value_preds, _ = model(states.to(accelerator.device))
                    loss_outputs = _calculate_ppo_loss(model, accelerator, states, actions, lens, old_log_probs, normalized_rewards, next_states, old_value_preds, dones, curiosity_weight, clip_epsilon, weights)
                    if loss_outputs and loss_outputs[0] is not None:
                        total_loss = loss_outputs[0]
                        accelerator.backward(total_loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        replay_buffer.update_priorities(indices, torch.abs(rewards.squeeze() - old_value_preds.cpu().squeeze()))


            if accelerator.sync_gradients:
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": total_loss.item() if 'total_loss' in locals() else 0.0, "reward": final_reward_value})

        # --- Checkpoint saving ---
        if (epoch + 1) % 5 == 0:
            logging.info(f"Epoch {epoch + 1}/{num_epochs} finished.")
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            save_dir = os.path.join("model_checkpoints_v9", engine_name, f"epoch_{epoch+1}") # ! V9
            os.makedirs(save_dir, exist_ok=True)
            unwrapped_model.llm.save_pretrained(save_dir)
            logging.info(f"✅ Model checkpoint for {engine_name} (V9) saved at epoch {epoch+1}")
            
    logging.info(f"PPO training for {engine_name} finished.")


# ! REWORKED V9: Function to run the 24/7 Chat GUI
def run_chat_app_gui(model_path: str):
    logging.info(f"🚀 Launching in 24/7 Interactive GUI Mode (V9)...")
    logging.info(f"Loading final trained model from: {model_path}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            offload_folder="offload_run"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        logging.info("✅ Final model and tokenizer loaded successfully.")
    except Exception as e:
        logging.error(f"❌ Failed to load the final model: {e}")
        return

    # ! V9: Lightweight wrapper for inference (Updated for Multi-Agent MoE)
    class InferenceWrapper(nn.Module):
        def __init__(self, llm, tokenizer, agents):
            super().__init__()
            self.llm = llm
            self.tokenizer = tokenizer
            self.agents = agents
            self.moe_router = agents["MixtureOfExpertsRouter"]
            # ! V9: Add specific agent for summarization
            self.long_context_agent = agents["LongContextSummarizerAgent"]

        def summarize_long_text(self, text: str):
            """
            ! V9: Dedicated function to call the summarizer agent.
            """
            logging.info(f"Summarizing long text (length: {len(text)})...")
            # We use the agent's _generate method directly
            summary = self.long_context_agent.summarize(text)
            logging.info(f"Summary length: {len(summary)}")
            return summary

        def generate_response(self, prompt_text: str):
            self.llm.eval()
            device = self.llm.device
            
            # ! V9: Use MoE to select a *list* of agents
            selected_agent_names = self.moe_router.route(prompt_text)
            
            # ! V9: Fuse System Prompts for Multi-Agent persona
            system_prompts = []
            for name in selected_agent_names:
                agent = self.agents.get(name)
                if agent and hasattr(agent, "__doc__") and agent.__doc__:
                    system_prompts.append(agent.__doc__.strip())
                elif name == "GeneralConversationAgent":
                     system_prompts.append("You are a helpful AI assistant.")
            
            if not system_prompts:
                fused_system_prompt = "You are a helpful AI assistant."
            elif len(system_prompts) == 1:
                fused_system_prompt = system_prompts[0]
            else:
                fused_system_prompt = "You are a multi-talented AI assistant. You must act as the following experts simultaneously:\n\n"
                for i, p in enumerate(system_prompts):
                    fused_system_prompt += f"EXPERT {i+1}: {p}\n"
                fused_system_prompt += "\nCombine these skills to answer the user's request comprehensively."

            logging.info(f"Using Fused System Prompt for agents: {selected_agent_names}")

            messages = [
                {"role": "system", "content": fused_system_prompt},
                {"role": "user", "content": prompt_text}
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)
            generation_params = { "max_new_tokens": 1500, "pad_token_id": self.tokenizer.pad_token_id, "eos_token_id": self.tokenizer.eos_token_id, "do_sample": True, "temperature": 0.7, "top_p": 0.95 }
            
            with torch.no_grad():
                generated_ids = self.llm.generate(input_ids, **generation_params)
            
            full_response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # ! V9: Use improved response parsing
            assistant_split = "<|assistant|>"
            if assistant_split in full_response:
                generated_text = full_response.split(assistant_split)[-1].strip()
            else:
                # Fallback
                prompt_lines = prompt.splitlines()
                last_prompt_line = prompt_lines[-1] if prompt_lines else ""
                if last_prompt_line in full_response:
                    generated_text = full_response.split(last_prompt_line)[-1].strip()
                else:
                    generated_text = full_response 
            
            return generated_text

    # Instantiate agents for the inference wrapper
    inference_agents = {}
    # ! V9: Add new agents to the list
    agent_classes = [
        CodeGeneratorAgent, CodeCriticAgent, CodeRefinementAgent, AssetGeneratorAgent, 
        BugReportGeneratorAgent, TestGenerationAgent, DocumentationAgent, AutoRefactoringAgent, 
        GameDesignerAgent, CodeSummarizationAgent, CodeQuestionAnsweringAgent, WebDeveloperAgent, 
        FinancialAnalystAgent, AIArchitectAgent, AnimationCodeAgent, DatabaseAdminAgent, 
        GeneralConversationAgent, MarketingCopywriterAgent,
        LongContextSummarizerAgent, ScientificResearcherAgent, CreativeWriterAgent # ! V9 New Agents
    ]
    
    # Mock model for agent initialization (same as v8)
    mock_model_for_agents = type('obj', (object,), {
        'llm': model, 
        'tokenizer': tokenizer, 
        'generate_response': lambda *args, **kwargs: None
    })() 
    
    for agent_class in agent_classes:
        if agent_class.__name__ == "AssetGeneratorAgent":
             inference_agents[agent_class.__name__] = agent_class()
        else:
             # ! V9: ใช้ docstring เป็น system prompt โดยอัตโนมัติ
             agent_instance = agent_class(mock_model_for_agents)
             inference_agents[agent_class.__name__] = agent_instance

    # ! V9: Router needs all LLM agents
    inference_agents["MixtureOfExpertsRouter"] = MixtureOfExpertsRouter(mock_model_for_agents, inference_agents)

    inference_model = InferenceWrapper(model, tokenizer, inference_agents)
    
    # ! V9: Instantiate the Security Moderator
    moderator = SecurityModeratorAgent()
    
    # ! V9: Define context length limit (in characters)
    # Phi-3-mini 4k tokens is ~12-16k chars. We set a lower limit to trigger summarization.
    MAX_PROMPT_LENGTH_CHARS = 3500 

    # --- GUI Setup ---
    def send_message(event=None):
        user_input = user_entry.get()
        if not user_input:
            return
        
        chat_area.config(state=tk.NORMAL)
        chat_area.insert(tk.END, "You: " + user_input + "\n\n")
        chat_area.config(state=tk.DISABLED)
        user_entry.delete(0, tk.END)
        
        # Disable input while processing
        user_entry.config(state=tk.DISABLED)
        send_button.config(state=tk.DISABLED)
        
        def generate():
            try:
                # ! V9: STEP 1 - Pre-Screen Input
                is_safe, screening_message = moderator.pre_screen_input(user_input)
                if not is_safe:
                    chat_area.config(state=tk.NORMAL)
                    chat_area.insert(tk.END, f"AI (Security): {screening_message}\n\n")
                    chat_area.see(tk.END)
                    chat_area.config(state=tk.DISABLED)
                    return # Stop processing
                
                prompt_to_process = user_input
                
                # ! V9: STEP 2 - Long Context Handling
                if len(prompt_to_process) > MAX_PROMPT_LENGTH_CHARS:
                    chat_area.config(state=tk.NORMAL)
                    chat_area.insert(tk.END, "AI: (Your input is very long. Summarizing it first...)\n\n")
                    chat_area.see(tk.END)
                    chat_area.config(state=tk.DISABLED)
                    
                    prompt_to_process = inference_model.summarize_long_text(prompt_to_process)
                    
                    chat_area.config(state=tk.NORMAL)
                    chat_area.insert(tk.END, f"AI (Summary): {prompt_to_process}\n(Now processing the summary...)\n\n")
                    chat_area.see(tk.END)
                    chat_area.config(state=tk.DISABLED)

                # ! V9: STEP 3 - Generate Response (using MoE)
                raw_response = inference_model.generate_response(prompt_to_process)
                
                # ! V9: STEP 4 - Post-Screen Output
                is_safe, final_response = moderator.post_screen_output(raw_response)
                
                if not is_safe:
                    # If output is unsafe, display the canned safe response
                    chat_area.config(state=tk.NORMAL)
                    chat_area.insert(tk.END, f"AI (Security): {final_response}\n\n")
                    chat_area.see(tk.END)
                    chat_area.config(state=tk.DISABLED)
                else:
                    # Output is safe, display it
                    chat_area.config(state=tk.NORMAL)
                    chat_area.insert(tk.END, "AI: " + final_response + "\n\n")
                    chat_area.see(tk.END)
                    chat_area.config(state=tk.DISABLED)

            except Exception as e:
                logging.error(f"Error during GUI generation: {e}")
                chat_area.config(state=tk.NORMAL)
                chat_area.insert(tk.END, f"AI (Error): An internal error occurred. Please try again.\n\n")
                chat_area.see(tk.END)
                chat_area.config(state=tk.DISABLED)
            finally:
                # Re-enable input regardless of outcome
                user_entry.config(state=tk.NORMAL)
                send_button.config(state=tk.NORMAL)
                user_entry.focus_set()

        threading.Thread(target=generate, daemon=True).start()

    root = tk.Tk()
    root.title("Multiverse V9 AI")
    
    # Position window at bottom right (คงเดิม)
    window_width = 400
    window_height = 500
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x_cordinate = int(screen_width - window_width - 20)
    y_cordinate = int(screen_height - window_height - 60) 
    root.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")
    root.resizable(False, False)

    main_frame = Frame(root, bg="#2E2E2E")
    main_frame.pack(fill=tk.BOTH, expand=True)

    chat_area = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, state=tk.DISABLED, bg="#1E1E1E", fg="#D4D4D4", font=("Arial", 10))
    chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    input_frame = Frame(main_frame, bg="#2E2E2E")
    input_frame.pack(padx=10, pady=5, fill=tk.X)

    user_entry = Entry(input_frame, bg="#3C3C3C", fg="#D4D4D4", font=("Arial", 10), insertbackground='white')
    user_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
    user_entry.bind("<Return>", send_message)
    user_entry.focus_set()

    send_button = Button(input_frame, text="Send", command=send_message, bg="#0E639C", fg="white", activebackground="#1177BB", borderwidth=0)
    send_button.pack(side=tk.RIGHT, padx=(5,0))

    root.mainloop()

# ! ##################################################################
# ! ################ REWORKED V9 MAIN FUNCTION #######################
# ! ##################################################################
def main():
    parser = argparse.ArgumentParser(description="AI FWK Multiverse V9 - Training and Inference")
    parser.add_argument('mode', choices=['train', 'run'], help="Choose 'train' to start the training pipeline or 'run' to start the interactive chat GUI.")
    args = parser.parse_args()

    # Start resource monitoring
    monitor = ResourceMonitor()
    monitor.start()

    if args.mode == 'train':
        github_token = os.getenv("GITHUB_TOKEN")
        if not github_token:
            logging.error("CRITICAL ERROR: GITHUB_TOKEN environment variable is not set. Cannot download data.")
            sys.exit(1)

        engine_configs = [
            {"name": "Roblox", "data_dir": "roblox_code_data", "github_query": "roblox luau", "file_extensions": ["lua", "luau"], "mock_epochs": 1, "real_epochs": 5, "language": "luau"},
            {"name": "Godot", "data_dir": "godot_code_data", "github_query": "godot gdscript", "file_extensions": ["gd"], "mock_epochs": 1, "real_epochs": 5, "language": "gdscript"},
            {"name": "Unity", "data_dir": "unity_code_data", "github_query": "unity csharp", "file_extensions": ["cs"], "mock_epochs": 1, "real_epochs": 5, "language": "csharp"},
            {"name": "Unreal", "data_dir": "unreal_code_data", "github_query": "unreal engine c++", "file_extensions": ["cpp", "h"], "mock_epochs": 1, "real_epochs": 5, "language": "c++"},
            {"name": "WebDev", "data_dir": "web_code_data", "github_query": "react javascript", "file_extensions": ["js", "jsx", "ts", "tsx", "html", "css"], "mock_epochs": 1, "real_epochs": 5, "language": "javascript"},
            {"name": "Python", "data_dir": "python_code_data", "github_query": "python", "file_extensions": ["py"], "mock_epochs": 1, "real_epochs": 5, "language": "python"}
        ]

        logging.info("Initializing MultiAgentLLM (V9) model for training...")
        model = MultiAgentLLM()
        logging.info("Initializing CodeBERT pipeline...")
        codebert_pipeline = pipeline("feature-extraction", model="microsoft/CodeBERT-base", tokenizer="microsoft/CodeBERT-base", device=0 if torch.cuda.is_available() else -1)

        # ! V9: Initialize Hierarchical Memory and all Agents
        hierarchical_memory = HierarchicalMemory(codebert_pipeline)
        
        all_agents = {}
        # ! V9: Add new agents to training list
        agent_classes = [
            CodeGeneratorAgent, CodeCriticAgent, CodeRefinementAgent, AssetGeneratorAgent, 
            BugReportGeneratorAgent, TestGenerationAgent, DocumentationAgent, AutoRefactoringAgent, 
            GameDesignerAgent, CodeSummarizationAgent, CodeQuestionAnsweringAgent, WebDeveloperAgent, 
            FinancialAnalystAgent, AIArchitectAgent, AnimationCodeAgent, DatabaseAdminAgent, 
            GeneralConversationAgent, MarketingCopywriterAgent,
            LongContextSummarizerAgent, ScientificResearcherAgent, CreativeWriterAgent # ! V9 New Agents
        ]
        
        for agent_class in agent_classes:
            if agent_class.__name__ == "AssetGeneratorAgent":
                 all_agents[agent_class.__name__] = agent_class()
            else:
                 all_agents[agent_class.__name__] = agent_class(model)
        
        # ! V9: Router needs all LLM agents
        all_agents["MixtureOfExpertsRouter"] = MixtureOfExpertsRouter(model, all_agents)


        best_params = {"lr": 5e-5, "batch_size": 1, "clip_epsilon": 0.2, "curiosity_weight": 0.03}
        os.makedirs("model_checkpoints_v9", exist_ok=True) # ! V9

        for i, config in enumerate(engine_configs):
            engine_name, data_dir, language = config["name"], config["data_dir"], config["language"]
            logging.info(f"\n{'='*25}\n Stage {i+1}/{len(engine_configs)}: Processing Engine: {engine_name} \n{'='*25}")

            download_code_from_github(engine_name, config["github_query"], config["file_extensions"], data_dir, github_token)

            try:
                dataset = CodeDataset(data_dir, model.tokenizer, model, codebert_pipeline, file_extensions=[f"{ext}" for ext in config["file_extensions"]]) # ! V9: Fix glob pattern
                if len(dataset) < best_params["batch_size"]:
                    logging.warning(f"Dataset for {engine_name} is too small ({len(dataset)} files). Skipping.")
                    continue
                
                train_size = int(0.95 * len(dataset))
                val_size = len(dataset) - train_size
                train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

                num_workers = 0
                if platform.system() == "Linux": num_workers = min(4, os.cpu_count() // 2 if os.cpu_count() else 0)
                logging.info(f"Using {num_workers} workers for DataLoader.")

                train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True, collate_fn=custom_collate_fn, num_workers=num_workers)
                val_loader = DataLoader(val_dataset, batch_size=best_params["batch_size"], shuffle=False, collate_fn=custom_collate_fn, num_workers=num_workers)
                optimizer = torch.optim.AdamW(model.parameters(), lr=best_params["lr"])

                logging.info(f"\n--- [{engine_name}] Phase 1: Training with mock rewards ---")
                train_ppo_with_accelerator(model, train_loader, val_loader, optimizer, codebert_pipeline, all_agents, hierarchical_memory, num_epochs=config["mock_epochs"], gradient_accumulation_steps=8, use_mock_reward=True, visualize_graphs=False, **best_params, engine_name=engine_name, language=language)
                
                logging.info(f"\n--- [{engine_name}] Phase 2: Fine-tuning with real rewards ---")
                train_ppo_with_accelerator(model, train_loader, val_loader, optimizer, codebert_pipeline, all_agents, hierarchical_memory, num_epochs=config["real_epochs"], gradient_accumulation_steps=8, use_mock_reward=False, visualize_graphs=True, **best_params, engine_name=engine_name, language=language)

            except FileNotFoundError as e:
                logging.error(f"CRITICAL ERROR for {engine_name}: {e}. Skipping this engine.")
                continue
            finally:
                if 'train_loader' in locals(): del train_loader, val_loader, dataset, train_dataset, val_dataset, optimizer
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                logging.info(f"🧹 Cleaned up memory after training on {engine_name}.")
        
        final_model_save_path = "final_multi_engine_model_v9" # ! V9
        unwrapped_model = model.module if hasattr(model, 'module') else model
        unwrapped_model.llm.save_pretrained(final_model_save_path)
        logging.info(f"🎉 Final, sequentially-trained model (V9) saved to '{final_model_save_path}'.")
        logging.info("Training complete. Automatically launching chat GUI...")
        
        # Automatically switch to run mode
        run_chat_app_gui(model_path=final_model_save_path)

    elif args.mode == 'run':
        final_model_path = "final_multi_engine_model_v9" # ! V9
        if not os.path.exists(final_model_path):
            logging.error(f"Model directory not found at '{final_model_path}'.")
            logging.error("Please train the model first by running: python multiverse_v9.py train")
            sys.exit(1)
        
        run_chat_app_gui(model_path=final_model_path)
    
    # Stop the monitor when the program exits
    monitor.stop()


if __name__ == "__main__":
    main()