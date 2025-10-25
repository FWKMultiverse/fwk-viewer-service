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
import gc # ! NEW: Import garbage collector for memory management
from datetime import datetime
import heapq
import platform

# --- Configuration & Setup (from original) ---
def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

set_seed(42)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

logging.basicConfig(filename='training_errors.log', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

try:
    from luaparser import ast as luaparser_ast
    from luaparser import parser as luaparser_parser
    logging.info("Using luaparser for Lua AST-based graphs.")
except ImportError:
    luaparser_ast = None
    luaparser_parser = None
    logging.warning("luaparser not found. Cannot build AST-based graphs.")

try:
    from pygments.lexers.lua import LuaLexer
    from pygments import lex
    from pygments.token import Token
except ImportError:
    logging.warning("Pygments with LuaLexer not found. Falling back to regex.")
    LuaLexer = None

# --- NEW: Prioritized Experience Replay (PER) Buffer ---
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_end=1.0):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start # Store start value for correct increment calculation
        self.beta_end = beta_end
        # FIX A: Correct beta annealing increment calculation
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

        # FIX A: Correctly update beta value
        self.beta = min(self.beta + self.beta_increment, self.beta_end)

        # FIX B: Handle next_state which is a list of tensors or None
        valid_next_states = [n for n in next_state if n is not None]
        batched_next_state = torch.stack(valid_next_states) if valid_next_states else None

        return (
            torch.stack(state),
            torch.stack(action_padded),
            torch.stack(action_len),
            torch.stack(log_prob),
            torch.stack(reward),
            batched_next_state, # Return the correctly batched tensor
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

# --- 1. Enhanced MARL Architecture with Knowledge Graph Diffusion ---

# NEW: Long-Term Memory using Vectorized Storage (RAG simulation)
class VectorizedMemory:
    def __init__(self, embedding_pipeline):
        self.memory = {}
        self.vectors = []
        self.code_snippets = []
        self.embedding_pipeline = embedding_pipeline
        logging.info("VectorizedMemory (Long-Term Memory) initialized.")

    def add_experience(self, code_snippet, reward):
        # Only store high-quality experiences
        if reward < 5.0:
            return

        # Avoid duplicates
        if code_snippet in self.memory:
            return

        try:
            # Generate embedding for the good code snippet
            embedding = self.embedding_pipeline(code_snippet)
            vector = np.array(embedding).mean(axis=1).flatten()

            self.memory[code_snippet] = vector
            self.vectors.append(vector)
            self.code_snippets.append(code_snippet)
            logging.info(f"Added a high-quality code snippet to Long-Term Memory.")
        except Exception as e:
            logging.error(f"Could not process and add experience to VectorizedMemory: {e}")


    def retrieve_similar(self, query_code, k=2):
        if not self.vectors:
            return []

        try:
            query_embedding = self.embedding_pipeline(query_code)
            query_vector = np.array(query_embedding).mean(axis=1).flatten()

            # Calculate cosine similarity
            vectors_matrix = np.array(self.vectors)
            dot_product = np.dot(vectors_matrix, query_vector)
            norm_query = np.linalg.norm(query_vector)
            norm_vectors = np.linalg.norm(vectors_matrix, axis=1)
            similarities = dot_product / (norm_vectors * norm_query)

            # Get top-k most similar snippets
            top_k_indices = np.argsort(similarities)[::-1][:k]
            return [self.code_snippets[i] for i in top_k_indices]
        except Exception as e:
            logging.error(f"Could not retrieve similar code from VectorizedMemory: {e}")
            return []

class ProjectGraphMemory(nn.Module):
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

def guess_lora_targets(model):
    names = []
    for n, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and ("attn" in n or "attention" in n or "q_proj" in n or "k_proj" in n or "v_proj" in n):
            names.append(n.split('.')[-1])
    common = ["query_key_value", "dense", "out_proj", "c_attn", "c_proj"]
    return list(set(names + common))

class MultiAgentLLM(nn.Module):
    def __init__(self, llm_name="Salesforce/codegen-2B-mono", lora_rank=8, lora_alpha=16, lora_dropout=0.05):
        super(MultiAgentLLM, self).__init__()

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logging.info(f"Detected GPU: {gpu_name}")
            if "RTX 30" in gpu_name or "RTX 40" in gpu_name or hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
                logging.info("Using bfloat16 for better performance on modern GPUs.")
                compute_dtype = torch.bfloat16
            else:
                logging.info("Using float16 as a fallback for older GPUs.")
                compute_dtype = torch.float16

            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logging.info(f"Total VRAM: {vram:.2f} GB")

            # ! KEY IMPROVEMENT for 8GB VRAM: Force "auto" to enable offloading
            # This will use CPU RAM when VRAM is full, which is crucial for your setup.
            device_map_setting = "auto"
            logging.info(f"Setting device_map to '{device_map_setting}' to manage limited VRAM.")


            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
        else:
            logging.info("No CUDA device found. Using CPU.")
            if os.cpu_count() < 3:
                logging.warning("Less than 3 CPU cores detected. Performance may be very slow.")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float32,
                bnb_4bit_use_double_quant=True,
            )
            device_map_setting = "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map=device_map_setting,
            offload_folder="offload" # ! Specify folder for offloaded layers
        )

        self.llm.config.torch_dtype = bnb_config.bnb_4bit_compute_dtype
        self.llm = prepare_model_for_kbit_training(self.llm)

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

        # FIX D: Use a fixed dimension for graph node embeddings to avoid dynamic resizing
        codebert_embedding_dim = 768
        extra_features_dim = 1  # For usage data, etc.
        self.fixed_graph_embedding_dim = codebert_embedding_dim + extra_features_dim

        self.embedding_proj = nn.Linear(self.fixed_graph_embedding_dim, self.llm.config.hidden_size)

        self.graph_memory = ProjectGraphMemory(num_features=self.llm.config.hidden_size)

        self.graph_attn = nn.MultiheadAttention(
            embed_dim=self.llm.config.hidden_size,
            num_heads=4,
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

    def _add_emojis(self, text):
        emojis = {
            "hello": "👋", "hi": "👋", "hey": "👋", "thank you": "🙏", "thanks": "🙏",
            "great": "👍", "good": "👍", "ok": "👍", "sorry": "😔", "apologize": "😔",
            "happy": "😊", "exciting": "✨", "code": "💻", "error": "❌", "bug": "🐛",
            "success": "✅", "completed": "✅", "question": "🤔", "help": "💡",
            "problem": "🤯"
        }
        for word, emoji in emojis.items():
            text = re.sub(r'\b' + re.escape(word) + r'\b', word + ' ' + emoji, text, flags=re.IGNORECASE)
        return text

    def forward(self, input_ids, attention_mask=None, project_graph_embedding=None):
        device = self._model_device()
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        llm_outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = llm_outputs.hidden_states[-1]

        if project_graph_embedding is not None:
            batch_size, seq_len, _ = last_hidden_state.shape
            expanded_graph_embedding = project_graph_embedding.unsqueeze(1).repeat(1, seq_len, 1)
            fused_state, _ = self.graph_attn(last_hidden_state, expanded_graph_embedding, expanded_graph_embedding)
            fused_state = self.graph_norm(fused_state)
            fused_state = fused_state + last_hidden_state
        else:
            fused_state = last_hidden_state

        logits = self.policy_head(fused_state.to(self.policy_head[1].weight.dtype))
        last_token_hidden_state = fused_state[:, -1, :]
        value = self.value_head(last_token_hidden_state.to(self.value_head[1].weight.dtype))

        return logits, value, fused_state

    def generate_response(self, prompt_text: str, max_new_tokens=512, do_sample=True, temperature=0.8, top_k=50, top_p=0.95):
        self.eval()
        device = self._model_device()

        template_prompt = (
            "You are a helpful and experienced AI assistant for game development. "
            "Your task is to provide a complete solution to the user's request. "
            "First, give a friendly and clear explanation of the solution in Thai, using emojis to make it easy to understand. "
            "Then, provide the complete, well-commented code. "
            "The user's request is: "
            f"'{prompt_text}'\n"
            "Here is the solution:"
        )

        input_ids = self.tokenizer.encode(template_prompt, return_tensors="pt").to(device)
        attention_mask = torch.ones_like(input_ids).to(device)

        generation_params = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "num_return_sequences": 1,
            "return_dict_in_generate": True,
            "output_scores": True,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p
        }

        if do_sample:
            temp_decay_factor = 0.95
            generation_params['temperature'] *= temp_decay_factor

        generated_outputs = self.llm.generate(**generation_params)
        generated_ids = generated_outputs.sequences[0]
        full_response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        start_index = full_response.find(template_prompt)
        if start_index != -1:
            generated_text = full_response[start_index + len(template_prompt):].strip()
        else:
            # Fallback if the prompt is not found in the output
            # This can happen with some models that rephrase the beginning
            input_text_decoded = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            if full_response.startswith(input_text_decoded):
                 generated_text = full_response[len(input_text_decoded):].strip()
            else:
                 generated_text = full_response

        return generated_text, generated_outputs.scores

# --- 2. Advanced Code Evaluation and MARL Agents ---
class CodeEvaluator:
    def __init__(self, use_mock_reward=False, language="lua"):
        self.use_mock_reward = use_mock_reward
        self.language = language
        # Patterns for different languages
        self.safe_patterns_lua = [
            r'io\.open', r'os\.execute', r'os\.getenv', r'require\(["\'](http|socket)',
            r'HttpService', r'RunService', r'pcall\s*\(\s*require'
        ]
        self.safe_patterns_cs = [
            r'System\.IO\.File', r'System\.Diagnostics\.Process', r'System\.Net\.Http'
        ]
        self.safe_patterns_gd = [
            r'OS\.execute', r'File\.new', r'Directory\.new'
        ]
        # NEW: More advanced security vulnerability patterns
        self.vulnerability_patterns = {
            "lua": [r'loadstring'], # Potential Remote Code Execution
            "c#": [r'SqlCommand.*\.CommandText\s*=\s*".*"\s*\+'], # Potential SQL Injection
            "gdscript": [],
            "c++": [r'strcpy', r'sprintf'] # Potential Buffer Overflows
        }
        self.llm = None # Will be set during training

    def _pre_check_code(self, code_string):
        patterns = []
        if self.language == "lua":
            patterns = self.safe_patterns_lua
        elif self.language == "c#":
            patterns = self.safe_patterns_cs
        elif self.language == "gdscript":
            patterns = self.safe_patterns_gd

        for pattern in patterns:
            if re.search(pattern, code_string, re.IGNORECASE):
                return False, f"Potential security risk detected: {pattern}"
        return True, "No security risks detected."

    # NEW: Enhanced security audit method
    def _security_audit(self, code_string):
        """
        Performs a more detailed security check for common vulnerabilities.
        Returns a penalty score and a log of findings.
        """
        penalty = 0.0
        findings = []
        lang_patterns = self.vulnerability_patterns.get(self.language, [])
        for pattern in lang_patterns:
            if re.search(pattern, code_string):
                findings.append(f"High-risk pattern found: '{pattern}'. This could lead to security vulnerabilities.")
                penalty -= 5.0 # Apply a significant penalty
        if not findings:
            return 0.0, "No high-risk security patterns found."
        return penalty, "\n".join(findings)

    def _static_analysis(self, code_string):
        if self.language != "lua":
            logging.warning(f"Static analysis not implemented for {self.language}. Skipping.")
            return 0.0, "Skipped", "Static analysis not available."

        temp_file_path = "temp_code.lua"
        with open(temp_file_path, "w", encoding="utf-8") as f:
            f.write(code_string)

        try:
            result = subprocess.run(['luacheck', temp_file_path], capture_output=True, text=True, timeout=5)
            os.remove(temp_file_path)
            if "No issues found" in result.stdout:
                return 1.0, "Syntax check: OK", result.stdout
            else:
                return -1.0, f"Syntax check failed: {result.stdout}", result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
            return -1.0, f"Error during static analysis: {e}", ""

    def _dynamic_analysis(self, code_string):
        if self.language != "lua":
            logging.warning(f"Dynamic analysis not implemented for {self.language}. Skipping.")
            return 0.0, "Skipped", "Dynamic analysis not available."

        if shutil.which("docker") is None:
            logging.warning("Docker not installed, skipping functional test.")
            return 0.0, "Docker not available", "Docker not found."

        temp_file_path = "temp_code.lua"
        with open(temp_file_path, "w", encoding="utf-8") as f:
            f.write(code_string)

        docker_command = [
            'docker', 'run', '--rm', '--network', 'none', '--pids-limit', '256',
            '--cpus', '1', '--memory', '512m', '--ulimit', 'cpu=5',
            '--read-only', '-v', f'{os.getcwd()}/temp_code.lua:/app/temp_code.lua:ro',
            '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
            'luau_runtime_image', 'luau', '/app/temp_code.lua'
        ]

        try:
            start_time = time.time()
            result = subprocess.run(docker_command, capture_output=True, text=True, timeout=10)
            execution_time = time.time() - start_time

            if result.returncode == 0:
                reward = 10.0
                efficiency_reward = max(0, 1.0 - execution_time / 5.0)
                return reward + efficiency_reward, f"Functional test: Success! Output: {result.stdout}", result.stdout + "\n" + result.stderr
            else:
                return -5.0, f"Functional test failed. Exit code: {result.returncode}", result.stdout + "\n" + result.stderr
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
            return -2.0, f"Error during dynamic analysis: {e}", ""
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def _assess_readability(self, code_string):
        # A simple heuristic-based assessment
        lines = code_string.split('\n')
        comment_lines = sum(1 for line in lines if line.strip().startswith('--') or line.strip().startswith('//'))
        blank_lines = sum(1 for line in lines if not line.strip())

        total_lines = len(lines)
        if total_lines == 0: return 0.0

        # Reward for comments and blank lines, penalize for long lines
        comment_ratio = comment_lines / total_lines
        blank_ratio = blank_lines / total_lines

        long_line_penalty = sum(1 for line in lines if len(line) > 80)

        readability_score = (comment_ratio * 0.5) + (blank_ratio * 0.3) - (long_line_penalty * 0.1)
        return max(0, min(1, readability_score))

    def evaluate(self, code_string, project_graph):
        if self.use_mock_reward:
            return {
                "total_reward": random.uniform(-1, 10),
                "correctness_score": random.uniform(0, 10),
                "efficiency_score": random.uniform(0, 1),
                "knowledge_graph_score": random.uniform(0, 5),
                "readability_score": random.uniform(0, 1),
                "security_score": 0.0,
                "luacheck_log": "Mock luacheck log",
                "docker_log": "Mock docker log",
                "security_log": "Security audit skipped for mock reward."
            }

        is_safe, security_log = self._pre_check_code(code_string)
        if not is_safe:
            return {
                "total_reward": -10.0,
                "correctness_score": -10.0,
                "efficiency_score": 0.0,
                "knowledge_graph_score": 0.0,
                "readability_score": 0.0,
                "security_score": -10.0,
                "luacheck_log": "Skipped due to security risk.",
                "docker_log": security_log,
                "security_log": security_log
            }

        security_score, detailed_security_log = self._security_audit(code_string)
        syntax_score, _, luacheck_log = self._static_analysis(code_string)
        functional_score, _, docker_log = self._dynamic_analysis(code_string)
        correctness_score = (syntax_score + functional_score) / 2
        efficiency_score = max(0, 1.0 - len(code_string) / 1000)
        readability_score = self._assess_readability(code_string)

        kg_score = 0
        if project_graph is not None and project_graph.num_nodes > 0:
            asset_nodes = [i for i, data in enumerate(project_graph.x.tolist()) if project_graph.node_type[i] == 1]
            if asset_nodes:
                if 'Instance.new("Part")' in code_string or 'TextureId' in code_string:
                    kg_score += 2.0
                asset_node_index = asset_nodes[0]
                asset_id = project_graph.node_id[asset_node_index]
                if asset_id and str(asset_id) in code_string:
                    kg_score += 3.0

            # Check for library/API usage in the graph
            if hasattr(project_graph, 'node_type') and 2 in project_graph.node_type:
                if 'Roblox.HttpService' in code_string:
                    kg_score += 2.0
                if 'UnityEngine.Rigidbody' in code_string:
                    kg_score += 2.0

        # NEW: Integrate security score into the total reward calculation
        total_reward = (0.35 * correctness_score) + \
                       (0.15 * efficiency_score) + \
                       (0.15 * kg_score) + \
                       (0.15 * readability_score) + \
                       (0.20 * security_score) # Security has a significant weight

        return {
            "total_reward": total_reward,
            "correctness_score": correctness_score,
            "efficiency_score": efficiency_score,
            "knowledge_graph_score": kg_score,
            "readability_score": readability_score,
            "security_score": security_score,
            "luacheck_log": luacheck_log,
            "docker_log": docker_log,
            "security_log": detailed_security_log
        }

class AssetGeneratorAgent:
    def __init__(self):
        logging.info("AssetGeneratorAgent initialized.")
        self.asset_database = {}

    def generate_asset(self, prompt):
        asset_id = f"asset_{hash(prompt) % 10000}"
        asset_type = "Texture" if "texture" in prompt.lower() else "Model" if "model" in prompt.lower() else "Part"
        self.asset_database[asset_id] = {"prompt": prompt, "type": asset_type}
        logging.info(f"Generated mock asset with ID {asset_id} for prompt '{prompt}'")
        return asset_id

# --- NEW: Bug Report Generator Agent ---
class BugReportGeneratorAgent:
    def __init__(self, model):
        self.model = model

    def generate_report(self, code_string, error_log):
        prompt = f"""
        Analyze the following code and error log to create a professional bug report.
        Code:
        ```
        {code_string}
        ```
        Error Log:
        {error_log}

        Provide a clear bug report with the following sections:
        - **Bug Description:** What is the issue?
        - **Steps to Reproduce:** How can this bug be replicated?
        - **Expected Behavior:** What should happen?
        - **Actual Behavior:** What is happening instead?
        - **Relevant Code Line:** Specify the line number where the bug likely originates.
        """
        bug_report, _ = self.model.generate_response(prompt, max_new_tokens=256, temperature=0.6)

        # Extract relevant info
        description = re.search(r'\*\*Bug Description:\*\*(.*?)\n\n', bug_report, re.DOTALL)
        line_num = re.search(r'\*\*Relevant Code Line:\*\*(.*?)\n', bug_report)

        return {
            'full_report': bug_report,
            'description': description.group(1).strip() if description else 'No description.',
            'line_number': int(line_num.group(1).strip()) if line_num and line_num.group(1).strip().isdigit() else None
        }

class CodeGeneratorAgent:
    def __init__(self, model):
        self.model = model

    def generate(self, prompt, project_graph_embedding=None, context_examples=None):
        # NEW: Augment prompt with retrieved examples (RAG)
        if context_examples:
            examples_text = "\n\n---\nHere are some relevant examples of good code:\n---\n"
            for ex in context_examples:
                examples_text += f"```lua\n{ex}\n```\n"
            prompt += examples_text

        return self.model.generate_response(prompt)

class CodeCriticAgent:
    def __init__(self, model):
        self.model = model

    def critique(self, code_string, evaluation_results, asset_id=None):
        # NEW: Include security score in the critique prompt
        critique_prompt = f"""
        Analyze the following code and its evaluation results.
        Code:
        ```
        {code_string}
        ```

        Evaluation Results:
        Correctness Score: {evaluation_results['correctness_score']}
        Efficiency Score: {evaluation_results['efficiency_score']}
        Knowledge Graph Score: {evaluation_results['knowledge_graph_score']}
        Readability Score: {evaluation_results['readability_score']}
        Security Score: {evaluation_results.get('security_score', 'N/A')}
        Logs: {evaluation_results['luacheck_log']} | {evaluation_results['docker_log']}
        Security Log: {evaluation_results.get('security_log', 'N/A')}
        Asset ID: {asset_id}

        Provide a constructive critique. Identify bugs, errors, inefficiencies, or security risks. Suggest specific improvements.
        """
        critique_response, _ = self.model.generate_response(critique_prompt, max_new_tokens=256, temperature=0.5)
        return critique_response

class CodeRefinementAgent:
    def __init__(self, model):
        self.model = model

    def refine(self, original_code, critique, bug_report=None, asset_id=None, context_examples=None):
        # NEW: Augment prompt with retrieved examples (RAG)
        refinement_prompt = f"""
        Based on the critique and bug report, refactor the original code.
        Original Code:
        ```
        {original_code}
        ```
        Critique:
        {critique}

        Bug Report (if available):
        {bug_report['full_report'] if bug_report else "N/A"}

        Consider the asset with ID: {asset_id}.
        """
        if context_examples:
            examples_text = "\n\n---\nHere are some relevant examples to guide your refinement:\n---\n"
            for ex in context_examples:
                examples_text += f"```lua\n{ex}\n```\n"
            refinement_prompt += examples_text

        refinement_prompt += "\nProvide the complete, corrected, and improved code. Only output the code block."

        refined_code, _ = self.model.generate_response(refinement_prompt, max_new_tokens=512, temperature=0.7)

        match = re.search(r'```(?:\w+)?\n(.*?)\n```', refined_code, re.DOTALL)
        if match:
            return match.group(1).strip()
        return refined_code

# --- NEW: Specialized Agents ---
class TestGenerationAgent:
    def __init__(self, model):
        self.model = model

    def generate_tests(self, code_string, language="Luau"):
        """Generates unit tests for the given code."""
        prompt = f"""
        You are an expert in software testing. Your task is to generate a comprehensive set of unit tests for the following {language} code.
        The tests should cover normal cases, edge cases, and potential error conditions.
        Use a common testing framework if applicable for the language (e.g., TestEZ for Luau).

        Code to test:
        ```
        {code_string}
        ```

        Provide the complete unit test script.
        """
        tests, _ = self.model.generate_response(prompt, max_new_tokens=512, temperature=0.6)
        return tests

    # NEW: Method for generating integration tests
    def generate_integration_tests(self, code_module_A, code_module_B, description, language="Luau"):
        """Generates integration tests for two interacting code modules."""
        prompt = f"""
        You are a senior QA engineer. Your task is to write an integration test for two modules.
        Description of interaction: {description}

        Module A:
        ```
        {code_module_A}
        ```

        Module B:
        ```
        {code_module_B}
        ```

        Write a complete integration test script in {language} that verifies the correct interaction between these two modules.
        Focus on testing the data flow and function calls between them.
        """
        tests, _ = self.model.generate_response(prompt, max_new_tokens=600, temperature=0.65)
        return tests


class DocumentationAgent:
    def __init__(self, model):
        self.model = model

    def generate_docs(self, code_string, language="Luau"):
        """Generates clear documentation for the given code."""
        prompt = f"""
        You are an expert technical writer. Your task is to create clear, concise, and easy-to-understand documentation for the following {language} code.
        Explain what the code does, describe its main functions/classes, and provide examples of how to use it.
        Format the output in Markdown.

        Code to document:
        ```
        {code_string}
        ```

        Provide the complete documentation.
        """
        docs, _ = self.model.generate_response(prompt, max_new_tokens=400, temperature=0.7)
        return docs

# NEW: Agent for automatic code refactoring
class AutoRefactoringAgent:
    def __init__(self, model):
        self.model = model

    def refactor(self, code_string, language="Luau"):
        """Automatically refactors code for cleanliness and maintainability."""
        prompt = f"""
        You are an AI code quality assistant. Your task is to refactor the following {language} code to improve its structure and readability without changing its functionality.
        Focus on these principles:
        - **DRY (Don't Repeat Yourself):** Consolidate redundant code.
        - **KISS (Keep It Simple, Stupid):** Simplify complex logic.
        - **Readability:** Improve variable names and add comments where necessary.
        - **Modularity:** Break down large functions into smaller, single-purpose functions.

        Code to refactor:
        ```
        {code_string}
        ```

        Provide only the refactored code block.
        """
        refactored_code, _ = self.model.generate_response(prompt, max_new_tokens=512, temperature=0.6)
        match = re.search(r'```(?:\w+)?\n(.*?)\n```', refactored_code, re.DOTALL)
        if match:
            return match.group(1).strip()
        return refactored_code


# NEW: Agent for game design ideas
class GameDesignerAgent:
    def __init__(self, model):
        self.model = model

    def propose_feature(self, context_prompt):
        """Proposes a new game feature based on a given context."""
        prompt = f"""
        You are a creative and experienced game designer.
        Based on the following request, propose a new, innovative game feature.
        Describe the feature, how it works, and why it would be fun for players.

        Request: "{context_prompt}"

        **Feature Proposal:**
        """
        proposal, _ = self.model.generate_response(prompt, max_new_tokens=300, temperature=0.85)
        return proposal


# --- NEW: Multi-Task Learning Agents ---
class CodeSummarizationAgent:
    def __init__(self, model):
        self.model = model

    def summarize(self, code_string, language="Luau"):
        prompt = f"""
        Summarize the main functionality of the following {language} code in a single, concise sentence.

        Code:
        ```
        {code_string}
        ```

        Summary:
        """
        summary, _ = self.model.generate_response(prompt, max_new_tokens=50, temperature=0.5)
        return summary

class CodeQuestionAnsweringAgent:
    def __init__(self, model):
        self.model = model

    def answer_question(self, code_string, question, language="Luau"):
        prompt = f"""
        Analyze the following {language} code and answer the question provided.

        Code:
        ```
        {code_string}
        ```

        Question: {question}

        Answer:
        """
        answer, _ = self.model.generate_response(prompt, max_new_tokens=100, temperature=0.5)
        return answer

# --- Data Handling (IMPROVED) ---
def download_code_from_github(engine_name: str, github_query: str, file_extensions: list, save_dir: str, github_token: str):
    """
    Downloads code files for a specific game engine from GitHub with rate limit handling.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    headers = {"Authorization": f"token {github_token}"}
    search_url = "https://api.github.com/search/repositories"

    # Count existing files to avoid re-downloading
    downloaded_count = 0
    for ext in file_extensions:
        downloaded_count += len(glob.glob(os.path.join(save_dir, f"*{ext}")))

    page = 1
    target_count = 1200 # <-- Increased target

    logging.info(f"[{engine_name}] Found {downloaded_count} existing files. Target is {target_count}.")

    initial_downloaded_count = downloaded_count

    while downloaded_count < target_count:
        params = {"q": github_query, "per_page": 100, "page": page}

        logging.info(f"[{engine_name}] Searching GitHub repositories (page {page})...")
        try:
            response = requests.get(search_url, headers=headers, params=params, timeout=30)

            # --- NEW: Rate Limit Handling ---
            if 'X-RateLimit-Remaining' in response.headers:
                remaining = int(response.headers['X-RateLimit-Remaining'])
                if remaining < 10:
                    reset_time = int(response.headers['X-RateLimit-Reset'])
                    sleep_duration = max(0, reset_time - time.time()) + 5 # Add 5s buffer
                    logging.warning(f"GitHub API rate limit low ({remaining} left). Sleeping for {sleep_duration:.0f} seconds.")
                    time.sleep(sleep_duration)

            response.raise_for_status()
            repos = response.json().get("items", [])

            if not repos:
                logging.info(f"[{engine_name}] No more repositories found.")
                break

        except requests.exceptions.RequestException as e:
            logging.error(f"Error connecting to GitHub API: {e}. Retrying search in 15 seconds...")
            time.sleep(15)
            continue

        for repo in repos:
            repo_name = repo["full_name"]
            default_branch = repo["default_branch"]
            try:
                files_url = f"https://api.github.com/repos/{repo_name}/git/trees/{default_branch}?recursive=1"
                files_response = requests.get(files_url, headers=headers, timeout=30)
                files_response.raise_for_status()
                files_tree = files_response.json().get("tree", [])

                for file in files_tree:
                    if any(file["path"].endswith(ext) for ext in file_extensions) and file.get("size", 0) > 100:
                        save_path = os.path.join(save_dir, f"{repo_name.replace('/', '_')}_{os.path.basename(file['path'])}")

                        if os.path.exists(save_path):
                            continue

                        file_content_url = f"https://api.github.com/repos/{repo_name}/contents/{file['path']}?ref={default_branch}"

                        retries = 3
                        for i in range(retries):
                            try:
                                content_response = requests.get(file_content_url, headers=headers, timeout=30)
                                content_response.raise_for_status()
                                file_data = content_response.json()

                                try:
                                    content = base64.b64decode(file_data["content"]).decode("utf-8")
                                    with open(save_path, "w", encoding="utf-8") as f:
                                        f.write(content)
                                    downloaded_count += 1
                                    if downloaded_count % 50 == 0:
                                        logging.info(f"[{engine_name}] Downloaded {downloaded_count}/{target_count} files.")
                                    break
                                except UnicodeDecodeError:
                                    logging.warning(f"Skipping file {file['path']} due to encoding error.")
                                    break

                            except requests.exceptions.RequestException as e:
                                logging.error(f"Connection error for {file['path']}. Retrying {i+1}/{retries}...")
                                time.sleep(5 * (i + 1)) # Exponential backoff
                            except KeyError:
                                logging.error(f"Could not decode content for {file['path']}. Skipping.")
                                break
                        else:
                            logging.error(f"Failed to download {file['path']} after {retries} retries.")
                            continue

                        if downloaded_count >= target_count:
                            break

                        time.sleep(1) # Be respectful to the API

                if downloaded_count >= target_count:
                    break

            except requests.exceptions.RequestException as e:
                logging.error(f"Skipping repo {repo_name} due to API error: {e}")
                continue

        if downloaded_count >= target_count:
            break

        page += 1

    logging.info(f"[{engine_name}] Successfully downloaded a total of {downloaded_count - initial_downloaded_count} new files.")
    return downloaded_count

def get_func_name(node):
    try:
        if hasattr(node.name.id, 'id'):
            return node.name.id.id
        return node.name.id
    except Exception:
        return None

def cache_embeddings(code_chunks, codebert_pipeline, cache_file="embeddings_cache.joblib"):
    # FIX C: Ensure codebert_pipeline is not None
    if codebert_pipeline is None:
        logging.error("CodeBERT pipeline is None. Cannot generate embeddings.")
        return {} # Return empty cache if pipeline is missing

    if os.path.exists(cache_file):
        try:
            cache = joblib.load(cache_file)
        except (IOError, joblib.UnpicklingError) as e:
            logging.error(f"Error loading embedding cache: {e}. Creating a new cache.")
            cache = {}
    else:
        cache = {}

    new_chunks = [chunk for chunk in code_chunks if chunk not in cache]
    if new_chunks:
        logging.info(f"Generating embeddings for {len(new_chunks)} new chunks...")
        try:
            # Batch processing for efficiency
            new_embeddings_raw = codebert_pipeline(new_chunks, batch_size=8)
            new_embeddings = [torch.tensor(emb).squeeze() for emb in new_embeddings_raw]

            for chunk, embedding in zip(new_chunks, new_embeddings):
                if embedding.dim() > 1:
                    embedding = embedding.mean(dim=0)
                cache[chunk] = embedding.tolist()
        except Exception as e:
            logging.error(f"Error generating CodeBERT embeddings: {e}")
            return {}

        joblib.dump(cache, cache_file)
        logging.info("Embeddings cached.")

    return cache

def visualize_graph(graph, filename="code_graph.png", max_nodes=50):
    if not hasattr(graph, 'x') or graph.x.shape[0] == 0:
        logging.warning("Cannot visualize an empty graph.")
        return

    num_nodes_to_display = min(graph.x.shape[0], max_nodes)

    if graph.x.shape[0] > max_nodes:
        logging.warning(f"Graph too large ({graph.x.shape[0]} nodes). Visualizing a random subgraph of {max_nodes} nodes.")
        node_indices = random.sample(range(graph.num_nodes), max_nodes)

        g = nx.DiGraph()
        node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(node_indices)}

        for i in range(graph.edge_index.shape[1]):
            src, dest = graph.edge_index[0][i].item(), graph.edge_index[1][i].item()
            if src in node_map and dest in node_map:
                g.add_edge(node_map[src], node_map[dest])
    else:
        g = nx.DiGraph()
        g.add_nodes_from(range(graph.x.shape[0]))
        if hasattr(graph, 'edge_index') and graph.edge_index.shape[1] > 0:
            edges = graph.edge_index.t().tolist()
            g.add_edges_from(edges)

    plt.figure(figsize=(14, 14))
    pos = nx.spring_layout(g) # Spring layout is often good for complex graphs
    nx.draw(g, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=8, edge_color='gray', arrows=True)
    plt.title("Code Knowledge Graph Visualization")
    plt.savefig(filename)
    plt.close() # Close the figure to free memory
    logging.info(f"Graph visualization saved to {filename}.")

# --- IMPROVED Graph Building with Usage Data and Libraries ---
def build_real_code_graph_ast(code_content, model, codebert_pipeline, asset_id=None, language="lua", design_doc=None, user_feedback=None):
    logging.info("Using AST-based graph builder with luaparser.")

    if language != "lua" or not luaparser_parser:
        logging.error("luaparser not found or language is not Lua. Cannot build AST graph.")
        return None

    try:
        ast_tree = luaparser_parser.parse(code_content)
        function_definitions = {}
        function_calls = []
        library_usages = set()
        variable_usages = {}

        def find_nodes(node, parent=None):
            if isinstance(node, luaparser_ast.Function):
                func_name = get_func_name(node)
                if func_name and func_name not in function_definitions:
                    function_definitions[func_name] = node
            elif isinstance(node, luaparser_ast.Call):
                func_name = get_func_name(node.func)
                if func_name:
                    function_calls.append({'name': func_name, 'node': node})
            elif isinstance(node, luaparser_ast.Index):
                if isinstance(node.idx, luaparser_ast.Id):
                    library_usages.add(node.idx.id)
            elif isinstance(node, luaparser_ast.Id):
                var_name = node.id
                if var_name not in variable_usages:
                    variable_usages[var_name] = 0
                variable_usages[var_name] += 1

            for child in luaparser_ast.walk(node):
                if child is not node:
                    find_nodes(child)

        find_nodes(ast_tree)

        all_chunks = re.split(r'\n(function|local function)', code_content)
        chunks = []
        for i in range(1, len(all_chunks), 2):
            if i + 1 < len(all_chunks):
                chunks.append(all_chunks[i] + all_chunks[i+1])
        valid_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

        if not valid_chunks:
            return None

        embeddings_cache = cache_embeddings(valid_chunks, codebert_pipeline)
        if not embeddings_cache: # Handle case where embedding fails
            return None

        G = nx.DiGraph()
        node_map = {}
        xs = []
        node_types = []
        node_ids = []

        for i, chunk in enumerate(valid_chunks):
            embedding = embeddings_cache.get(chunk)
            if embedding is not None:
                G.add_node(i, type='code')
                node_types.append(0) # 0 for code
                node_ids.append(None)
                xs.append(torch.tensor(embedding, dtype=torch.float32))
                func_name_match = re.search(r'(?:function|local function)\s+([a-zA-Z0-9_:]+)', chunk)
                if func_name_match:
                    node_map[func_name_match.group(1)] = i

        # FIX D: Add 'usage_data' as a feature and pad to fixed dimension
        for i, chunk in enumerate(valid_chunks):
            chunk_usage = sum(variable_usages.get(var, 0) for var in re.findall(r'\b[a-zA-Z0-9_]+\b', chunk))
            if i < len(xs):
                current_embedding = xs[i]
                usage_feature = torch.tensor([chunk_usage / 10.0]) # Normalize usage
                combined_features = torch.cat([current_embedding, usage_feature], dim=0)

                padding_needed = model.fixed_graph_embedding_dim - combined_features.shape[0]
                if padding_needed < 0: # Truncate if too long
                     combined_features = combined_features[:model.fixed_graph_embedding_dim]
                elif padding_needed > 0: # Pad if too short
                    combined_features = pad(combined_features, (0, padding_needed), 'constant', 0)
                xs[i] = combined_features

        # Add library/API nodes
        library_node_start_idx = len(xs)
        for lib in library_usages:
            # Pad to fixed dimension
            lib_embedding = torch.randn(model.fixed_graph_embedding_dim, dtype=torch.float32)
            xs.append(lib_embedding)
            node_types.append(2) # 2 for library/API
            node_ids.append(lib)
            node_map[lib] = len(xs) - 1

        if not xs:
            return None

        edge_list = []
        edge_attr = []
        for call in function_calls:
            caller_name = None
            for func_name, def_node in function_definitions.items():
                if def_node.location and call['node'].location:
                    if def_node.location.line <= call['node'].location.line and def_node.end_location.line >= call['node'].location.line:
                        caller_name = func_name
                        break

            if caller_name and call['name'] in node_map and caller_name in node_map:
                caller_node_idx = node_map[caller_name]
                callee_node_idx = node_map[call['name']]
                if caller_node_idx != callee_node_idx:
                    edge_list.append((caller_node_idx, callee_node_idx))
                    edge_attr.append(0) # 0 for 'calls'

        # Add library/API edges
        for i, chunk in enumerate(valid_chunks):
            for lib in library_usages:
                if lib in chunk:
                    edge_list.append((i, node_map[lib]))
                    edge_attr.append(2) # 2 for 'uses_library'

        if asset_id:
            asset_node_idx = len(xs)
            # Pad to fixed dimension
            asset_embedding = torch.zeros(model.fixed_graph_embedding_dim, dtype=torch.float32)
            xs.append(asset_embedding)
            node_types.append(1) # 1 for asset
            node_ids.append(asset_id)

            for i, chunk in enumerate(valid_chunks):
                if 'Instance.new' in chunk or asset_id in chunk:
                    edge_list.append((i, asset_node_idx))
                    edge_attr.append(1) # 1 for 'uses'

        # NEW: Add Game Design Doc and User Feedback nodes (simulation)
        if design_doc:
            doc_node_idx = len(xs)
            doc_embedding = torch.randn(model.fixed_graph_embedding_dim)
            xs.append(doc_embedding)
            node_types.append(3) # 3 for design doc
            node_ids.append("design_doc_1")
            # Connect all code nodes to the design doc
            for i in range(len(valid_chunks)):
                edge_list.append((i, doc_node_idx))
                edge_attr.append(3) # 3 for 'implements_design'

        if user_feedback:
            feedback_node_idx = len(xs)
            feedback_embedding = torch.randn(model.fixed_graph_embedding_dim)
            xs.append(feedback_embedding)
            node_types.append(4) # 4 for user feedback
            node_ids.append("user_feedback_1")
            # Connect feedback to a random code chunk
            if valid_chunks:
                edge_list.append((random.randint(0, len(valid_chunks)-1), feedback_node_idx))
                edge_attr.append(4) # 4 for 'addresses_feedback'


        edge_list = list(set(edge_list))

        x = torch.stack(xs, dim=0)

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(1) if edge_list else torch.empty((0, 1), dtype=torch.float)

        py_g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        py_g = py_g.to(model._model_device())

        # FIX D: Removed dynamic resizing of the embedding projection layer.
        # The model is now initialized with a fixed-size layer.

        proj_x = model.embedding_proj(py_g.x)
        py_g.x = proj_x
        py_g.node_type = torch.tensor(node_types, dtype=torch.long)
        py_g.node_id = node_ids

        return py_g
    except Exception as e:
        logging.error(f"Error building AST graph: {e}")
        return None

def build_real_code_graph_fallback(code_content, model, codebert_pipeline):
    logging.info("Using fallback regex-based graph builder.")

    code_chunks = []
    parts = re.split(r'\n(function|local function)', code_content)
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            code_chunks.append(parts[i] + parts[i+1])

    valid_chunks = [chunk.strip() for chunk in code_chunks if chunk.strip()]
    if not valid_chunks:
        return None

    embeddings_cache = cache_embeddings(valid_chunks, codebert_pipeline)
    if not embeddings_cache:
        return None

    node_map = {}
    xs = []
    node_types = []

    for i, chunk in enumerate(valid_chunks):
        embedding_list = embeddings_cache.get(chunk)
        if embedding_list is not None:
            embedding = torch.tensor(embedding_list, dtype=torch.float32)
            # FIX D: Pad fallback embeddings to the fixed dimension as well
            padding_needed = model.fixed_graph_embedding_dim - embedding.shape[0]
            if padding_needed > 0:
                embedding = pad(embedding, (0, padding_needed), 'constant', 0)
            elif padding_needed < 0:
                embedding = embedding[:model.fixed_graph_embedding_dim]

            xs.append(embedding)
            node_types.append(0)
            func_name_match = re.search(r'(?:function|local function)\s+([a-zA-Z0-9_:]+)', valid_chunks[i])
            if func_name_match:
                node_map[func_name_match.group(1)] = len(xs) - 1

    if not xs: return None

    edge_list = []
    edge_attr = []
    for i, chunk in enumerate(valid_chunks):
        for func_name, node_idx in node_map.items():
            if i == node_idx: continue
            if f'{func_name}(' in valid_chunks[i]:
                edge_list.append((i, node_idx))
                edge_attr.append(0)

    edge_list = list(set(edge_list))

    x = torch.stack(xs, dim=0)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(1) if edge_list else torch.empty((0, 1), dtype=torch.float)

    py_g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    py_g = py_g.to(model._model_device())
    proj_x = model.embedding_proj(py_g.x)
    py_g.x = proj_x
    py_g.node_type = torch.tensor(node_types, dtype=torch.long)
    py_g.node_id = [None] * len(node_types)
    return py_g


class CodeDataset(Dataset):
    def __init__(self, data_dir, tokenizer, model, codebert_pipeline, max_length=512, graph_cache_dir="graph_cache", file_extensions=None):
        if file_extensions is None:
            file_extensions = ["*.lua", "*.luau"]

        self.file_paths = []
        for ext in file_extensions:
            self.file_paths.extend(glob.glob(os.path.join(data_dir, ext)))

        if not self.file_paths:
            raise FileNotFoundError(f"No files with extensions {file_extensions} found in {data_dir}.")

        self.tokenizer = tokenizer
        self.model = model
        self.codebert_pipeline = codebert_pipeline
        self.max_length = max_length
        self.graph_cache_dir = graph_cache_dir
        if not os.path.exists(self.graph_cache_dir):
            os.makedirs(self.graph_cache_dir)
        self.tokenized_data = []
        self._load_data()

    def _load_data(self):
        logging.info(f"Loading and preprocessing {len(self.file_paths)} files...")
        for file_path in tqdm(self.file_paths):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                tokenized_data = self.tokenizer(
                    content,
                    return_tensors="pt",
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length"
                )

                self.tokenized_data.append({
                    'file_path': file_path,
                    'input_ids': tokenized_data['input_ids'].squeeze(),
                    'attention_mask': tokenized_data['attention_mask'].squeeze(),
                })
            except Exception as e:
                logging.error(f"Skipping file {file_path} due to read/tokenize error: {e}")

    def get_graph(self, file_path, asset_id=None):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        content_hash = str(hash(content))
        cache_path = os.path.join(self.graph_cache_dir, f"{content_hash}.joblib")

        if os.path.exists(cache_path):
            try:
                return joblib.load(cache_path)
            except Exception as e:
                logging.error(f"Error loading graph from cache: {e}. Rebuilding...")

        graph_data = build_real_code_graph_ast(content, self.model, self.codebert_pipeline, asset_id, design_doc="mock", user_feedback="mock")

        if graph_data is not None:
            joblib.dump(graph_data, cache_path)

        return graph_data

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        item = self.tokenized_data[idx]
        file_path = item['file_path']

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        graph_data = self.get_graph(file_path)

        return {
            'input_ids': item['input_ids'],
            'attention_mask': item['attention_mask'],
            'code_content': content,
            'code_graph_data': graph_data
        }

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    code_contents = [item['code_content'] for item in batch]
    graph_data_list = [item['code_graph_data'] for item in batch if item['code_graph_data'] is not None]

    batched_graph = None
    if graph_data_list:
        batched_graph = Batch.from_data_list(graph_data_list)

    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'code_contents': code_contents, 'code_graph_data': batched_graph}

def _calculate_ppo_loss(model, accelerator, state, action, action_len, old_log_prob, reward, next_state_embedding, old_value_preds, done, curiosity_weight, clip_epsilon, weights=None):
    # Move next_state_embedding to the correct device if it exists
    if next_state_embedding is not None:
        next_state_embedding = next_state_embedding.to(accelerator.device)

    full_input_ids = torch.cat([state, action], dim=1).to(accelerator.device)

    if full_input_ids.size(1) > model.llm.config.max_position_embeddings:
        logging.warning("Sequence length exceeds max position embeddings. Truncating.")
        full_input_ids = full_input_ids[:, :model.llm.config.max_position_embeddings]

    logits, value, fused_state = model(full_input_ids, None, project_graph_embedding=next_state_embedding)

    if action.numel() == 0 or logits.numel() == 0:
        return None, None, None, None, None

    logits_gen = logits[:, state.size(1)-1:-1, :]
    log_probs = log_softmax(logits_gen, dim=-1)

    action_mask = (action != model.tokenizer.pad_token_id).to(accelerator.device)

    action_log_probs = log_probs.gather(2, action.unsqueeze(-1)).squeeze(-1)
    masked_action_log_probs = action_log_probs * action_mask
    current_log_prob = masked_action_log_probs.sum(dim=-1) / action_mask.sum(dim=-1).clamp(min=1e-6)

    gamma = 0.99

    # FIX E: Corrected Advantage and Return calculation for batch-based Actor-Critic
    # Ensure shapes are compatible for broadcasting
    reward = reward.squeeze() if reward.dim() > 1 else reward
    done = done.squeeze() if done.dim() > 1 else done
    value = value.squeeze()
    old_value_preds = old_value_preds.squeeze()

    # Advantage A(s,a) = r + gamma * V(s') * (1-done) - V(s)
    advantages = reward + gamma * value.detach() * (1 - done.int()) - old_value_preds.detach()
    returns = advantages + old_value_preds.detach()


    current_state_features = fused_state[:, -2, :]
    next_state_features = fused_state[:, -1, :]
    action_features = model.llm.get_input_embeddings()(action).mean(dim=1) # Average embeddings over sequence length

    curiosity_loss = model.curiosity_module(current_state_features, action_features, next_state_features)
    intrinsic_reward = curiosity_loss.detach()

    total_reward = reward + curiosity_weight * intrinsic_reward

    ratio = torch.exp(current_log_prob - old_log_prob.detach())
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)

    policy_loss_unweighted = -torch.min(ratio * advantages, clipped_ratio * advantages)
    value_loss_unweighted = mse_loss(value, returns)

    if weights is not None:
        policy_loss = (policy_loss_unweighted * weights).mean()
        value_loss = (value_loss_unweighted * weights).mean()
    else:
        policy_loss = policy_loss_unweighted.mean()
        value_loss = value_loss_unweighted.mean()


    entropy_beta = 0.01
    probs = softmax(logits_gen, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()

    total_loss = policy_loss + 0.5 * value_loss - entropy_beta * entropy + curiosity_loss

    return total_loss, policy_loss, value_loss, entropy, curiosity_loss


# NEW: Mock function to simulate human feedback in the loop
def get_human_feedback(code_string):
    """Simulates a user providing a rating for the generated code."""
    # In a real system, this would be a UI where a user rates the code.
    # Here, we'll mock it based on code properties.
    if "error" in code_string.lower() or "bug" in code_string.lower():
        return -2.0 # User is unhappy with buggy code
    if len(code_string) > 500:
        return 0.5 # User might find very long code less helpful
    return 1.5 # User is generally happy

def train_ppo_with_accelerator(model, data_loader, val_loader, optimizer, codebert_pipeline, num_epochs, gradient_accumulation_steps, use_mock_reward, visualize_graphs, clip_epsilon, curiosity_weight, engine_name="", language="lua"):
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
    model, optimizer, data_loader, val_loader = accelerator.prepare(model, optimizer, data_loader, val_loader)

    # --- Initialize all agents ---
    code_evaluator = CodeEvaluator(use_mock_reward=use_mock_reward, language=language)
    asset_generator_agent = AssetGeneratorAgent()
    bug_report_generator_agent = BugReportGeneratorAgent(model)
    code_generator_agent = CodeGeneratorAgent(model)
    code_critic_agent = CodeCriticAgent(model)
    code_refinement_agent = CodeRefinementAgent(model)
    test_generation_agent = TestGenerationAgent(model)
    documentation_agent = DocumentationAgent(model)
    # NEW: Initialize new agents
    auto_refactoring_agent = AutoRefactoringAgent(model)
    game_designer_agent = GameDesignerAgent(model)
    vectorized_memory = VectorizedMemory(codebert_pipeline)


    logging.info(f"Starting PPO training with Accelerator for {engine_name}...")
    model.train()

    total_steps = (num_epochs * len(data_loader)) // gradient_accumulation_steps
    progress_bar = tqdm(range(total_steps), desc=f"Training ({engine_name})")

    replay_buffer = PrioritizedReplayBuffer(capacity=1024)
    visualize_interval = 100

    # FIX F: Initialize logging variables to prevent potential undefined errors
    total_loss, policy_loss, value_loss, entropy, curiosity_loss = [torch.tensor(0.0) for _ in range(5)]

    for epoch in range(num_epochs):
        for step, batch in enumerate(data_loader):
            if batch is None:
                continue

            initial_code_string = batch['code_contents'][0]

            # --- NEW: Game Designer Agent Step ---
            if step % 75 == 0:
                 design_proposal = game_designer_agent.propose_feature(f"A new feature for a {engine_name} game involving player interaction.")
                 logging.info(f"\n--- Game Design Proposal (Step {step}) ---\n{design_proposal}\n")

            # --- RAG Step 1: Retrieve similar code from long-term memory ---
            retrieved_examples = vectorized_memory.retrieve_similar(initial_code_string)

            asset_prompt = "A red spinning part"
            generated_asset_id = asset_generator_agent.generate_asset(asset_prompt)

            # --- Pass retrieved examples to the generator ---
            generated_code_string, _ = code_generator_agent.generate(
                prompt=f"Improve the following code to use the asset ID {generated_asset_id}:\n{initial_code_string}",
                context_examples=retrieved_examples
            )

            # FIX C: Pass the codebert_pipeline to the graph builder
            project_graph = build_real_code_graph_ast(generated_code_string, model, codebert_pipeline, asset_id=generated_asset_id)
            reward_dict = code_evaluator.evaluate(generated_code_string, project_graph)

            # --- Bug Fixing and Refinement Loop ---
            if reward_dict['total_reward'] < 0 and not use_mock_reward:
                bug_report = bug_report_generator_agent.generate_report(generated_code_string, reward_dict['docker_log'])
                critique = code_critic_agent.critique(generated_code_string, reward_dict, asset_id=generated_asset_id)
                refined_code_string = code_refinement_agent.refine(generated_code_string, critique, bug_report=bug_report, asset_id=generated_asset_id, context_examples=retrieved_examples)
            else:
                critique = code_critic_agent.critique(generated_code_string, reward_dict, asset_id=generated_asset_id)
                refined_code_string = code_refinement_agent.refine(generated_code_string, critique, asset_id=generated_asset_id, context_examples=retrieved_examples)
                bug_report = None

            # --- NEW: Auto-Refactoring Step ---
            if reward_dict['readability_score'] < 0.5: # Only refactor if readability is low
                logging.info(f"Readability score is low ({reward_dict['readability_score']:.2f}). Attempting auto-refactoring...")
                refined_code_string = auto_refactoring_agent.refactor(refined_code_string, language)


            # --- NEW: Multi-Task Learning Step ---
            summary = CodeSummarizationAgent(model).summarize(refined_code_string, language)
            question = "What does the main function do?"
            answer = CodeQuestionAnsweringAgent(model).answer_question(refined_code_string, question, language)

            if step % 50 == 0:
                logging.info(f"Step {step} - Summary: {summary}")
                logging.info(f"Step {step} - Q: {question} A: {answer}")

            # --- Use specialized agents ---
            generated_tests = test_generation_agent.generate_tests(refined_code_string, language)
            generated_docs = documentation_agent.generate_docs(refined_code_string, language)

            if step % 50 == 0: # Log occasionally to avoid spam
                logging.info(f"\n--- Generated Tests (Step {step}) ---\n{generated_tests}")
                logging.info(f"\n--- Generated Docs (Step {step}) ---\n{generated_docs}")

            refined_project_graph = build_real_code_graph_ast(refined_code_string, model, codebert_pipeline, asset_id=generated_asset_id)
            refined_reward_dict = code_evaluator.evaluate(refined_code_string, refined_project_graph)

            # --- NEW: Human-in-the-Loop Feedback Simulation ---
            human_reward = get_human_feedback(refined_code_string)
            final_reward_value = refined_reward_dict['total_reward'] + human_reward
            reward = torch.tensor([final_reward_value]).float()

            # --- RAG Step 2: Add high-quality code to long-term memory ---
            vectorized_memory.add_experience(refined_code_string, final_reward_value)


            project_graph_embedding = None
            if refined_project_graph is not None:
                graph_data = refined_project_graph.to(accelerator.device)
                project_graph_embedding = model.graph_memory(graph_data.x, graph_data.edge_index, graph_data.edge_attr, graph_data.batch)

                if visualize_graphs and step % visualize_interval == 0 and not use_mock_reward:
                    visualize_graph(graph_data.cpu(), filename=f"graph_{engine_name}_epoch{epoch}_step{step}.png")
            else:
                logging.warning(f"Skipping graph creation for this step due to an issue.")

            with torch.no_grad():
                gen_ids = model.tokenizer.encode(refined_code_string, return_tensors="pt")
                full_input_ids = torch.cat([batch['input_ids'], gen_ids], dim=1).to(accelerator.device)

                if full_input_ids.size(1) > model.llm.config.max_position_embeddings:
                    full_input_ids = full_input_ids[:, :model.llm.config.max_position_embeddings]

                logits_full, value_preds, _ = model(full_input_ids, None, project_graph_embedding=project_graph_embedding)

                gen_len = gen_ids.size(1)
                logits_gen = logits_full[:, -gen_len-1:-1, :]
                log_probs = log_softmax(logits_gen, dim=-1)

                action_mask = (gen_ids != model.tokenizer.pad_token_id).to(accelerator.device)
                action_log_probs = log_probs.gather(2, gen_ids.unsqueeze(-1).to(accelerator.device)).squeeze(-1)
                masked_action_log_probs = action_log_probs * action_mask

                gathered_log_probs = masked_action_log_probs.sum(dim=-1) / action_mask.sum(dim=-1).clamp(min=1e-6)

            # FIX B: Detach embedding and move to CPU before pushing to buffer to save VRAM
            project_graph_embedding_cpu = project_graph_embedding.detach().cpu() if project_graph_embedding is not None else None
            experience = (
                batch['input_ids'].cpu(),
                gen_ids.cpu(),
                torch.tensor([gen_ids.size(1)], dtype=torch.long).cpu(),
                gathered_log_probs.cpu(),
                reward.cpu(),
                project_graph_embedding_cpu,
                torch.tensor([False]).cpu()
            )
            replay_buffer.push(experience)

            if len(replay_buffer) >= 64:
                batch_data = replay_buffer.sample(64)
                if batch_data is None: continue

                batch_states, batch_actions, batch_action_lens, batch_old_log_probs, batch_rewards, batch_next_states, batch_dones, weights, indices = batch_data

                rewards_tensor = batch_rewards
                mean_reward = torch.mean(rewards_tensor)
                std_reward = torch.std(rewards_tensor)
                if std_reward.item() == 0:
                    std_reward = 1e-8
                normalized_rewards = (rewards_tensor - mean_reward) / std_reward

                total_loss_sum = 0
                losses_list = []

                for i in range(batch_states.size(0)):
                    with accelerator.accumulate(model):
                        state = batch_states[i].unsqueeze(0)
                        action = batch_actions[i].unsqueeze(0)[:, :batch_action_lens[i]]
                        old_log_prob = batch_old_log_probs[i].unsqueeze(0)
                        reward_val = normalized_rewards[i].unsqueeze(0)
                        next_state_item = batch_next_states[i].unsqueeze(0) if batch_next_states is not None and i < batch_next_states.size(0) else None
                        done = batch_dones[i].unsqueeze(0)

                        with torch.no_grad():
                            _, old_value_preds, _ = model(state.to(accelerator.device), None, project_graph_embedding=next_state_item.to(accelerator.device) if next_state_item is not None else None)

                        loss_outputs = _calculate_ppo_loss(
                            model, accelerator, state, action, batch_action_lens[i], old_log_prob, reward_val, next_state_item, old_value_preds, done, curiosity_weight, clip_epsilon, weights=weights[i].unsqueeze(0)
                        )
                        if loss_outputs is not None:
                             total_loss, policy_loss, value_loss, entropy, curiosity_loss = loss_outputs
                        else:
                             continue

                        if total_loss is not None:
                            accelerator.backward(total_loss)
                            total_loss_sum += total_loss.item()
                            losses_list.append(total_loss.item())

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                    if losses_list:
                        replay_buffer.update_priorities(indices, torch.tensor(losses_list))


                if total_loss_sum > 0:
                    avg_loss = total_loss_sum / len(losses_list) if losses_list else 0
                    logging.info(f"PPO Batch Loss: {avg_loss:.4f}, Mean Reward: {mean_reward.item():.2f}")

            if accelerator.sync_gradients:
                progress_bar.update(1)

        if (epoch + 1) % 5 == 0:
            logging.info(f"Epoch {epoch + 1}/{num_epochs}, Total Loss: {total_loss.item():.4f}, Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}, Curiosity Loss: {curiosity_loss.item():.4f}")
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)

            save_dir = os.path.join("model_checkpoints", engine_name, f"epoch_{epoch+1}")
            os.makedirs(save_dir, exist_ok=True)
            unwrapped_model.llm.save_pretrained(save_dir)
            logging.info(f"Model checkpoint for {engine_name} saved at epoch {epoch+1}")

    logging.info(f"PPO training for {engine_name} finished.")

# Optuna objective is now simplified, as it's run once before the main loop
def objective(trial, data_dir, model, codebert_pipeline, file_extensions):
    try:
        dataset = CodeDataset(data_dir, model.tokenizer, model, codebert_pipeline, file_extensions=file_extensions)
    except FileNotFoundError as e:
        logging.error(f"ERROR during Optuna setup: {e}")
        return float('inf')

    if len(dataset) == 0:
        logging.error("ERROR: Dataset is empty for hyperparameter tuning.")
        return float('inf')

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, _ = random_split(dataset, [train_size, val_size])

    lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [1, 2, 4]) # Smaller batches for tuning
    clip_epsilon = trial.suggest_float("clip_epsilon", 0.1, 0.3)
    curiosity_weight = trial.suggest_float("curiosity_weight", 0.01, 0.1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    accelerator = Accelerator()
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    total_loss_list = []
    max_steps = 50
    for step, batch in enumerate(train_loader):
        if step >= max_steps:
            break
        if batch is None: continue

        reward = torch.tensor([random.uniform(-1, 10)]).float().to(accelerator.device)
        gen_ids = model.tokenizer.encode("print('hello world')", return_tensors="pt")

        with torch.no_grad():
            full_input_ids = torch.cat([batch['input_ids'], gen_ids], dim=1).to(accelerator.device)
            if full_input_ids.size(1) > model.llm.config.max_position_embeddings:
                full_input_ids = full_input_ids[:, :model.llm.config.max_position_embeddings]
            _, value_preds, _ = model(full_input_ids, None, project_graph_embedding=None)
            log_probs = torch.randn(1).to(accelerator.device) # Mock log_probs

        loss_outputs = _calculate_ppo_loss(
            model, accelerator, batch['input_ids'], gen_ids, gen_ids.size(1), log_probs, reward, None, value_preds, torch.tensor([False]).to(accelerator.device), curiosity_weight, clip_epsilon
        )
        if loss_outputs is not None:
            total_loss = loss_outputs[0]
            if total_loss is not None:
                accelerator.backward(total_loss)
                optimizer.step()
                optimizer.zero_grad()
                total_loss_list.append(total_loss.item())

    if total_loss_list:
        return np.mean(total_loss_list)
    return float('inf')

def run_inference_examples(model):
    logging.info("\n--- Running Inference Examples ---")
    prompts = [
        "Create a script to make a part in Roblox spin constantly.",
        "Write a GDScript function in Godot to save player data to a JSON file.",
        "How do I handle character movement in Unity using C# and the new Input System?",
        "Show a basic C++ example of spawning an Actor in Unreal Engine."
    ]

    for i, prompt in enumerate(prompts):
        logging.info(f"\n--- Example {i+1}: Prompt: '{prompt}' ---")
        try:
            generated_response, _ = model.generate_response(prompt)
            logging.info(f"Generated Response:\n{generated_response}")
        except Exception as e:
            logging.error(f"Failed to generate response for prompt '{prompt}': {e}")

# ! ##################################################################
# ! ################ REWORKED MAIN FUNCTION ##########################
# ! ##################################################################
def main():
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        logging.error("CRITICAL ERROR: GITHUB_TOKEN environment variable is not set. Cannot download data.")
        sys.exit()

    # --- Configuration for each training stage ---
    engine_configs = [
        {
            "name": "Roblox",
            "data_dir": "roblox_code_data",
            "github_query": "roblox luau language:lua size:>100",
            "file_extensions": ["*.lua", "*.luau"],
            "mock_epochs": 10,
            "real_epochs": 15,
            "language": "luau"
        },
        {
            "name": "Godot",
            "data_dir": "godot_code_data",
            "github_query": "godot gdscript language:gdscript size:>100",
            "file_extensions": ["*.gd"],
            "mock_epochs": 10,
            "real_epochs": 15,
            "language": "gdscript"
        },
        {
            "name": "Unity",
            "data_dir": "unity_code_data",
            "github_query": "unity c# language:c# size:>100",
            "file_extensions": ["*.cs"],
            "mock_epochs": 10,
            "real_epochs": 15,
            "language": "c#"
        },
        {
            "name": "Unreal",
            "data_dir": "unreal_code_data",
            "github_query": "unreal engine language:c++ size:>100",
            "file_extensions": ["*.cpp", "*.h"],
            "mock_epochs": 10,
            "real_epochs": 15,
            "language": "c++"
        }
    ]

    # --- 1. Initialize Model and Pipelines ONCE ---
    # The same model instance will be sequentially fine-tuned on each engine's data.
    logging.info("Initializing MultiAgentLLM model...")
    model = MultiAgentLLM()
    logging.info("Initializing CodeBERT pipeline...")
    codebert_pipeline = pipeline("feature-extraction", model="microsoft/CodeBERT-base", tokenizer="microsoft/CodeBERT-base", device=0 if torch.cuda.is_available() else -1)

    # --- 2. Hyperparameter Tuning (Optional, done once at the start) ---
    best_params = {
        "lr": 2e-5,
        "batch_size": 2, # Smaller batch size is better for low VRAM
        "clip_epsilon": 0.2,
        "curiosity_weight": 0.05
    }
    try:
        logging.info("--- Starting Hyperparameter Tuning with Optuna on Roblox data (as a baseline) ---")
        first_engine = engine_configs[0]
        # Download data just for the tuning phase
        download_code_from_github(
            engine_name=first_engine["name"],
            github_query=first_engine["github_query"],
            file_extensions=[ext.strip("*") for ext in first_engine["file_extensions"]],
            save_dir=first_engine["data_dir"],
            github_token=github_token
        )
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, first_engine["data_dir"], model, codebert_pipeline, first_engine["file_extensions"]), n_trials=5) # Reduced trials for speed

        logging.info(f"Best trial value: {study.best_trial.value}")
        logging.info(f"Best hyperparameters from Optuna: {study.best_trial.params}")
        best_params.update(study.best_trial.params)

    except Exception as e:
        logging.error(f"Error during hyperparameter tuning: {e}. Falling back to default parameters.")

    os.makedirs("model_checkpoints", exist_ok=True)

    # --- 3. Sequential Training Loop ---
    # This is the core of the new, robust pipeline.
    # We iterate through each engine, train the model, save it, clean up, and then move to the next.
    for i, config in enumerate(engine_configs):
        engine_name = config["name"]
        data_dir = config["data_dir"]
        language = config["language"]
        
        logging.info(f"\n{'='*25}\n Stage {i+1}/{len(engine_configs)}: Processing Engine: {engine_name} \n{'='*25}")

        # --- Stage 3.1: Download Data for the CURRENT engine ---
        download_code_from_github(
            engine_name=engine_name,
            github_query=config["github_query"],
            file_extensions=[ext.strip("*") for ext in config["file_extensions"]],
            save_dir=data_dir,
            github_token=github_token
        )

        try:
            # --- Stage 3.2: Create Dataset and Dataloaders for the CURRENT engine ---
            # Using a try-except block to make the process robust.
            # If one engine fails (e.g., no data), it will skip to the next.
            dataset = CodeDataset(data_dir, model.tokenizer, model, codebert_pipeline, file_extensions=config["file_extensions"])

            if len(dataset) < best_params["batch_size"]:
                logging.warning(f"Dataset for {engine_name} is too small ({len(dataset)} files). Skipping training for this engine.")
                continue

            train_size = int(0.9 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

            # ! KEY IMPROVEMENT for i7-3770: Smartly set num_workers
            num_workers = 0 # Default to 0 for Windows or if unsure
            if platform.system() == "Linux" or platform.system() == "Darwin":
                cpu_count = os.cpu_count()
                if cpu_count is not None:
                    # For an older i7, using fewer workers can prevent system lag
                    num_workers = min(4, cpu_count // 2)
            logging.info(f"Using {num_workers} workers for DataLoader.")

            train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True, collate_fn=custom_collate_fn, num_workers=num_workers, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=best_params["batch_size"], shuffle=False, collate_fn=custom_collate_fn, num_workers=num_workers, pin_memory=True)

            # --- Stage 3.3: Create a fresh Optimizer for this fine-tuning stage ---
            optimizer = torch.optim.AdamW(model.parameters(), lr=best_params["lr"])

            # --- Stage 3.4: Train on the CURRENT engine's data ---
            logging.info(f"\n--- [{engine_name}] Starting Phase 1: Training with mock rewards ---")
            train_ppo_with_accelerator(
                model, train_loader, val_loader, optimizer, codebert_pipeline,
                num_epochs=config["mock_epochs"],
                gradient_accumulation_steps=4,
                use_mock_reward=True, visualize_graphs=False,
                clip_epsilon=best_params["clip_epsilon"], curiosity_weight=best_params["curiosity_weight"],
                engine_name=engine_name, language=language
            )

            logging.info(f"\n--- [{engine_name}] Starting Phase 2: Fine-tuning with real rewards ---")
            train_ppo_with_accelerator(
                model, train_loader, val_loader, optimizer, codebert_pipeline,
                num_epochs=config["real_epochs"],
                gradient_accumulation_steps=4,
                use_mock_reward=False, visualize_graphs=True,
                clip_epsilon=best_params["clip_epsilon"], curiosity_weight=best_params["curiosity_weight"],
                engine_name=engine_name, language=language
            )

            # --- Stage 3.5: Save an intermediate checkpoint for this engine ---
            intermediate_save_path = os.path.join("model_checkpoints", f"model_after_{engine_name}_finetune")
            unwrapped_model = model.module if hasattr(model, 'module') else model
            unwrapped_model.llm.save_pretrained(intermediate_save_path)
            logging.info(f"Intermediate model fine-tuned on {engine_name} saved to '{intermediate_save_path}'.")

        except FileNotFoundError as e:
            logging.error(f"CRITICAL ERROR for {engine_name}: {e}. Skipping this engine and moving to the next.")
            continue # Continue to the next engine in the list
        
        finally:
            # --- Stage 3.6: Clean up memory before starting the next stage ---
            # ! This is CRUCIAL for stability in long training runs on your hardware.
            if 'train_loader' in locals():
                del train_loader, val_loader, dataset, train_dataset, val_dataset, optimizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logging.info(f"Cleaned up memory after training on {engine_name}.")


    # --- 4. Save the Final Model ---
    # This model has now been trained on all engines sequentially.
    final_model_save_path = "final_multi_engine_model"
    unwrapped_model = model.module if hasattr(model, 'module') else model
    unwrapped_model.llm.save_pretrained(final_model_save_path)
    logging.info(f"Final, sequentially-trained model saved to '{final_model_save_path}'.")

    # --- 5. Run Inference Examples with the Final Model ---
    run_inference_examples(model)

if __name__ == "__main__":
    main()