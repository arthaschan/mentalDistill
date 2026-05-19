#!/usr/bin/env python3
import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

try:
  from vllm import LLM, SamplingParams
  HAS_VLLM = True
except ImportError:
  HAS_VLLM = False
  LLM = None
  SamplingParams = None


HTML_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --paper: rgba(255, 251, 245, 0.84);
      --panel: rgba(255, 255, 255, 0.76);
      --ink: #1f2430;
      --muted: #5d6271;
      --accent: #8c1d18;
      --accent-soft: #c56a52;
      --deep: #12343b;
      --gold: #b48a3a;
      --line: rgba(18, 52, 59, 0.12);
      --shadow: 0 24px 60px rgba(31, 36, 48, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "PingFang SC", "Noto Sans SC", "Microsoft YaHei", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(180, 138, 58, 0.20), transparent 24%),
        radial-gradient(circle at 85% 12%, rgba(140, 29, 24, 0.18), transparent 22%),
        linear-gradient(145deg, #fcf7ef, var(--bg));
      min-height: 100vh;
    }}
    .shell {{ max-width: 1260px; margin: 0 auto; padding: 28px 20px 44px; }}
    .hero {{
      position: relative;
      overflow: hidden;
      margin-bottom: 22px;
      padding: 28px;
      border: 1px solid var(--line);
      border-radius: 30px;
      background:
        linear-gradient(140deg, rgba(18, 52, 59, 0.94), rgba(26, 34, 42, 0.9) 52%, rgba(140, 29, 24, 0.92));
      color: #f9f3ea;
      box-shadow: var(--shadow);
    }}
    .hero::after {{
      content: "";
      position: absolute;
      inset: auto -120px -140px auto;
      width: 360px;
      height: 360px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(255,255,255,0.18), rgba(255,255,255,0));
    }}
    .hero-grid {{
      position: relative;
      z-index: 1;
      display: grid;
      gap: 22px;
      grid-template-columns: minmax(0, 1.45fr) minmax(280px, 0.75fr);
      align-items: start;
    }}
    .eyebrow {{
      display: inline-flex;
      align-items: center;
      gap: 10px;
      padding: 8px 14px;
      border: 1px solid rgba(255,255,255,0.16);
      border-radius: 999px;
      background: rgba(255,255,255,0.08);
      letter-spacing: 0.14em;
      text-transform: uppercase;
      font-size: 12px;
    }}
    .hero h1 {{ font-size: clamp(30px, 4.3vw, 56px); margin: 16px 0 10px; line-height: 1.06; }}
    .hero .subtitle {{ max-width: 820px; font-size: 17px; line-height: 1.75; color: rgba(249,243,234,0.84); margin: 0; }}
    .thesis-en {{ margin-top: 12px; font-size: 14px; letter-spacing: 0.04em; color: rgba(249,243,234,0.72); }}
    .identity-card {{
      display: grid;
      gap: 14px;
      padding: 20px;
      border-radius: 24px;
      border: 1px solid rgba(255,255,255,0.14);
      background: rgba(255,255,255,0.08);
      backdrop-filter: blur(14px);
    }}
    .logo-wrap {{
      display: flex;
      align-items: center;
      gap: 14px;
      padding-bottom: 12px;
      border-bottom: 1px solid rgba(255,255,255,0.14);
    }}
    .logo-wrap img {{
      width: 78px;
      height: 78px;
      object-fit: contain;
      border-radius: 18px;
      background: rgba(255,255,255,0.92);
      padding: 10px;
    }}
    .logo-wrap strong {{ display: block; font-size: 18px; line-height: 1.35; }}
    .logo-wrap span {{ color: rgba(249,243,234,0.72); font-size: 13px; }}
    .info-grid {{ display: grid; gap: 12px; grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .info-item {{
      padding: 12px 14px;
      border-radius: 18px;
      background: rgba(255,255,255,0.08);
      border: 1px solid rgba(255,255,255,0.1);
    }}
    .info-item .k {{ font-size: 12px; color: rgba(249,243,234,0.66); margin-bottom: 4px; }}
    .info-item .v {{ font-size: 15px; font-weight: 700; }}
    .deployment {{
      padding: 12px 14px;
      border-radius: 18px;
      background: rgba(255,255,255,0.1);
      font-size: 13px;
      line-height: 1.6;
      color: rgba(249,243,234,0.78);
    }}
    .grid {{ display: grid; gap: 20px; grid-template-columns: 1.04fr 0.96fr; }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 28px;
      padding: 24px;
      backdrop-filter: blur(16px);
      box-shadow: var(--shadow);
    }}
    .section-title {{ display: flex; justify-content: space-between; gap: 12px; align-items: start; margin-bottom: 16px; }}
    .section-title h2 {{ margin: 0; font-size: 24px; line-height: 1.15; }}
    .section-title p {{ margin: 6px 0 0; color: var(--muted); font-size: 14px; line-height: 1.6; }}
    .tabs {{ display: flex; gap: 10px; margin-bottom: 16px; flex-wrap: wrap; }}
    .tab {{
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.66);
      border-radius: 999px;
      padding: 11px 16px;
      cursor: pointer;
      transition: 160ms ease;
    }}
    .tab.active {{ background: linear-gradient(135deg, var(--accent), var(--accent-soft)); color: #fff; border-color: transparent; }}
    label {{ display: block; font-size: 13px; margin: 14px 0 8px; color: var(--muted); }}
    textarea, input, select {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px 15px;
      font: inherit;
      color: inherit;
      background: rgba(255,255,255,0.9);
    }}
    textarea {{ min-height: 170px; resize: vertical; }}
    .options {{ min-height: 168px; }}
    .helper-grid {{ display: grid; gap: 10px; margin-top: 12px; grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .helper-chip {{
      padding: 12px 14px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.62);
      color: var(--muted);
      font-size: 13px;
      line-height: 1.55;
    }}
    .actions {{ display: flex; gap: 12px; align-items: center; margin-top: 18px; flex-wrap: wrap; }}
    button {{
      border: 0;
      border-radius: 999px;
      padding: 12px 20px;
      font: inherit;
      background: linear-gradient(135deg, var(--accent), var(--accent-soft));
      color: #fff;
      cursor: pointer;
      box-shadow: 0 12px 30px rgba(140, 29, 24, 0.18);
    }}
    .ghost {{ background: rgba(255,255,255,0.82); color: var(--ink); border: 1px solid var(--line); box-shadow: none; }}
    .status {{ font-size: 13px; color: var(--muted); }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      margin: 0;
      font-family: "JetBrains Mono", "SFMono-Regular", monospace;
      font-size: 14px;
      line-height: 1.7;
    }}
    .output-shell {{
      padding: 18px;
      border-radius: 24px;
      background: linear-gradient(180deg, rgba(18,52,59,0.98), rgba(35,43,51,0.96));
      color: #f6efe4;
    }}
    .answer-letter {{ font-size: 52px; line-height: 1; font-weight: 800; color: #f3c66d; margin-bottom: 14px; }}
    .answer-panel {{ min-height: 220px; }}
    .meta {{ display: grid; gap: 12px; grid-template-columns: repeat(2, minmax(0, 1fr)); margin-top: 16px; }}
    .pill {{ border: 1px solid rgba(255,255,255,0.1); border-radius: 18px; padding: 12px 14px; background: rgba(255,255,255,0.06); }}
    .pill strong {{ display: block; color: rgba(246,239,228,0.7); font-size: 12px; margin-bottom: 6px; }}
    .wide {{ grid-column: 1 / -1; }}
    .toolbar {{ display: flex; gap: 10px; align-items: end; flex-wrap: wrap; }}
    .toolbar .grow {{ flex: 1 1 320px; }}
    .toolbar button {{ white-space: nowrap; }}
    .mini {{ font-size: 12px; color: rgba(246,239,228,0.64); margin-top: 10px; line-height: 1.6; }}
    .footer-note {{ margin-top: 14px; color: var(--muted); font-size: 13px; line-height: 1.7; }}
    @media (max-width: 980px) {{
      .hero-grid, .grid, .info-grid, .helper-grid {{ grid-template-columns: 1fr; }}
      .shell {{ padding-inline: 16px; }}
      .hero, .card {{ padding: 20px; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="hero-grid">
        <div>
          <div class="eyebrow">Master Thesis Demo System</div>
          <h1>基于知识蒸馏的牙科选择题自动答题系统</h1>
          <p class="subtitle">该演示页面用于现场展示毕业论文对应的牙科智能答题系统，支持牙科问答题与五选一选择题两种交互模式，并直接连接当前最佳蒸馏模型。</p>
          <div class="thesis-en">A Knowledge Distillation-Based Automatic Dental Multiple-Choice Question Answering System</div>
        </div>
        <aside class="identity-card">
          <div class="logo-wrap">
            <img src="https://www.chuhai.edu.hk/_nuxt/logo_v_c.eba84de3.png" alt="香港珠海学院校徽">
            <div>
              <strong>香港珠海学院</strong>
              <span>Hong Kong Chu Hai College</span>
            </div>
          </div>
          <div class="info-grid">
            <div class="info-item">
              <div class="k">姓名</div>
              <div class="v">陈天元</div>
            </div>
            <div class="info-item">
              <div class="k">学号</div>
              <div class="v">256360231</div>
            </div>
            <div class="info-item">
              <div class="k">导师</div>
              <div class="v">熊体操</div>
            </div>
            <div class="info-item">
              <div class="k">系统模式</div>
              <div class="v">问答题 / 选择题</div>
            </div>
          </div>
          <div class="deployment">当前部署：{title}<br>本页适合课堂、答辩和论文演示场景，支持直接展示模型输出与候选选项结果。</div>
        </aside>
      </div>
    </section>
    <section class="grid">
      <div class="card">
        <div class="section-title">
          <div>
            <h2>交互输入区</h2>
            <p>在选择题模式下输入题干与 A-E 五个选项；在问答模式下输入牙科临床或科普问题。</p>
          </div>
        </div>
        <div class="tabs">
          <button class="tab active" data-mode="choice" type="button">选择题</button>
          <button class="tab" data-mode="qa" type="button">问答</button>
        </div>
        <div class="toolbar">
          <div class="grow">
            <label for="adapter-select">当前 Adapter / Checkpoint</label>
            <select id="adapter-select"></select>
          </div>
          <button id="reload-adapters" type="button" class="ghost">刷新列表</button>
        </div>
        <label for="question">问题</label>
        <textarea id="question" placeholder="输入题干或牙科问题"></textarea>
        <div id="choice-fields">
          <label for="options">选项</label>
          <textarea id="options" class="options" placeholder="A. ...\nB. ...\nC. ...\nD. ...\nE. ..."></textarea>
        </div>
        <div class="helper-grid">
          <div class="helper-chip">选择题建议：直接粘贴标准五选一题目，选项按 A、B、C、D、E 分行输入。</div>
          <div class="helper-chip">问答题建议：可要求模型按“诊断、依据、处理原则”结构作答，便于现场展示。</div>
        </div>
        <div class="actions">
          <button id="submit" type="button">生成结果</button>
          <button id="reset" type="button" class="ghost">清空</button>
          <span id="status" class="status">模型已加载后可直接提问</span>
        </div>
        <div class="footer-note">如果演示现场网络不稳定，页面会直接展示明确错误信息，避免出现难以解释的前端异常。</div>
      </div>
      <div class="card">
        <div class="section-title">
          <div>
            <h2>模型输出区</h2>
            <p>右侧展示选择题答案字母或问答题完整回复，适合投屏展示模型响应效果。</p>
          </div>
        </div>
        <div class="output-shell">
          <div class="answer-letter" id="answer-letter">-</div>
          <div class="answer-panel">
            <label>模型输出</label>
            <pre id="answer">等待输入...</pre>
          </div>
          <div class="meta">
            <div class="pill"><strong>模式</strong><div id="mode-view">choice</div></div>
            <div class="pill"><strong>推理后端</strong><div id="backend-view">-</div></div>
            <div class="pill wide"><strong>当前 Adapter</strong><div id="adapter-view">-</div></div>
            <div class="pill wide"><strong>基础模型</strong><div id="base-view">-</div></div>
            <div class="pill wide"><strong>候选数量</strong><div id="adapter-count">-</div></div>
          </div>
          <div class="mini">选择题模式下会自动抽取首个 A-E 字母作为结果高亮，便于答辩现场快速核对。</div>
        </div>
      </div>
    </section>
  </div>
  <script>
    const tabs = document.querySelectorAll('.tab');
    const choiceFields = document.getElementById('choice-fields');
    const questionBox = document.getElementById('question');
    const optionsBox = document.getElementById('options');
    const answerBox = document.getElementById('answer');
    const answerLetterBox = document.getElementById('answer-letter');
    const statusBox = document.getElementById('status');
    const modeView = document.getElementById('mode-view');
    const backendView = document.getElementById('backend-view');
    const adapterView = document.getElementById('adapter-view');
    const baseView = document.getElementById('base-view');
    const adapterCount = document.getElementById('adapter-count');
    const adapterSelect = document.getElementById('adapter-select');
    let mode = 'choice';

    function syncMode(nextMode) {{
      mode = nextMode;
      tabs.forEach(tab => tab.classList.toggle('active', tab.dataset.mode === nextMode));
      choiceFields.style.display = nextMode === 'choice' ? 'block' : 'none';
      modeView.textContent = nextMode;
      questionBox.placeholder = nextMode === 'choice' ? '输入题干' : '输入牙科问答问题';
    }}

    tabs.forEach(tab => tab.addEventListener('click', () => syncMode(tab.dataset.mode)));

    async function readResponsePayload(response) {{
      const contentType = (response.headers.get('content-type') || '').toLowerCase();
      if (contentType.includes('application/json')) {{
        return response.json();
      }}
      const text = await response.text();
      const compact = String(text || '').replace(/\\s+/g, ' ').trim();
      return {{
        error: compact || `HTTP ${{response.status}} ${{response.statusText}}`
      }};
    }}

    async function fetchState() {{
      const response = await fetch('/api/state');
      const payload = await readResponsePayload(response);
      if (!response.ok) throw new Error(payload.error || 'state error');
      backendView.textContent = payload.backend || '-';
      adapterView.textContent = payload.current_adapter_display || '(base only)';
      baseView.textContent = payload.base_model || '-';
      adapterCount.textContent = String((payload.adapters || []).length);
      adapterSelect.innerHTML = '';
      (payload.adapters || []).forEach(item => {{
        const option = document.createElement('option');
        option.value = item.value;
        option.textContent = item.label;
        option.selected = item.selected;
        adapterSelect.appendChild(option);
      }});
    }}

    async function changeAdapter() {{
      statusBox.textContent = '切换模型中...';
      answerBox.textContent = '正在重新加载 adapter，请稍候。';
      answerLetterBox.textContent = '-';
      const response = await fetch('/api/select_adapter', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ adapter: adapterSelect.value }})
      }});
      const payload = await readResponsePayload(response);
      if (!response.ok) throw new Error(payload.error || '切换失败');
      await fetchState();
      answerBox.textContent = 'adapter 切换完成。';
      statusBox.textContent = '已切换';
    }}

    document.getElementById('reload-adapters').addEventListener('click', async () => {{
      statusBox.textContent = '刷新中...';
      try {{
        await fetchState();
        statusBox.textContent = '列表已刷新';
      }} catch (error) {{
        statusBox.textContent = '刷新失败';
        answerBox.textContent = String(error);
      }}
    }});

    adapterSelect.addEventListener('change', async () => {{
      try {{
        await changeAdapter();
      }} catch (error) {{
        answerBox.textContent = String(error);
        statusBox.textContent = '切换失败';
      }}
    }});

    document.getElementById('submit').addEventListener('click', async () => {{
      statusBox.textContent = '生成中...';
      answerBox.textContent = '正在调用模型，请稍候。';
      answerLetterBox.textContent = '-';
      try {{
        const response = await fetch('/api/generate', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{
            mode,
            question: questionBox.value,
            options: optionsBox.value
          }})
        }});
        const payload = await readResponsePayload(response);
        if (!response.ok) throw new Error(payload.error || '请求失败');
        answerBox.textContent = payload.answer || '(空响应)';
        answerLetterBox.textContent = payload.answer_letter || '-';
        backendView.textContent = payload.backend || '-';
        adapterView.textContent = payload.current_adapter_display || '(base only)';
        statusBox.textContent = '完成';
      }} catch (error) {{
        answerBox.textContent = String(error);
        answerLetterBox.textContent = '-';
        statusBox.textContent = '失败';
      }}
    }});

    document.getElementById('reset').addEventListener('click', () => {{
      questionBox.value = '';
      optionsBox.value = '';
      answerBox.textContent = '等待输入...';
      answerLetterBox.textContent = '-';
      statusBox.textContent = '模型已加载后可直接提问';
    }});

    fetchState().catch(error => {{
      answerBox.textContent = String(error);
      statusBox.textContent = '初始化失败';
    }});
  </script>
</body>
</html>
"""


def is_adapter_only_model(model_path):
    return (
        model_path
        and os.path.isdir(model_path)
        and os.path.exists(os.path.join(model_path, "adapter_config.json"))
        and not os.path.exists(os.path.join(model_path, "config.json"))
    )


def build_choice_prompt(question, options):
    return (
        "<|im_start|>system\n"
        "你是一名专业的牙科医生，只需输出一个字母（A、B、C、D、E）作为结果，不要附带任何解释或空格。\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"问题：{question}\n"
        f"选项：\n{options}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_qa_prompt(question):
    return (
        "<|im_start|>system\n"
        "你是一名专业的牙科医生，擅长解答各类口腔医学问题，回答需专业、准确、通俗易懂，符合中文表达习惯。\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{question}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def extract_answer_char(text):
    for char in str(text or "").strip().upper():
        if char in {"A", "B", "C", "D", "E"}:
            return char
    return ""


def discover_adapters(adapter_root):
    if not adapter_root:
        return []
    root = Path(adapter_root)
    if not root.exists():
        return []

    candidates = []
    seen = set()
    for config_path in root.rglob("adapter_config.json"):
        adapter_dir = config_path.parent.resolve()
        key = str(adapter_dir)
        if key in seen:
            continue
        seen.add(key)
        try:
            label = str(adapter_dir.relative_to(root.resolve()))
        except ValueError:
            label = adapter_dir.name
        candidates.append({"value": key, "label": label})

    candidates.sort(key=lambda item: item["label"])
    return candidates


class InferenceBackend:
    def __init__(self, base_model, adapter_dir, adapter_root, max_new_tokens, gpu_memory_utilization):
        self.base_model = base_model
        self.adapter_dir = adapter_dir or ""
        self.adapter_root = adapter_root or ""
        self.max_new_tokens = max_new_tokens
        self.gpu_memory_utilization = gpu_memory_utilization
        self.backend_name = "transformers"
        self._llm = None
        self._sampling_params = None
        self._model = None
        self._tokenizer = None
        self._device = None
        self._lock = threading.RLock()
        self._adapter_choices = []
        self.refresh_adapters()
        self.load(self.adapter_dir)

    def refresh_adapters(self):
        with self._lock:
            discovered = discover_adapters(self.adapter_root)
            current = self.adapter_dir or ""
            if current and not any(item["value"] == current for item in discovered):
                current_path = Path(current)
                discovered.insert(0, {"value": str(current_path), "label": current_path.name})
            discovered.insert(0, {"value": "", "label": "(base model only)"})
            deduped = []
            seen = set()
            for item in discovered:
                if item["value"] in seen:
                    continue
                seen.add(item["value"])
                deduped.append(item)
            self._adapter_choices = deduped

    def load(self, adapter_dir):
        with self._lock:
            self.adapter_dir = adapter_dir or ""
            self._llm = None
            self._sampling_params = None
            self._model = None
            self._tokenizer = None
            self._device = None
            self.backend_name = "transformers"
            self._build()

    def _build(self):
        if HAS_VLLM and not is_adapter_only_model(self.adapter_dir):
            try:
                model_path = self.adapter_dir or self.base_model
                self._llm = LLM(
                    model=model_path,
                    tokenizer=self.base_model,
                    tensor_parallel_size=1,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    dtype=torch.bfloat16,
                )
                self._sampling_params = SamplingParams(
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=self.max_new_tokens,
                    stop=["<|endoftext|>", "</s>"]
                )
                self.backend_name = "vllm"
                return
            except Exception as error:
                print(f"[WARN] vLLM load failed, falling back to transformers: {error}")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        if self.adapter_dir:
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(self._model, self.adapter_dir)
        self._model.eval()
        self._device = next(self._model.parameters()).device

    def generate(self, prompt):
        with self._lock:
            if self.backend_name == "vllm":
                outputs = self._llm.generate([prompt], self._sampling_params)
                return outputs[0].outputs[0].text.strip()

            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
                )
            return self._tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

    def state(self):
        with self._lock:
            adapters = []
            for item in self._adapter_choices:
                adapters.append({
                    "value": item["value"],
                    "label": item["label"],
                    "selected": item["value"] == (self.adapter_dir or ""),
                })
            return {
                "base_model": self.base_model,
                "current_adapter": self.adapter_dir,
                "current_adapter_display": self.adapter_dir or "(base model only)",
                "backend": self.backend_name,
                "adapter_root": self.adapter_root,
                "adapters": adapters,
            }


def make_handler(backend, title):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                body = HTML_PAGE.format(title=title).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/api/state":
                backend.refresh_adapters()
                body = json.dumps(backend.state(), ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            else:
                self.send_error(404)
                return

        def do_POST(self):
            if self.path not in {"/api/generate", "/api/select_adapter"}:
                self.send_error(404)
                return

            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            try:
                payload = json.loads(raw_body.decode("utf-8"))
                if self.path == "/api/select_adapter":
                    requested_adapter = str(payload.get("adapter") or "")
                    backend.refresh_adapters()
                    valid_values = {item["value"] for item in backend.state()["adapters"]}
                    if requested_adapter not in valid_values:
                        raise ValueError("unknown adapter selection")
                    backend.load(requested_adapter)
                    response = backend.state()
                else:
                    mode = str(payload.get("mode") or "choice").strip().lower()
                    question = str(payload.get("question") or "").strip()
                    options = str(payload.get("options") or "").strip()

                    if not question:
                        raise ValueError("question is required")
                    if mode == "choice" and not options:
                        raise ValueError("options are required in choice mode")

                    prompt = build_qa_prompt(question) if mode == "qa" else build_choice_prompt(question, options)
                    answer = backend.generate(prompt)
                    response = {
                        "answer": answer,
                        "answer_letter": extract_answer_char(answer) if mode == "choice" else "",
                        "mode": mode,
                        "backend": backend.backend_name,
                        "current_adapter_display": backend.state()["current_adapter_display"],
                    }
                body = json.dumps(response, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
            except Exception as error:
                body = json.dumps({"error": str(error)}, ensure_ascii=False).encode("utf-8")
                self.send_response(400)

            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            return

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Serve a local dental QA/MCQ web app.")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_dir", default="")
    parser.add_argument("--adapter_root", default="")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--title", default="Dental Model App")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    args = parser.parse_args()

    backend = InferenceBackend(
        base_model=args.base_model,
        adapter_dir=args.adapter_dir,
      adapter_root=args.adapter_root,
        max_new_tokens=args.max_new_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    print(
      f"[INFO] backend={backend.backend_name} base_model={args.base_model} "
      f"adapter_dir={args.adapter_dir or '(none)'} adapter_root={args.adapter_root or '(none)'}"
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(backend, args.title))
    print(f"[INFO] serving http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()