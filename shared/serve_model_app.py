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
      --bg: #f3efe4;
      --panel: rgba(255,255,255,0.82);
      --ink: #1f2a1f;
      --accent: #0f766e;
      --accent-2: #b45309;
      --line: rgba(31,42,31,0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(180,83,9,0.18), transparent 28%),
        radial-gradient(circle at right 20%, rgba(15,118,110,0.18), transparent 24%),
        linear-gradient(135deg, #f8f5ea, var(--bg));
      min-height: 100vh;
    }}
    .shell {{ max-width: 1080px; margin: 0 auto; padding: 32px 20px 48px; }}
    .hero {{ margin-bottom: 20px; }}
    .eyebrow {{ letter-spacing: 0.16em; text-transform: uppercase; font-size: 12px; opacity: 0.68; }}
    h1 {{ font-size: clamp(28px, 4vw, 52px); margin: 10px 0 8px; line-height: 1.04; }}
    .hero p {{ max-width: 760px; font-size: 16px; line-height: 1.7; margin: 0; }}
    .grid {{ display: grid; gap: 18px; grid-template-columns: 1.08fr 0.92fr; }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 20px;
      backdrop-filter: blur(14px);
      box-shadow: 0 20px 50px rgba(31,42,31,0.08);
    }}
    .tabs {{ display: flex; gap: 10px; margin-bottom: 14px; flex-wrap: wrap; }}
    .tab {{
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.56);
      border-radius: 999px;
      padding: 10px 14px;
      cursor: pointer;
      transition: 160ms ease;
    }}
    .tab.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
    label {{ display: block; font-size: 13px; margin: 14px 0 8px; opacity: 0.78; }}
    textarea, input, select {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px 15px;
      font: inherit;
      color: inherit;
      background: rgba(255,255,255,0.84);
    }}
    textarea {{ min-height: 160px; resize: vertical; }}
    .options {{ min-height: 148px; }}
    .actions {{ display: flex; gap: 12px; align-items: center; margin-top: 18px; flex-wrap: wrap; }}
    button {{
      border: 0;
      border-radius: 999px;
      padding: 12px 18px;
      font: inherit;
      background: linear-gradient(135deg, var(--accent), #155e75);
      color: #fff;
      cursor: pointer;
    }}
    .ghost {{ background: rgba(255,255,255,0.7); color: var(--ink); border: 1px solid var(--line); }}
    .status {{ font-size: 13px; opacity: 0.74; }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      margin: 0;
      font-family: "JetBrains Mono", "SFMono-Regular", monospace;
      font-size: 14px;
      line-height: 1.7;
    }}
    .meta {{ display: grid; gap: 12px; grid-template-columns: repeat(2, minmax(0, 1fr)); margin-top: 16px; }}
    .pill {{ border: 1px solid var(--line); border-radius: 18px; padding: 12px 14px; background: rgba(255,255,255,0.54); }}
    .wide {{ grid-column: 1 / -1; }}
    .toolbar {{ display: flex; gap: 10px; align-items: end; flex-wrap: wrap; }}
    .toolbar .grow {{ flex: 1 1 320px; }}
    .toolbar button {{ white-space: nowrap; }}
    .answer-letter {{ font-size: 40px; line-height: 1; font-weight: 800; color: var(--accent-2); margin-bottom: 12px; }}
    .mini {{ font-size: 12px; opacity: 0.72; margin-top: 8px; }}
    @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="eyebrow">Dental Distillation Workspace</div>
      <h1>{title}</h1>
      <p>这个页面用于快速验证当前实验目录产出的学生模型。支持牙科问答与口腔选择题推理，默认走本地模型，不依赖额外前端工程。</p>
    </section>
    <section class="grid">
      <div class="card">
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
        <div class="actions">
          <button id="submit" type="button">生成结果</button>
          <button id="reset" type="button" class="ghost">清空</button>
          <span id="status" class="status">模型已加载后可直接提问</span>
        </div>
      </div>
      <div class="card">
        <div class="answer-letter" id="answer-letter">-</div>
        <label>模型输出</label>
        <pre id="answer">等待输入...</pre>
        <div class="meta">
          <div class="pill"><strong>模式</strong><div id="mode-view">choice</div></div>
          <div class="pill"><strong>推理后端</strong><div id="backend-view">-</div></div>
          <div class="pill wide"><strong>当前 Adapter</strong><div id="adapter-view">-</div></div>
          <div class="pill wide"><strong>基础模型</strong><div id="base-view">-</div></div>
          <div class="pill wide"><strong>候选数量</strong><div id="adapter-count">-</div></div>
        </div>
        <div class="mini">选择题模式下会额外抽取首个 A-E 字母，便于快速核对预测。</div>
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

    async function fetchState() {{
      const response = await fetch('/api/state');
      const payload = await response.json();
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
      const payload = await response.json();
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
        const payload = await response.json();
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