#!/usr/bin/env python3
"""人机交互测试界面：从测试集加载选择题，用户作答后即时反馈正误。

用法:
    python shared/quiz_app.py --test_data data/test.jsonl [--port 7870] [--title "Quiz"]

功能:
    - 从 JSONL 文件加载选择题（Question, Options, Answer 字段）
    - 网页显示题目和选项，用户选择答案
    - 即时反馈正误 + 正确答案
    - 统计当前答题得分
    - 支持顺序/随机模式
"""
import argparse
import json
import os
import random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HTML_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      --bg: #f3efe4;
      --panel: rgba(255,255,255,0.85);
      --ink: #1f2a1f;
      --accent: #0f766e;
      --correct: #16a34a;
      --wrong: #dc2626;
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
    .shell {{ max-width: 900px; margin: 0 auto; padding: 32px 20px 60px; }}
    .hero {{ margin-bottom: 20px; }}
    .eyebrow {{ letter-spacing: 0.16em; text-transform: uppercase; font-size: 12px; opacity: 0.68; }}
    h1 {{ font-size: clamp(24px, 3.5vw, 44px); margin: 8px 0; line-height: 1.1; }}
    .score-bar {{
      display: flex; gap: 16px; align-items: center; flex-wrap: wrap;
      margin-bottom: 18px; font-size: 15px;
    }}
    .score-bar .pill {{
      border: 1px solid var(--line); border-radius: 999px; padding: 8px 16px;
      background: var(--panel);
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 24px;
      backdrop-filter: blur(14px);
      box-shadow: 0 20px 50px rgba(31,42,31,0.08);
      margin-bottom: 18px;
    }}
    .q-number {{ font-size: 13px; opacity: 0.6; margin-bottom: 8px; }}
    .q-text {{ font-size: 17px; line-height: 1.7; margin-bottom: 16px; white-space: pre-wrap; }}
    .option-btn {{
      display: block; width: 100%; text-align: left;
      border: 2px solid var(--line); border-radius: 14px;
      padding: 14px 18px; margin-bottom: 10px;
      font: inherit; font-size: 15px; line-height: 1.6;
      background: rgba(255,255,255,0.7); cursor: pointer;
      transition: 180ms ease;
    }}
    .option-btn:hover:not(.locked) {{ border-color: var(--accent); background: rgba(15,118,110,0.06); }}
    .option-btn.selected {{ border-color: var(--accent); background: rgba(15,118,110,0.10); font-weight: 600; }}
    .option-btn.correct {{ border-color: var(--correct); background: rgba(22,163,74,0.12); }}
    .option-btn.wrong {{ border-color: var(--wrong); background: rgba(220,38,38,0.10); }}
    .option-btn.locked {{ cursor: default; opacity: 0.85; }}
    .feedback {{
      margin-top: 16px; padding: 14px 18px; border-radius: 14px;
      font-size: 16px; font-weight: 600; display: none;
    }}
    .feedback.show {{ display: block; }}
    .feedback.correct {{ background: rgba(22,163,74,0.12); color: var(--correct); }}
    .feedback.wrong {{ background: rgba(220,38,38,0.10); color: var(--wrong); }}
    .actions {{ display: flex; gap: 12px; margin-top: 18px; flex-wrap: wrap; }}
    button {{
      border: 0; border-radius: 999px; padding: 12px 22px;
      font: inherit; font-size: 15px;
      background: linear-gradient(135deg, var(--accent), #155e75);
      color: #fff; cursor: pointer; transition: 120ms ease;
    }}
    button:hover {{ filter: brightness(1.08); }}
    .ghost {{
      background: rgba(255,255,255,0.7); color: var(--ink);
      border: 1px solid var(--line);
    }}
    .nav-btns {{ display: flex; gap: 10px; margin-top: 16px; }}
    .hidden {{ display: none; }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="eyebrow">Dental Quiz Interface</div>
      <h1>{title}</h1>
    </section>
    <div class="score-bar">
      <span class="pill" id="score-display">得分: 0 / 0</span>
      <span class="pill" id="acc-display">正确率: --</span>
      <span class="pill" id="progress-display">进度: 0 / 0</span>
      <button class="ghost" onclick="resetQuiz()" style="font-size:13px; padding:8px 14px;">重新开始</button>
      <button class="ghost" onclick="toggleMode()" id="mode-btn" style="font-size:13px; padding:8px 14px;">切换为随机模式</button>
    </div>
    <div class="card" id="quiz-card">
      <div class="q-number" id="q-number"></div>
      <div class="q-text" id="q-text">加载中...</div>
      <div id="options-container"></div>
      <div class="feedback" id="feedback"></div>
      <div class="actions">
        <button id="submit-btn" onclick="submitAnswer()" disabled>提交答案</button>
        <button id="next-btn" class="ghost hidden" onclick="nextQuestion()">下一题</button>
      </div>
    </div>
  </div>
  <script>
    let questions = [];
    let currentIdx = 0;
    let score = 0;
    let answered = 0;
    let selectedOption = null;
    let isLocked = false;
    let randomMode = false;
    let questionOrder = [];

    async function loadQuestions() {{
      const resp = await fetch('/api/questions');
      questions = await resp.json();
      questionOrder = questions.map((_, i) => i);
      showQuestion();
      updateProgress();
    }}

    function showQuestion() {{
      if (currentIdx >= questionOrder.length) {{
        document.getElementById('q-text').textContent = '所有题目已完成！';
        document.getElementById('options-container').innerHTML = '';
        document.getElementById('feedback').className = 'feedback';
        document.getElementById('submit-btn').classList.add('hidden');
        document.getElementById('next-btn').classList.add('hidden');
        return;
      }}
      const q = questions[questionOrder[currentIdx]];
      document.getElementById('q-number').textContent =
        '第 ' + (currentIdx + 1) + ' / ' + questionOrder.length + ' 题（原始编号 #' + (questionOrder[currentIdx] + 1) + '）';
      document.getElementById('q-text').textContent = q.question;

      const container = document.getElementById('options-container');
      container.innerHTML = '';
      q.options.forEach((opt) => {{
        const btn = document.createElement('button');
        btn.className = 'option-btn';
        btn.textContent = opt.label + '. ' + opt.text;
        btn.dataset.letter = opt.label;
        btn.onclick = () => selectOption(btn, opt.label);
        container.appendChild(btn);
      }});

      selectedOption = null;
      isLocked = false;
      document.getElementById('feedback').className = 'feedback';
      document.getElementById('feedback').textContent = '';
      document.getElementById('submit-btn').disabled = true;
      document.getElementById('submit-btn').classList.remove('hidden');
      document.getElementById('next-btn').classList.add('hidden');
    }}

    function selectOption(btn, letter) {{
      if (isLocked) return;
      document.querySelectorAll('.option-btn').forEach(b => b.classList.remove('selected'));
      btn.classList.add('selected');
      selectedOption = letter;
      document.getElementById('submit-btn').disabled = false;
    }}

    function submitAnswer() {{
      if (!selectedOption || isLocked) return;
      isLocked = true;
      const q = questions[questionOrder[currentIdx]];
      const correct = selectedOption === q.answer;

      document.querySelectorAll('.option-btn').forEach(btn => {{
        btn.classList.add('locked');
        if (btn.dataset.letter === q.answer) btn.classList.add('correct');
        if (btn.dataset.letter === selectedOption && !correct) btn.classList.add('wrong');
      }});

      const fb = document.getElementById('feedback');
      if (correct) {{
        fb.textContent = '✓ 回答正确！';
        fb.className = 'feedback show correct';
        score++;
      }} else {{
        fb.textContent = '✗ 回答错误。正确答案是 ' + q.answer;
        fb.className = 'feedback show wrong';
      }}
      answered++;
      updateProgress();
      document.getElementById('submit-btn').classList.add('hidden');
      document.getElementById('next-btn').classList.remove('hidden');
    }}

    function nextQuestion() {{
      currentIdx++;
      showQuestion();
    }}

    function updateProgress() {{
      document.getElementById('score-display').textContent = '得分: ' + score + ' / ' + answered;
      const acc = answered > 0 ? (score / answered * 100).toFixed(1) + '%' : '--';
      document.getElementById('acc-display').textContent = '正确率: ' + acc;
      document.getElementById('progress-display').textContent = '进度: ' + Math.min(currentIdx + 1, questionOrder.length) + ' / ' + questionOrder.length;
    }}

    function resetQuiz() {{
      currentIdx = 0;
      score = 0;
      answered = 0;
      if (randomMode) {{
        questionOrder = questions.map((_, i) => i);
        for (let i = questionOrder.length - 1; i > 0; i--) {{
          const j = Math.floor(Math.random() * (i + 1));
          [questionOrder[i], questionOrder[j]] = [questionOrder[j], questionOrder[i]];
        }}
      }} else {{
        questionOrder = questions.map((_, i) => i);
      }}
      showQuestion();
      updateProgress();
    }}

    function toggleMode() {{
      randomMode = !randomMode;
      document.getElementById('mode-btn').textContent = randomMode ? '切换为顺序模式' : '切换为随机模式';
      resetQuiz();
    }}

    loadQuestions();
  </script>
</body>
</html>
"""


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_options(options_str):
    """Parse options string like 'A. xxx\\nB. yyy\\n...' into structured list."""
    parsed = []
    for line in str(options_str).strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if len(line) >= 2 and line[0].upper() in "ABCDE" and line[1] in ".:：、 ":
            label = line[0].upper()
            text = line[2:].strip()
            parsed.append({"label": label, "text": text})
        elif parsed:
            # continuation line
            parsed[-1]["text"] += " " + line
    return parsed


def prepare_questions(rows):
    questions = []
    for row in rows:
        q_text = row.get("Question", "")
        opts_raw = row.get("Options", "")
        answer = row.get("Answer", "").strip().upper()
        if not q_text or not answer:
            continue
        opts = parse_options(opts_raw)
        if not opts:
            continue
        questions.append({
            "question": q_text,
            "options": opts,
            "answer": answer[0] if answer else "",
        })
    return questions


class QuizHandler(BaseHTTPRequestHandler):
    questions = []
    title = "Quiz"

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "":
            html = HTML_PAGE.format(title=self.title)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
        elif parsed.path == "/api/questions":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(self.questions, ensure_ascii=False).encode("utf-8"))
        else:
            self.send_error(404)

    def do_POST(self):
        self.send_error(405)


def main():
    parser = argparse.ArgumentParser(description="人机交互测试界面")
    parser.add_argument("--test_data", required=True, help="测试集 JSONL 文件路径")
    parser.add_argument("--host", default="0.0.0.0", help="绑定地址")
    parser.add_argument("--port", type=int, default=7870, help="端口号")
    parser.add_argument("--title", default="医学选择题测试", help="页面标题")
    args = parser.parse_args()

    test_path = Path(args.test_data)
    if not test_path.exists():
        print(f"错误: 测试数据文件不存在: {test_path}")
        return

    rows = load_jsonl(str(test_path))
    questions = prepare_questions(rows)
    print(f"已加载 {len(questions)} 道题目")

    QuizHandler.questions = questions
    QuizHandler.title = args.title

    server = ThreadingHTTPServer((args.host, args.port), QuizHandler)
    print(f"人机交互测试界面已启动: http://{args.host}:{args.port}")
    print("按 Ctrl+C 停止")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
        server.server_close()


if __name__ == "__main__":
    main()
