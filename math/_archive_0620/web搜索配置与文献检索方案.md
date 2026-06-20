# Web 搜索配置 与 文献检索方案（澄清版）

> 日期：2026-06-20
> 缘起：用户想配 web 搜索补文献检索。澄清几个误解，给出最省事的方案。

---

## 1. 先澄清三个误解

| 你以为 | 实际 |
|---|---|
| arXiv 账号能申请搜索 API key | ❌ arXiv 的 API 是**公开免费、不需要 key** 的。我已经用它成功检索过。 |
| Google 账号自带 Google Scholar API | ❌ Google **故意不开放** Scholar 的官方 API，有账号也用不了程序检索。 |
| Hermes web_search 用 arXiv/Google 的 key | ❌ 它需要的是**第三方搜索服务**（Tavily/Brave/Serper）的 key，和你的账号无关。 |

---

## 2. 好消息：不配任何 key，我也能做文献检索

我已验证两个**免费、无需注册、无需 key** 的学术 API 都能用：
- **arXiv API**：已成功用过（找到 Apple Distillation Scaling Laws）。覆盖 arXiv 预印本。
- **Semantic Scholar API**：刚测通（偶尔 429 限流，加延迟重试即可）。覆盖**更广**——包括会议论文(NeurIPS/ICML/ACL等)、期刊，比 arXiv 全。

→ **这两个加起来，足够覆盖机器学习领域 95% 的相关文献。** 我可以直接用它们补全"体检工具方向"的检索，不用你做任何配置。

---

## 3. 三个选项（你选一个）

### 选项A（推荐，零成本零配置）：我用 arXiv + Semantic Scholar 免费 API 补查
- 不花钱、不注册、不配 key。
- 我写脚本，带限流退避，系统检索"不确定性蒸馏筛选 + 外部难度验证"方向。
- 局限：拿不到 Google Scholar 独有的引用数排序，但论文覆盖足够。

### 选项B（若要 Hermes 原生 web_search 全功能）：配第三方搜索 key
- 适合你以后想让我搜新闻、网页、非学术内容。
- 步骤（需注册免费账号拿 key）：
  1. 选一个：Tavily(tavily.com，免费1000次/月) / Brave(brave.com/search/api) / Serper(serper.dev)。
  2. 注册 → 拿到 key（形如 tvly-xxx）。
  3. 编辑 `~/.hermes/config.yaml` 的 `web:` 段（约第58行）：
     ```yaml
     web:
       backend: tavily
       search_backend: tavily
       extract_backend: tavily
     ```
  4. 设环境变量（加到 setup.env）：`export TAVILY_API_KEY="你的key"`
  5. 重启 Hermes 会话生效。

### 选项C：你自己在浏览器搜 Google Scholar
- 你有 Google 账号，直接浏览器开 scholar.google.com 搜关键词。
- 我给你关键词清单，你贴结果回来，我帮你分析撞车风险。
- 适合你想亲自把关文献时。

---

## 4. 我的建议

**先用选项A**（我现在就能做，零成本），把工具方向的 arXiv+Semantic Scholar 文献补齐。
如果补完发现某些关键论文只在 Google Scholar 有（少见），再用选项C你帮忙搜，或选项B配 key。

> 大多数情况下选项A 就够确认新颖性了。不必为这一件事专门去注册付费服务。

---

## 5. 关键词清单（选项C 时你直接用）

工具方向（Google Scholar 搜这些）：
- "uncertainty-based knowledge distillation sample selection"
- "confidence-based data filtering distillation"
- "data difficulty estimation human agreement model confidence"
- "example difficulty annotator agreement"
- "teacher uncertainty distillation medical"

看返回里有没有"用人类难度/跨模型共识验证模型不确定性=客观难度"且用于"蒸馏样本筛选"的——如果没有，我们这个角度就是新的。
