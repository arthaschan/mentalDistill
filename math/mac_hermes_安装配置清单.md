# Mac Air M4 上安装 Hermes + 配置灵眸(lmuai) 照做清单

目标：在 MacBook Air M4 上装好 Hermes Agent，并接上你在 H100 用的同一个"灵眸"
(lmuai) 大模型接口，跑起来和 H100 上一样。

参照的 H100 现状：Hermes v0.17.0，灵眸走 custom provider + OpenAI 兼容接口。

---

## 名词大白话

- 灵眸 / lmuai：一个 OpenAI 兼容的模型 API 服务(接口地址 https://api.lmuai.com/v1)。
  Hermes 里没有它的内置按钮，所以用"自定义 provider"(custom)这条通用通道接：
  只要给三样东西——接口地址(base_url)、密钥(api_key)、模型名(default)——就能连。
- config.yaml：Hermes 的主配置文件(设置放这)。
- .env：放密钥的文件。你 H100 上灵眸的 key 是直接写在 config.yaml 里的，不在 .env，
  所以下面 Mac 也照这个方式来，最省事。

---

## 第 0 步：先从 H100 取出灵眸真实密钥(1 条命令)

在 H100 这台机器上跑(不要贴到 Mac)，把打印出来的完整 key 记下来，等下填进 Mac：

```bash
grep '^  api_key:' ~/.hermes/config.yaml | head -1
```

输出形如 `  api_key: sk-b0a............adb4`，`sk-` 开头那一整串就是你要的密钥。
(注意：我这边工具输出会自动把密钥打码成 sk-b0a...adb4，所以必须你自己在 H100 上跑
这条命令拿明文。)

顺便记下当前默认模型名，Mac 上保持一致：

```bash
grep -E '^  default:|^  base_url:|^  reasoning_effort:' ~/.hermes/config.yaml
```

H100 现在是：
- default: claude-opus-4-8
- base_url: https://api.lmuai.com/v1
- reasoning_effort: high

---

## 第 1 步：在 Mac 上装 Hermes

打开 Mac 的"终端"(Terminal.app)，粘贴：

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

装完后，让 hermes 命令在当前终端立即可用(或重开一个终端窗口)：

```bash
source ~/.zshrc 2>/dev/null; source ~/.bash_profile 2>/dev/null
hermes --version
```

看到版本号(如 Hermes Agent v0.17.x)即安装成功。

前置依赖：Mac 需要有较新的 Python(3.10+)。M4 的 macOS 一般自带，装脚本也会处理。
若 --version 报找不到命令，重开终端窗口再试(PATH 需要刷新)。

---

## 第 2 步：配置灵眸(两种方式，选一种)

### 方式 A(推荐，最快)：直接命令行写入四个值

在 Mac 终端依次跑(把 <你的KEY> 换成第 0 步取到的真实密钥)：

```bash
hermes config set model.provider custom
hermes config set model.base_url https://api.lmuai.com/v1
hermes config set model.default claude-opus-4-8
hermes config set model.api_key <你的KEY>
hermes config set model.reasoning_effort high
```

### 方式 B：手动编辑配置文件

```bash
hermes config edit
```

把最顶部的 model 段改成(注意缩进是 2 个空格)：

```yaml
model:
  default: claude-opus-4-8
  provider: custom
  reasoning_effort: high
  base_url: https://api.lmuai.com/v1
  api_key: sk-你的完整KEY
```

存盘退出。

---

## 第 3 步：验证连通

```bash
hermes doctor
hermes chat -q "用一句话确认你是谁、用的什么模型"
```

- doctor 检查配置和依赖是否 OK。
- chat -q 发一句测试，如果能正常回话，说明灵眸接通了。

若报鉴权/401 错误 → key 没填对或有多余空格，回第 2 步重设 model.api_key。
若报连接超时 → 确认 Mac 能访问外网、base_url 没打错。

---

## 第 4 步(可选)：把 H100 的常用设置也搬过来

H100 上灵眸给了 30 分钟超时(慢请求不容易断)。Mac 上如果也想要，加：

```bash
hermes config set providers.lumai.request_timeout_seconds 1800
```

其它个性化(皮肤、工具集、语音等)可日后用 `hermes setup` 逐项配，不影响灵眸接通。

---

## 常见坑

1. 密钥打码：任何工具/日志里看到 sk-b0a...adb4 这种带省略号的都是打过码的，
   不能直接用。真实明文只在第 0 步 H100 那条 grep 命令里能拿到。
2. 缩进：手改 config.yaml 时 model 段每级必须 2 空格，用 Tab 会报错。
3. 配置不生效：改完 config 要开新的 hermes 会话(退出重进),旧会话不会热加载。
4. 找配置文件路径：`hermes config path`(Mac 上一般是 ~/.hermes/config.yaml)。
5. 别把 H100 整个 config.yaml 直接拷到 Mac：里面有 vllm-local(localhost:8000 的
   本地 Qwen3-32B)之类只在 H100 成立的东西，Mac 上没有那服务。只搬 model 段即可。

---

## 一分钟极简版(会命令行的话)

```bash
# Mac 上
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
hermes config set model.provider custom
hermes config set model.base_url https://api.lmuai.com/v1
hermes config set model.default claude-opus-4-8
hermes config set model.api_key <从H100取到的KEY>
hermes chat -q "hello"
```
