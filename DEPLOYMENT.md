# 部署说明

本文档说明如何在本机、局域网服务器或 Streamlit Community Cloud 上部署本项目。

## 1. 本机部署

### 环境要求

- Python 3.10 或更高版本
- Windows / macOS / Linux 均可
- 可访问互联网，用于抓取行情、新闻和公告

### 安装步骤

进入项目目录：

```powershell
cd market-agent-demo
```

创建虚拟环境，任选一种方式。

方式 A：使用 Anaconda Python：

```powershell
D:\Anaconda\python.exe -m pip install -r requirements.txt
```

方式 B：使用系统 Python：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

启动：

```powershell
python -m streamlit run app.py
```

访问：

```text
http://localhost:8501
```

## 2. 局域网部署

如果希望同一局域网内其他设备访问：

```powershell
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

然后在其他设备浏览器打开：

```text
http://服务器IP:8501
```

注意检查防火墙是否放行 `8501` 端口。

## 3. Linux 服务器部署

```bash
cd market-agent-demo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

后台运行可用 `nohup`：

```bash
nohup .venv/bin/python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501 > streamlit.out.log 2> streamlit.err.log &
```

## 4. API 模型配置

本项目不强制要求 API Key。没有 API Key 时：

- 报告生成使用本地规则智能体。
- 专家意见自动回退到本地规则专家。

如需调用 DeepSeek 或其他 OpenAI-compatible API，在项目根目录创建 `.env`：

```text
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-v4-flash
DEEPSEEK_API_KEY=你的密钥
LLM_TIMEOUT_SECONDS=60
LLM_TEMPERATURE=0.2
```

也可以在网页左侧“专家模型”区域选择“API 模型专家”，直接填入 API 配置。

## 5. Streamlit Community Cloud 部署

1. 将项目上传到 GitHub 仓库。
2. 登录 Streamlit Community Cloud。
3. 新建 App。
4. 选择仓库和分支。
5. Main file path 填：

```text
app.py
```

6. Secrets 中可选配置：

```toml
OPENAI_BASE_URL = "https://api.deepseek.com"
OPENAI_MODEL = "deepseek-v4-flash"
DEEPSEEK_API_KEY = "你的密钥"
LLM_TIMEOUT_SECONDS = "60"
LLM_TEMPERATURE = "0.2"
```

部署后打开 Streamlit 提供的公网 URL 即可。

## 6. 常见问题

### 页面显示旧报告

点击左侧“清除缓存后重新抓取”。

### 新闻数量不足

系统会使用本地行业演示新闻补足数量，保证 Demo 稳定运行。

### 某些接口抓取失败

公开网页接口可能出现网络超时或限流。重新点击“清除缓存后重新抓取”，或稍后再试。

### 专家模型调用失败

检查：

- API Key 是否正确。
- Base URL 是否兼容 OpenAI Chat Completions。
- 模型名称是否正确。
- 网络是否可访问该 API 服务。

失败时可以切换为“本地规则专家”继续演示。
