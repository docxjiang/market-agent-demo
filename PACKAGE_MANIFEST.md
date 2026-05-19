# 交付清单

本交付包包含源代码、说明文档和 Demo 示例输出。

## 源代码

```text
app.py
src/
scripts/
prompts/
```

说明：

- `app.py`：Streamlit 网页应用入口。
- `src/`：核心模块，包括数据抓取、智能体、技术指标、图表、专家对话。
- `scripts/`：命令行报告生成脚本。
- `prompts/`：智能体提示词模板。

## 配置和依赖

```text
requirements.txt
.env.example
```

说明：

- `requirements.txt`：项目依赖。
- `.env.example`：OpenAI-compatible API 配置示例。

## 数据和 Demo

```text
data/
outputs/
```

说明：

- `data/`：本地示例数据、股票池、兜底数据。
- `outputs/`：已生成的 Demo 报告、新闻、公告、行情 CSV 和技术图表。

推荐查看：

```text
outputs/verify_expert_beiyuan_report.md
outputs/000657_中钨高新_report.md
outputs/600519_贵州茅台_report.md
outputs/charts/
```

## 文档

```text
README.md
DEPLOYMENT.md
DEMO_GUIDE.md
PROJECT_SPEC.md
PACKAGE_MANIFEST.md
```

说明：

- `README.md`：项目总说明。
- `DEPLOYMENT.md`：部署说明。
- `DEMO_GUIDE.md`：演示流程。
- `PROJECT_SPEC.md`：项目需求和设计边界。
- `PACKAGE_MANIFEST.md`：交付清单。

## 不包含内容

压缩包不包含：

- Python 缓存文件：`__pycache__/`、`*.pyc`
- 本地虚拟环境：`.venv/`
- 私密配置：`.env`
- 空日志文件：`streamlit.out.log`、`streamlit.err.log`

如需接入 API 模型，请在部署环境中自行创建 `.env` 或在网页左侧填写 API 配置。
