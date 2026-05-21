# A股个人持仓新闻解读与风险提醒系统

课程项目 Demo：输入任意 A 股股票代码和个人持仓信息，系统自动抓取行情、新闻、财报公告，生成技术图表、多智能体分析报告，并提供可对话的“专家意见”。
网站体验：https://market-agent-demo.streamlit.app/

> 免责声明：本系统仅用于课程演示和市场信息解读，不构成投资建议，不提供买入、卖出、持有、加仓、减仓等交易建议，不进行自动交易。

## 主要功能

- Streamlit 网页应用：左侧输入股票代码、持仓数量和成本价，一键生成报告。
- 自动识别股票资料：股票名称、交易所、行业、当前价。
- 行情与图表：抓取 A 股日 K 数据，生成 K 线均线图、MACD 图、成交额图。
- 新闻与公告：抓取相关新闻、东方财富公告和财报正文片段。
- 多智能体分析：新闻解读、财报公告、技术观察、风险提示、综合报告。
- 技术观察增强：基于 MA、MACD、成交量/成交额，输出短线走势初步判断。
- 专家意见对话：专家读取当前股票的全部报告数据后回答问题。
- API 模型可配置：默认可使用本地规则专家，也可接入 OpenAI-compatible API，例如 DeepSeek。
- 命令行生成报告：支持无网页环境下生成任意股票报告。

## 项目结构

```text
market-agent-demo/
  app.py                         # Streamlit 网页入口
  requirements.txt               # Python 依赖
  .env.example                   # API 配置模板
  README.md                      # 项目说明
  DEPLOYMENT.md                  # 部署说明
  DEMO_GUIDE.md                  # 演示说明
  PACKAGE_MANIFEST.md            # 交付清单
  data/                          # 本地样例数据和股票池
  outputs/                       # 示例报告、新闻、公告、图表
  prompts/                       # 智能体提示词
  scripts/                       # 命令行报告生成脚本
  src/                           # 核心代码
```

## 快速启动

进入项目目录：

```powershell
cd C:\Users\JIANG\Documents\Codex\2026-05-16\python-demo-demo-api-1-2\market-agent-demo
```

安装依赖：

```powershell
D:\Anaconda\python.exe -m pip install -r requirements.txt
```

启动网页：

```powershell
D:\Anaconda\python.exe -m streamlit run app.py
```

浏览器打开：

```text
http://localhost:8501
```

## 网页使用方式

1. 在左侧输入 6 位 A 股代码，例如 `601568`。
2. 输入持仓数量和成本价。
3. 点击“生成分析报告”。
4. 查看页面中的：
   - 概览
   - 技术图表
   - 智能体分析
   - 专家意见
   - 新闻与公告
   - 完整报告
5. 如果页面仍显示旧结果，点击左侧“清除缓存后重新抓取”。

## 专家意见

“专家意见”页签支持直接对话。

默认模式为“当前默认 AI 专家”：

- 如果环境变量中配置了 `OPENAI_API_KEY` 或 `DEEPSEEK_API_KEY`，会调用默认 API 模型。
- 如果没有配置 API Key，会自动回退到本地规则专家，不调用外部服务。

也可以在左侧切换为“API 模型专家”，手动填写：

```text
Base URL
模型名称
API Key
Temperature
```

API 要兼容 OpenAI Chat Completions 格式。

## 命令行生成报告

示例：生成北元集团报告。

```powershell
D:\Anaconda\python.exe scripts\generate_stock_report.py --code 601568 --name 北元集团 --sector 化学原料 --quantity 1000 --cost-price 4 --market-price 4.5 --output-prefix beiyuan
```

示例：生成中钨高新报告。

```powershell
D:\Anaconda\python.exe scripts\generate_stock_report.py --code 000657 --name 中钨高新 --sector 有色金属 --quantity 1000 --cost-price 50 --market-price 56.46 --output-prefix zhongwugaoxin
```

报告输出在：

```text
outputs/
```

## API 配置

复制 `.env.example` 为 `.env`，填入自己的 API Key：

```text
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-v4-flash
DEEPSEEK_API_KEY=你的密钥
USE_LLM_REFINER=0
LLM_TIMEOUT_SECONDS=60
LLM_TEMPERATURE=0.2
```

说明：

- `USE_LLM_REFINER=0`：报告生成仍使用本地规则智能体。
- `USE_LLM_REFINER=1`：命令行生成报告时可调用 LLM 精修报告。
- 网页“专家意见”可在页面左侧直接填写 API 配置，不必须创建 `.env`。

## 示例输出

项目已包含若干示例输出：

- `outputs/verify_expert_beiyuan_report.md`
- `outputs/000657_中钨高新_report.md`
- `outputs/600519_贵州茅台_report.md`
- `outputs/charts/`

## 重要限制

- 新闻和公告来自公开网页接口，可能受到网络、接口限流、源站结构变化影响。
- 行业和新闻相关性过滤是课程 Demo 级别，不等同于专业资讯数据库。
- 技术指标只做 MA、MACD、成交量等基础观察，不构成投资建议。
- 专家对话只基于当前抓取和生成的报告上下文，不保证覆盖全部公开信息。
