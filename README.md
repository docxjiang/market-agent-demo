# A股个人持仓新闻解读与风险提醒系统

A Streamlit-based multi-agent demo for A-share market news interpretation, technical chart generation, announcement summarization, and portfolio risk reminder.

本项目是一个课程项目 Demo。用户输入任意 6 位 A 股股票代码、个人持仓数量和成本价后，系统会自动抓取股票资料、行情数据、相关新闻、公告和财报片段，生成技术图表、多智能体分析报告，并提供可对话的“专家意见”模块。

网站体验：

```text
https://market-agent-demo.streamlit.app/
```

---

## 1. 重要免责声明

本系统仅用于课程项目演示和公开市场信息解读，不构成任何投资建议。

系统不会提供买入、卖出、持有、加仓、减仓等交易指令，也不会进行自动交易。系统输出内容基于公开网页数据、技术指标和模型生成结果，可能存在数据延迟、抓取失败、信息遗漏、模型误判或表述不准确等情况。

用户不应将本系统输出作为投资决策依据。

---

## 2. 项目简介

本项目围绕 A 股个人持仓场景，构建了一个轻量级市场信息解读系统。系统将股票行情、新闻公告、财报片段和技术指标整合到统一报告中，并通过多个分析模块分别完成新闻解读、公告财报梳理、技术观察、风险提示和综合总结。

项目支持两种使用方式：

1. **Streamlit 网页应用**：适合课程演示和交互体验；
2. **命令行报告生成**：适合在无网页环境下批量生成报告。

系统默认可以使用本地规则专家，不依赖外部 API；如果配置 OpenAI-compatible API，也可以启用模型专家对话和报告精修功能。

---

## 3. 功能亮点

* **一键生成股票分析报告**
  在网页左侧输入股票代码、持仓数量和成本价，即可生成当前股票的综合解读报告。

* **自动识别股票基础信息**
  输入 6 位 A 股代码后，系统自动识别股票名称、交易所、行业和当前价格。

* **行情数据与技术图表**
  自动抓取 A 股日 K 数据，生成 K 线均线图、MACD 图和成交额图，用于展示近期价格和成交变化。

* **新闻与公告抓取**
  自动抓取相关新闻、东方财富公告和财报正文片段，辅助判断近期市场关注点和公司基本面变化。

* **多智能体分析流程**
  系统将新闻、公告财报、技术指标和风险提示拆分给不同分析模块，最后生成综合报告。

* **技术观察增强**
  基于 MA、MACD、成交量和成交额等指标，输出短线技术状态的初步观察。

* **专家意见对话**
  专家模块会读取当前股票的全部报告上下文，用户可以围绕当前股票进行追问。

* **API 模型可配置**
  默认使用本地规则专家；也可以接入 OpenAI-compatible API，例如 DeepSeek 或其他兼容服务。

* **命令行生成报告**
  支持在无网页环境下，通过命令行生成任意股票报告。

---

## 4. 工作流程

系统整体工作流程如下：

1. 用户输入股票代码、持仓数量和成本价；
2. 系统自动识别股票名称、交易所、行业和当前价格；
3. 抓取近期行情数据；
4. 生成 K 线均线图、MACD 图和成交额图；
5. 抓取相关新闻、公告和财报正文片段；
6. 多个智能体分别进行新闻解读、公告财报分析、技术观察和风险提示；
7. 汇总生成完整报告；
8. 专家意见模块基于当前报告上下文回答用户问题。

---

## 5. 环境要求

建议使用：

```text
Python 3.9+
```

主要依赖包括：

* streamlit
* pandas
* numpy
* matplotlib / plotly
* requests
* python-dotenv
* openai 或 OpenAI-compatible SDK

具体依赖以项目中的 `requirements.txt` 为准。

---

## 6. 快速启动

进入项目目录：

```bash
cd market-agent-demo
```

安装依赖：

```bash
python -m pip install -r requirements.txt
```

启动网页应用：

```bash
python -m streamlit run app.py
```

浏览器打开：

```text
http://localhost:8501
```

如果使用 Anaconda 或指定 Python 解释器，可以将 `python` 替换为本地解释器路径，例如：

```bash
D:\Anaconda\python.exe -m pip install -r requirements.txt
D:\Anaconda\python.exe -m streamlit run app.py
```

---

## 7. 网页使用方式

启动 Streamlit 后，在网页左侧输入：

1. **股票代码**
   例如：`601568`、`000657`、`600519`

2. **持仓数量**
   例如：`1000`

3. **成本价**
   例如：`4.00`

点击：

```text
生成分析报告
```

系统会生成以下内容：

* 股票概览
* 技术图表
* 新闻解读
* 公告与财报分析
* 技术观察
* 风险提示
* 综合报告
* 专家意见对话
* 新闻与公告原文片段
* 完整 Markdown 报告

如果页面仍显示旧结果，可以点击左侧：

```text
清除缓存后重新抓取
```

---

## 8. 专家意见对话

“专家意见”页签支持对当前股票报告进行追问。

默认模式为：

```text
当前默认 AI 专家
```

如果环境变量中配置了 `OPENAI_API_KEY` 或 `DEEPSEEK_API_KEY`，系统会调用默认 API 模型；如果没有配置 API Key，系统会自动回退到本地规则专家，不调用外部服务。

也可以在网页左侧切换为：

```text
API 模型专家
```

并手动填写：

* Base URL
* 模型名称
* API Key
* Temperature

API 服务需要兼容 OpenAI Chat Completions 格式。

---

## 9. 命令行生成报告

项目支持在命令行中生成股票报告，适合无网页环境或批量测试场景。

### 示例 1：生成北元集团报告

```bash
python scripts\generate_stock_report.py --code 601568 --name 北元集团 --sector 化学原料 --quantity 1000 --cost-price 4 --market-price 4.5 --output-prefix beiyuan
```

### 示例 2：生成中钨高新报告

```bash
python scripts\generate_stock_report.py --code 000657 --name 中钨高新 --sector 有色金属 --quantity 1000 --cost-price 50 --market-price 56.46 --output-prefix zhongwugaoxin
```

报告默认输出到：

```text
outputs/
```

### 参数说明

| 参数              | 含义              |
| ----------------- | ----------------- |
| `--code`          | 6 位 A 股股票代码 |
| `--name`          | 股票名称          |
| `--sector`        | 所属行业          |
| `--quantity`      | 持仓数量          |
| `--cost-price`    | 持仓成本价        |
| `--market-price`  | 当前或假设市场价  |
| `--output-prefix` | 输出文件名前缀    |

如果使用指定 Python 解释器，可以写成：

```bash
D:\Anaconda\python.exe scripts\generate_stock_report.py --code 601568 --name 北元集团 --sector 化学原料 --quantity 1000 --cost-price 4 --market-price 4.5 --output-prefix beiyuan
```

---

## 10. API 与模型配置

本项目默认支持本地规则专家；如果配置 OpenAI-compatible API，可启用模型专家对话或报告精修功能。

复制 `.env.example` 为 `.env`：

```bash
copy .env.example .env
```

或手动新建 `.env` 文件。

示例配置：

```env
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-v4-flash
DEEPSEEK_API_KEY=你的密钥

USE_LLM_REFINER=0
LLM_TIMEOUT_SECONDS=60
LLM_TEMPERATURE=0.2
```

### 环境变量说明

| 环境变量              | 作用                            | 示例                       |
| --------------------- | ------------------------------- | -------------------------- |
| `OPENAI_BASE_URL`     | OpenAI-compatible API 服务地址  | `https://api.deepseek.com` |
| `OPENAI_MODEL`        | 默认调用模型名称                | `deepseek-v4-flash`        |
| `OPENAI_API_KEY`      | OpenAI-compatible API Key       | `sk-...`                   |
| `DEEPSEEK_API_KEY`    | DeepSeek API Key                | `sk-...`                   |
| `USE_LLM_REFINER`     | 是否在命令行报告中调用 LLM 精修 | `0` 或 `1`                 |
| `LLM_TIMEOUT_SECONDS` | API 请求超时时间，单位为秒      | `60`                       |
| `LLM_TEMPERATURE`     | 模型温度参数                    | `0.2`                      |

### 配置说明

* `USE_LLM_REFINER=0`：报告生成仍使用本地规则智能体；
* `USE_LLM_REFINER=1`：命令行生成报告时可调用 LLM 精修报告；
* 网页中的“专家意见”可以在页面左侧直接填写 API 配置，不必须创建 `.env`；
* 如果没有配置 API Key，系统会自动使用本地规则专家。

---

## 11. 项目结构

```text
market-agent-demo/
├── app.py                         # Streamlit 网页入口
├── requirements.txt               # Python 依赖
├── .env.example                   # API 配置模板
├── README.md                      # 项目说明
├── DEPLOYMENT.md                  # 部署说明
├── DEMO_GUIDE.md                  # 课堂演示说明
├── PACKAGE_MANIFEST.md            # 项目交付清单
├── data/                          # 本地样例数据和股票池
├── outputs/                       # 示例报告、新闻、公告和图表
├── prompts/                       # 多智能体提示词
├── scripts/                       # 命令行报告生成脚本
└── src/                           # 行情抓取、新闻抓取、图表生成和智能体核心代码
```

---

## 12. 示例输出

项目已包含若干示例输出：

```text
outputs/verify_expert_beiyuan_report.md
outputs/000657_中钨高新_report.md
outputs/600519_贵州茅台_report.md
outputs/charts/
```

示例报告可用于检查系统输出格式，也可作为课堂演示材料。

---

## 13. 隐私与安全说明

本项目不会进行自动交易，也不会连接证券账户。用户输入的持仓数量和成本价仅用于本地计算浮动盈亏和生成风险提示。

如果启用外部 API 模型专家，当前股票报告上下文和用户问题可能会发送至对应 API 服务商。请勿在对话中输入身份证号、证券账户、银行卡号、真实资产规模等敏感信息。

---

## 14. 已知限制

* **数据完整性限制**
  新闻、公告和财报片段来自公开网页接口，可能受到网络状态、接口限流或源站结构变化影响。

* **时效性限制**
  行情、新闻和公告可能存在延迟，不保证与交易所或专业金融终端完全一致。

* **数据源稳定性限制**
  公开网页接口可能出现字段变化、反爬限制或临时不可用情况。

* **技术指标限制**
  MA、MACD、成交量和成交额等指标只能反映历史价格与成交变化，不代表未来走势。

* **模型输出限制**
  专家意见由本地规则或大语言模型生成，可能存在误判、遗漏或表述不准确。

* **投资适用性限制**
  系统输出不构成投资建议，不应作为买入、卖出、持有或仓位调整依据。

---

## 15. 课程项目说明

本项目主要用于课程 Demo，目标是展示以下能力：

* 使用 Streamlit 构建交互式网页应用；
* 自动抓取和组织公开市场数据；
* 生成股票行情与技术指标图表；
* 将新闻、公告、财报和技术观察拆分为多个分析模块；
* 基于报告上下文实现专家问答；
* 通过本地规则和 OpenAI-compatible API 兼容模式实现可配置的智能体系统。

本项目不是专业投资系统，也不是证券交易工具。其主要价值在于展示市场信息聚合、多智能体分析流程和课程项目工程化实现。
