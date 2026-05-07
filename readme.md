# 离线文档智能解析与归档系统 (Offline Document Parsing & Archiving System)

## 1. 项目概述与核心定位
本项目是一个面向离线内网环境的个人 AI 知识中枢系统。
系统的核心使命是将多种来源、多种格式的办公文档统一解析为标准的 Markdown 格式，并利用本地大模型实现图文多模态的结构与语义双重解析，最终建立可持续检索的知识库。

系统设计的首要原则为 "A 机与 B 机环境隔离"：
- **A 机 (联网机)**：负责下载开源模型、收集 Python 依赖，打包生成跨机器的离线部署包。
- **B 机 (离线生产机)**：接收离线部署包，一键还原运行环境，承担所有的重负载模型推理工作。

## 2. 硬件与环境约束
### 电脑 A (准备端)
- 状态：可联网。无显卡要求。
- 硬件：i5-2500K;8G内存;Windows 11 专业版 24H2 26100.8246。
- 任务：使用 `build_env_mineru_pack.bat` 脚本，通过 pip download 等工具，将 Python 依赖打包为 wheel 文件。

### 电脑 B (生产端)
- 状态：完全离线。
- 硬件：配备 RTX 级独立显卡 (例如 RTX 4070 Ti SUPER)，至少 12GB 显存。64G内存。CPU是i7-14700KF。
- 任务：解压运行环境，运行 pipeline，为用户提供 Gradio WebUI。

## 3. 核心流水线架构 (Pipeline)
输入格式支持：doc, docx, pdf, ppt, pptx, xls, xlsx, csv
输出格式标准：包含原图位置与语义注释的 Markdown 文件

整个流水线被设计为不可颠倒的四个阶段：

**第一阶段：格式归一化**
- 工具：LibreOffice
- 动作：统一将所有 Office 系输入文档转换为 PDF 格式。

**第二阶段：结构优先的版面解析**
- 模型：MinerU (采用核心模型 MinerU2.5-Pro-2604-1.2B)
- 运行环境：env_mineru
- 动作：对 PDF 进行高精度版面切分，提取正文、表格、公式。将页面上的所有图片和图表单独裁切并保存为图片文件，同时输出初步的 Markdown 文件。

**第三阶段：语义增强辅佐**
- 模型：Qwen2.5-VL-7B-Instruct
- 运行环境：env_paper
- 动作：读取 MinerU 裁出的图片，结合 Markdown 上下文，利用多模态大语言模型理解图片内的业务含义 (如：判断这是系统截图还是流程图，并给出摘要)。
- 兜底机制：若检测到 GPU 显存溢出 (OOM) 或推理失败，系统会自动降级采用轻量级 RapidOCR 提取图片内的文字进行语义兜底，防止流水线崩溃。

**第四阶段：可视化核对与写入**
- 工具：Gradio WebUI
- 动作：提供渐进式的流式进度展示，支持左侧原图、右侧机器解析文本的对比面板。允许人工强行干预和修改语义错误，核对无误后将其精准写回 Markdown 中的图片原图下方位置。

## 4. 关键架构设计与技术亮点
### 图片哈希全局缓存机制 (Hash Cache)
针对大量相似文档或重复图片 (如公文红头、公司 Logo、重复图表)，系统会在解析前计算每一张图片的 SHA1 哈希值。
缓存文件保存在 workspace/cache/qwen_image_cache.json。
一旦发生哈希碰撞，直接提取上次的语义理解结果或人工修改结果，秒级跳过大模型推理步骤。

### 渐进式流式交互 (Progressive Yield)
在传统的批处理机制上进行了流式重构。
每完成一张图片的 Qwen 推理，立刻通过 yield 机制将其推送到前端 UI 的画廊和工作区中，解决了用户面对大型文档时的 "长时间黑盒等待" 问题。

### 依赖配置双轨解耦
核心依赖不再硬编码于 Shell 脚本中，而是下放至 requirements_mineru.txt 和 requirements_paper.txt。
支持 "仅更新 Wheels 依赖包" 的增量部署模式，避免每次增加一个依赖都需要传输数 GB 级别的完整环境包。

### 增量补丁机制 (Patch System) ⭐ NEW
为解决版本兼容性问题（如 transformers 5.x 与 Qwen2VL 不兼容），系统引入了增量补丁机制：

#### A 机操作（联网）：
1. 编辑 `patch_config.json` 配置文件，指定需要安装/卸载/升级的包
2. 运行 `build_env_mineru_pack.bat`，选择 `[2] 增量补丁包`
3. 脚本自动下载指定的包及其依赖，生成 `MinERU_Patch_vX.X.X_YYYYMMDD_HHMMSS.zip` 补丁包
4. 补丁包保存在 `D:\ai_offline_pack\patch\` 目录

#### B 机操作（离线）：
1. 将补丁包从 A 机复制到 `D:\ai_offline_pack\patch\` 目录
2. 运行 `deploy_env_mineru.bat`
3. 选择 `[2] 应用增量更新补丁`
4. 脚本自动：
   - 备份当前环境状态
   - 卸载指定包
   - 安装/升级新包
   - 运行冒烟测试验证兼容性
   - 测试失败可提示回滚

#### 配置文件示例 (patch_config.json)：
```json
{
  "version": "1.0.0",
  "description": "修复 transformers 与 Qwen2VL 兼容性问题",
  "packages": {
    "install": [
      {"spec": "transformers==4.57.2", "with_deps": true}
    ],
    "uninstall": ["some_old_package"],
    "upgrade": [
      {"spec": "another_package==2.0.0", "with_deps": false}
    ]
  },
  "smoke_test": {
    "enabled": true,
    "test_code": "import transformers; print(transformers.__version__)"
  }
}
```

## 5. 项目目录规范
```
D:\ai_offline_pack\
├─ envs\
│  ├─ env_mineru\           (负责第一与第二阶段的结构解析独立环境)
│  └─ env_paper\            (负责后续语义增强、WebUI 及文本大模型的独立环境)
├─ models\
│  ├─ llm\
│  │  └─ Qwen2.5-VL-7B-Instruct\
│  └─ embedding\
├─ workspace\
│  ├─ input\                (原始办公文档输入目录)
│  ├─ output\               (Markdown 产出及图片资源存放区)
│  ├─ cache\                (哈希缓存数据库存放区)
│  └─ tmp\                  (运行过程切片存放区)
├─ wheels\
│  └─ mineru\               (Python 依赖 wheel 文件)
├─ patch\                    (增量补丁包存放目录) ⭐ NEW
│  ├─ MinERU_Patch_v1_0_0_20250101_120000.zip
│  └─ backup_20250101_120000.json  (环境状态备份)
├─ scripts\                  (包含 build_* 与 deploy_* 的全套环境维护工具链)
├─ build_env_mineru_pack.bat (A 机构建脚本)
├─ deploy_env_mineru.bat     (B 机部署脚本)
├── requirements_mineru.txt        (依赖列表)
├── requirements_mineru_lock.txt   (锁定版本的依赖列表)
└── patch_config.json              (增量补丁配置文件) ⭐ NEW
```

## 6. AI 协作与后续开发提示 (针对其他接手 AI)
1. **语言规范**：所有脚本优先采用跨平台兼容性最好的 Bash (.sh)；对于一键式轻量启动，使用 Windows Batch (.bat)。
2. **架构红线**：绝对不允许用 Qwen 替代 MinerU 的结构解析职责。MinerU 负责解构，Qwen 负责语义，两套环境互相隔离、各司其职，这是本系统兼顾性能与精度的基石所在。
3. **扩展方向**：未来可在知识入库阶段接入 Baichuan2 / bge-m3 等文本侧模型，最终对接 ChromaDB 构建问答检索系统。
4. **补丁机制扩展**：
   - 可实现自动回滚功能（目前仅提示手动处理）
   - 可增加依赖冲突预检测
   - 可支持多个补丁包的依赖关系管理