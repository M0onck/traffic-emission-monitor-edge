
# 个人项目 Git 工作流与提交规范指南

## 一、 轻量级分支策略 (Feature Branching)

在个人项目中，建议采用 **主干常绿** 的分支模型：

* **`main` (或 `master`)**：**绝对稳定分支**。这个分支上的代码必须随时可以运行或部署。永远不要在这个分支上直接写代码。
* **`develop`**：**日常开发主线**。所有的新功能集成的目标分支。
* **`feature/*`**：**特性分支**。用于开发具体的新功能或进行重构。开发完成后合并回 `develop`，并删除该分支。
* *命名示例*：`feature/ui-simplification`, `feature/hailo-integration`


* **`bugfix/*` 或 `hotfix/***`：**修复分支**。用于修复特定的 Bug。
* *命名示例*：`bugfix/fps-drop-issue`



### 标准开发循环

1. 从 `develop` 创建并切换到新分支：`git switch -c feature/xxx`
2. 在分支上进行开发和多次小步提交 (Commit)。
3. 开发完成并测试通过后，切换回 `develop`：`git switch develop`
4. 将特性分支合并进来：`git merge feature/xxx`
5. 删除特性分支保持环境整洁：`git branch -d feature/xxx`

---

## 二、 提交信息 (Commit Message) 规范

推荐采用业界最通用的 **Angular 规范**。每次提交的信息应该像一句简短的说明书，结构如下：

```text
<type>(<scope>): <subject>

```

### 1. `type` (必须)：说明提交的类别

请严格使用以下标识符之一：

* **`feat`**: 新增功能 (Feature)
* **`fix`**: 修复 Bug
* **`refactor`**: 重构代码（既不新增功能，也不修复 Bug 的代码变动，如优化逻辑、重命名变量）
* **`docs`**: 仅仅修改了文档（如 `README.md`, 注释等）
* **`style`**: 不影响代码含义的修改（如删除空格、格式化、补充缺失的分号等）
* **`perf`**: 提升性能的代码修改
* **`test`**: 增加或修改测试用例
* **`chore`**: 构建过程或辅助工具的变动（如更新依赖库 `requirements.txt`）

### 2. `scope` (可选)：说明影响的范围

用于说明这次提交主要影响了哪个模块。对于我们的系统，可以这样划分：

* `(ui)`: 涉及 `renderer.py` 等界面渲染层
* `(engine)`: 涉及 `monitor_engine.py` 核心调度层
* `(physics)`: 涉及微观排放或动力学计算模块
* `(model)`: 涉及车牌分类或硬件模型管道

### 3. `subject` (必须)：简短描述

* 使用**英文**编写（避免中英文切换带来的乱码风险，且更符合开源习惯）。
* 使用**祈使句**（动词原形开头），例如使用 `add` 而不是 `added` 或 `adds`。
* 不要在句末加句号 `.`。

---

## 三、 优秀提交范例对比

**❌ 糟糕的提交 (模糊不清，难以回溯)：**

> "update files"
> "fix bug"
> "修改了画框的颜色"

**✅ 规范的提交 (清晰明了，意图明确)：**

> `feat(ui): add color-coded bounding boxes for specific vehicle types`
> `refactor(engine): simplify label formatting for low-resolution edge displays`
> `fix(physics): correct geometric distance accumulation logic`
> `chore: update hailo inference dependencies in setup script`

---

## 四、 黄金法则

1. **原子化提交 (Atomic Commits)**：一次 Commit 只做一件事。不要把“修复了排放计算的 Bug”和“重构了 UI 渲染”混在一个 Commit 里。如果文件改动跨越了两个逻辑，请使用 `git add <file>` 分别暂存并多次提交。
2. **代码工作前先看状态**：养成敲击 `git status` 的肌肉记忆，随时明确自己在哪个分支，有哪些文件被修改了。
