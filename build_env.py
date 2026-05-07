import os
import sys
import subprocess
import shutil
import json
from pathlib import Path
from datetime import datetime
import zipfile
import re

# --- 配置区域 ---
ROOT_DIR = Path(r"D:\ai_offline_pack")
MODEL_ROOT = ROOT_DIR / "models"

# 镜像源
MAIN_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"
EXTRA_INDEX_1 = "https://download.pytorch.org/whl/cu121"
EXTRA_INDEX_2 = "https://pypi.org/simple"

# 核心包 (用于兜底)
CORE_PKGS = ["pip", "setuptools", "wheel"]

# --- 工具函数 ---
class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    END = '\033[0m'
    BOLD = '\033[1m'

def log_info(msg): print(f"{Color.BLUE}[INFO]{Color.END} {msg}")
def log_ok(msg): print(f"{Color.GREEN}[OK]{Color.END} {msg}")
def log_warn(msg): print(f"{Color.YELLOW}[WARN]{Color.END} {msg}")
def log_fatal(msg): 
    print(f"{Color.RED}[FATAL]{Color.END} {msg}")
    input("按任意键退出...")
    sys.exit(1)

def run_cmd(cmd, shell=False, check=True):
    try:
        if isinstance(cmd, list) and not shell:
            proc = subprocess.run(cmd, check=check, shell=False)
        else:
            proc = subprocess.run(cmd, check=check, shell=True)
        return proc
    except subprocess.CalledProcessError as e:
        if check:
            log_fatal(f"命令执行失败：{e}")
        return None

def zip_directory_with_progress(src_dir: Path, zip_file: Path):
    src_dir = Path(src_dir)
    zip_file = Path(zip_file)
    if not src_dir.exists():
        log_warn(f"源目录不存在，无法打包: {src_dir}")
        return
        
    log_info(f"正在扫描文件: {src_dir}")
    file_list = []
    for root, dirs, files in os.walk(src_dir):
        for f in files:
            file_list.append(Path(root) / f)
            
    total_files = len(file_list)
    if total_files == 0:
        log_warn(f"源目录为空: {src_dir}")
        return
        
    log_info(f"开始打包 (共 {total_files} 个文件) -> {zip_file}")
    zip_file.parent.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as zipf:
        for i, file_path in enumerate(file_list, 1):
            arcname = file_path.relative_to(src_dir)
            zipf.write(file_path, arcname)
            
            # 简单的进度条
            if i % max(1, total_files // 100) == 0 or i == total_files:
                percent = i * 100 // total_files
                bar = "#" * (percent // 2) + "-" * (50 - percent // 2)
                sys.stdout.write(f"\r  [{bar}] {percent}% ({i}/{total_files})")
                sys.stdout.flush()
    print() # 换行
    log_ok(f"打包完成: {zip_file.name} (大小: {zip_file.stat().st_size / (1024*1024):.2f} MB)")

class EnvConfig:
    def __init__(self, name, env_dir_name, req_file, patch_file, wheel_dir_name, prefix_name):
        self.name = name
        self.prefix = prefix_name
        
        self.ENV_DIR = ROOT_DIR / "envs" / env_dir_name
        self.WHEEL_DIR = ROOT_DIR / "wheels" / wheel_dir_name
        
        # 导出目录 (全量产物)
        self.EXPORT_DIR = ROOT_DIR / "exported" / f"full_{prefix_name.lower()}"
        
        # 补丁目录 (增量产物)
        self.PATCH_DIR = ROOT_DIR / "patch" / f"patch_{prefix_name.lower()}"
        
        self.REQ_FILE = Path(__file__).parent / req_file
        self.LOCK_FILE = Path(__file__).parent / f"{req_file.replace('.txt', '_lock.txt')}"
        self.PATCH_CONFIG_FILE = ROOT_DIR / patch_file

ENV_PAPER_CONFIG = EnvConfig(
    name="大模型工作环境 (env_paper)",
    env_dir_name="env_paper",
    req_file="requirements_paper.txt",
    patch_file="patch_config_paper.json",
    wheel_dir_name="paper",
    prefix_name="Paper"
)

ENV_MINERU_CONFIG = EnvConfig(
    name="MinerU 工作环境 (env_mineru)",
    env_dir_name="env_mineru",
    req_file="requirements_mineru.txt",
    patch_file="patch_config_mineru.json",
    wheel_dir_name="mineru",
    prefix_name="MinERU"
)

# ─────────────────────────────────────────────────────────
# 补丁依赖解析辅助函数
# ─────────────────────────────────────────────────────────

def _parse_freeze(output: str) -> dict:
    """pip freeze 输出 → {包名(小写,连字符): 版本}"""
    pkgs = {}
    for line in output.strip().splitlines():
        line = line.strip()
        if "==" in line and not line.startswith(("#", "-e")):
            name, ver = line.split("==", 1)
            pkgs[name.strip().lower().replace("_", "-")] = ver.strip()
    return pkgs

def _load_lock(lock_file: Path) -> dict:
    """读取 lock 文件 → {包名: 版本}"""
    if not lock_file.exists():
        return {}
    with open(lock_file, "r", encoding="utf-8") as f:
        return _parse_freeze(f.read())

def _collect_targets() -> list:
    """交互式收集用户想要更新/安装的目标包列表"""
    print()
    print(f"{Color.CYAN}─── 请输入要更新/安装的包（每行一个）───{Color.END}")
    print(f"  格式示例: transformers==4.45.2")
    print(f"  支持多个，输入完毕后直接按 Enter 结束。")
    print()
    specs = []
    while True:
        try:
            entry = input(f"  [{len(specs)+1}] 包名==版本 (回车结束): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not entry:
            if not specs:
                print(f"  {Color.YELLOW}[提示] 至少输入一个包。{Color.END}")
                continue
            break
        specs.append(entry)
        print(f"  {Color.GREEN}✓ 已添加: {entry}{Color.END}")
    return specs

def _resolve_deps(config: "EnvConfig", target_specs: list) -> dict:
    """
    在临时 Python 3.10 环境中安装 target_specs，
    pip freeze 后与 lock 文件 diff，
    返回 {"install": [...], "upgrade": [...], "uninstall": [], "after": {}}
    """
    baseline = _load_lock(config.LOCK_FILE)
    log_info(f"基线包数量: {len(baseline)} (来自 {config.LOCK_FILE.name})" if baseline
             else "未找到 lock 文件，以空基线解析（所有包标记为 install）")

    conda_exe = shutil.which("conda")
    if not conda_exe:
        log_fatal("未找到 conda 命令。")

    tmp_env = ROOT_DIR / "envs" / f"_resolve_{config.prefix.lower()}"
    if tmp_env.exists():
        shutil.rmtree(tmp_env)

    log_info("创建临时解析环境 (Python 3.10)…")
    run_cmd([conda_exe, "create", "-y", "-p", str(tmp_env), "python=3.10"])
    tmp_py = str(tmp_env / "python.exe")

    try:
        log_info(f"安装目标包并解析依赖: {target_specs}")
        rc = run_cmd([
            tmp_py, "-m", "pip", "install",
            "--index-url", MAIN_INDEX,
            "--extra-index-url", EXTRA_INDEX_1,
            "--extra-index-url", EXTRA_INDEX_2,
        ] + target_specs, check=False)
        if rc is None or rc.returncode != 0:
            log_fatal(f"安装目标包失败，请检查包名/版本: {target_specs}")

        freeze_out = subprocess.run(
            [tmp_py, "-m", "pip", "freeze"],
            capture_output=True, text=True, check=True
        ).stdout
        after = _parse_freeze(freeze_out)
        log_info(f"临时环境共解析到 {len(after)} 个包")

        skip = {"pip", "setuptools", "wheel", "pkg-resources"}
        to_install, to_upgrade = [], []
        for name, ver in sorted(after.items()):
            if name in skip:
                continue
            bver = baseline.get(name)
            if bver is None:
                to_install.append(f"{name}=={ver}")
            elif bver != ver:
                to_upgrade.append(f"{name}=={ver}")

        return {"install": to_install, "upgrade": to_upgrade,
                "uninstall": [], "after": after}
    finally:
        log_info("清理临时解析环境…")
        shutil.rmtree(tmp_env, ignore_errors=True)


# ─────────────────────────────────────────────────────────
# 主函数：构建增量补丁包
# ─────────────────────────────────────────────────────────

def build_patch_package(config: EnvConfig):
    """构建增量补丁包（交互式依赖解析，自动生成 patch_config + zip）"""
    log_info(f"=== 开始构建增量补丁包 ({config.name}) ===")

    # ── Step 1: 收集目标包 ──────────────────────────────
    target_specs = _collect_targets()
    if not target_specs:
        log_warn("未输入任何目标包，操作取消。")
        return

    print(f"\n{Color.BOLD}目标包:{Color.END} {', '.join(target_specs)}\n")

    # ── Step 2: 解析级联依赖 ────────────────────────────
    dep = _resolve_deps(config, target_specs)
    to_install  = dep["install"]
    to_upgrade  = dep["upgrade"]
    to_uninstall = dep["uninstall"]
    all_download = to_install + to_upgrade

    # ── Step 3: 展示变化，请求确认 ──────────────────────
    baseline = _load_lock(config.LOCK_FILE)
    print(f"\n{Color.CYAN}════════ 依赖变化分析结果 ════════{Color.END}")

    if to_install:
        print(f"\n{Color.GREEN}新增安装 ({len(to_install)} 个):{Color.END}")
        for p in to_install:
            print(f"    + {p}")

    if to_upgrade:
        print(f"\n{Color.YELLOW}版本升级 ({len(to_upgrade)} 个):{Color.END}")
        for p in to_upgrade:
            name = p.split("==")[0]
            new_ver = p.split("==")[1]
            old_ver = baseline.get(name.lower().replace("_", "-"), "?")
            print(f"    ↑ {name}: {old_ver}  →  {new_ver}")

    if to_uninstall:
        print(f"\n{Color.RED}显式卸载 ({len(to_uninstall)} 个):{Color.END}")
        for p in to_uninstall:
            print(f"    - {p}")

    if not all_download and not to_uninstall:
        print(f"\n{Color.GREEN}当前环境与目标版本一致，无需生成补丁。{Color.END}")
        input("按任意键退出...")
        return

    print(f"\n{Color.CYAN}══════════════════════════════════{Color.END}")
    print(f"共需下载 {len(all_download)} 个 Wheel 文件。")

    # ── 用户输入版本号和描述 ────────────────────────────
    patch_version = input(f"\n请输入补丁版本号 [默认 1.0.0]: ").strip() or "1.0.0"
    patch_desc    = input("请输入补丁描述   [默认: 依赖版本更新]: ").strip() or "依赖版本更新"

    confirm = input(f"\n{Color.YELLOW}确认以上变化并生成补丁包？(y/n): {Color.END}").strip().lower()
    if confirm != "y":
        log_info("操作取消。")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.PATCH_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 4: 生成 patch_config JSON ──────────────────
    patch_config_data = {
        "version": patch_version,
        "description": patch_desc,
        "created_at": datetime.now().isoformat(),
        "target_packages": target_specs,
        "packages": {
            "install":   to_install,
            "upgrade":   to_upgrade,
            "uninstall": to_uninstall,
        }
    }
    # 带时间戳的归档版
    cfg_archive = config.PATCH_DIR / f"patch_config_{config.prefix.lower()}_{ts}.json"
    with open(cfg_archive, "w", encoding="utf-8") as f:
        json.dump(patch_config_data, f, indent=2, ensure_ascii=False)
    log_ok(f"已生成补丁配置（归档）: {cfg_archive.name}")

    # 同时覆写项目目录下供 deploy_env.py 使用的 patch_config
    cfg_live = Path(__file__).parent / f"patch_config_{config.prefix.lower()}.json"
    with open(cfg_live, "w", encoding="utf-8") as f:
        json.dump(patch_config_data, f, indent=2, ensure_ascii=False)
    log_ok(f"已同步更新: {cfg_live.name}")

    # ── Step 5: 下载 Wheels ─────────────────────────────
    patch_wheels_dir = config.PATCH_DIR / f"wheels_temp_{ts}"
    patch_wheels_dir.mkdir(parents=True, exist_ok=True)

    conda_exe = shutil.which("conda")
    tmp_dl_env = ROOT_DIR / "envs" / f"_dl_{config.prefix.lower()}"
    if tmp_dl_env.exists():
        shutil.rmtree(tmp_dl_env)
    log_info("创建临时下载环境 (Python 3.10)…")
    run_cmd([conda_exe, "create", "-y", "-p", str(tmp_dl_env), "python=3.10"])
    tmp_dl_py = str(tmp_dl_env / "python.exe")

    try:
        if all_download:
            log_info(f"正在下载 {len(all_download)} 个 Wheel 及其依赖…")
            run_cmd([
                tmp_dl_py, "-m", "pip", "download",
                "-d", str(patch_wheels_dir),
                "--index-url", MAIN_INDEX,
                "--extra-index-url", EXTRA_INDEX_1,
                "--extra-index-url", EXTRA_INDEX_2,
            ] + all_download)
    finally:
        log_info("清理临时下载环境…")
        shutil.rmtree(tmp_dl_env, ignore_errors=True)

    # ── Step 6: 写入 manifest.json 并打包 ───────────────
    with open(patch_wheels_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(patch_config_data, f, indent=2, ensure_ascii=False)

    patch_filename = f"{config.prefix}_Patch_v{patch_version}_{ts}.zip"
    patch_zip_path = config.PATCH_DIR / patch_filename
    log_info("正在打包增量补丁 ZIP…")
    zip_directory_with_progress(patch_wheels_dir, patch_zip_path)

    # 清理临时 wheels 目录
    shutil.rmtree(patch_wheels_dir, ignore_errors=True)

    print(f"\n{Color.GREEN}====================================================={Color.END}")
    print(f"{Color.BOLD}   补丁包构建成功！产物目录: {config.PATCH_DIR}{Color.END}")
    print(f"{Color.BOLD}   ZIP 文件: {patch_filename}{Color.END}")
    print(f"{Color.BOLD}   请将 ZIP 拷贝至 B 机，运行 deploy_env.py 进行增量部署。{Color.END}")
    print(f"{Color.GREEN}====================================================={Color.END}\n")
    input("按任意键退出...")


def build_full_package(config: EnvConfig):
    """全量构建逻辑"""
    log_info(f"=== 开始全量构建 ({config.name}) ===")
    
    # --- STEP 1: 检查 Conda ---
    log_info("检查 Conda 环境...")
    conda_exe = shutil.which("conda")
    if not conda_exe:
        log_fatal("未找到 conda 命令，请确保已安装 Anaconda/Miniconda 并加入 PATH。")

    # --- STEP 2: 创建/准备环境 ---
    log_info(f"正在清理并重建环境：{config.ENV_DIR}")
    if config.ENV_DIR.exists(): 
        shutil.rmtree(config.ENV_DIR)
    run_cmd([conda_exe, "create", "-y", "-p", str(config.ENV_DIR), "python=3.10"])
    
    python_exe = str(config.ENV_DIR / "python.exe")

    # --- STEP 3: 安装依赖 ---
    log_info("正在安装依赖到 A 机环境...")
    
    # 基础安装参数
    pip_args = [
        "-m", "pip", "install", 
        "--use-deprecated=legacy-resolver",
        "--index-url", MAIN_INDEX,
        "--extra-index-url", EXTRA_INDEX_1,
        "--extra-index-url", EXTRA_INDEX_2
    ]

    # 不同环境的特殊前置处理
    if config.prefix == "MinERU":
        log_info("安装 av...")
        run_cmd([python_exe] + pip_args + ["av"])
    elif config.prefix == "Paper":
        log_info("预安装 GPU Torch (Paper 环境要求)...")
        run_cmd([python_exe] + pip_args + [
            "torch==2.1.2", "torchvision==0.16.2", "torchaudio==2.1.2",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ])

    # 从 requirements 文件安装
    if config.REQ_FILE.exists():
        log_info(f"使用配置文件安装：{config.REQ_FILE}")
        run_cmd([python_exe] + pip_args + ["-r", str(config.REQ_FILE)])
    else:
        log_warn(f"未找到 {config.REQ_FILE.name}，直接安装核心包。")
        run_cmd([python_exe] + pip_args + CORE_PKGS)

    # 强制校验核心包版本
    log_info("强制校验核心依赖版本...")
    run_cmd([python_exe] + pip_args + CORE_PKGS)
    
    # 生成锁定表
    log_info(f"更新锁定表：{config.LOCK_FILE.name}")
    with open(config.LOCK_FILE, "w", encoding="utf-8") as f:
        subprocess.run([python_exe, "-m", "pip", "freeze"], stdout=f)

    # --- STEP 4: 下载 Wheels ---
    log_info("开始下载离线 Wheels 包...")
    config.WHEEL_DIR.mkdir(parents=True, exist_ok=True)
    
    download_args = [
        "-m", "pip", "download",
        "-d", str(config.WHEEL_DIR),
        "--index-url", MAIN_INDEX,
        "--extra-index-url", EXTRA_INDEX_1,
        "--extra-index-url", EXTRA_INDEX_2
    ]

    # 下载 GPU Torch
    if config.prefix == "MinERU":
        # 从 requirements_mineru.txt 中读取 torch 版本
        torch_version = None
        torchvision_version = None
        torchaudio_version = None
        with open(config.REQ_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("torch=="):
                    torch_version = line.split("==")[1]
                elif line.startswith("torchvision=="):
                    torchvision_version = line.split("==")[1]
                elif line.startswith("torchaudio=="):
                    torchaudio_version = line.split("==")[1]
        
        if torch_version and torchvision_version and torchaudio_version:
            log_info(f"下载 GPU Torch {torch_version} (cu121)...")
            run_cmd([python_exe] + download_args + [
                f"torch=={torch_version}", f"torchvision=={torchvision_version}", f"torchaudio=={torchaudio_version}",
                "--index-url", "https://download.pytorch.org/whl/cu121",
                "--no-deps"
            ])
        else:
            log_warn("未能在 requirements_mineru.txt 中找到 torch、torchvision、torchaudio 的版本信息，跳过下载。")
    elif config.prefix == "Paper":
        log_info("下载 GPU Torch 2.1.2 (cu121)...")
        run_cmd([python_exe] + download_args + [
            "torch==2.1.2", "torchvision==0.16.2", "torchaudio==2.1.2",
            "--index-url", "https://download.pytorch.org/whl/cu121/",
            "--no-deps"
        ])

    # 下载其他
    log_info("下载全量依赖包...")
    if config.LOCK_FILE.exists():
        run_cmd([python_exe] + download_args + ["-r", str(config.LOCK_FILE)])
    else:
        log_warn("锁定表不存在，跳过全量依赖下载。")

    # --- STEP 5: 下载模型 (仅限 MinERU) ---
    if config.prefix == "MinERU":
        log_info("准备 VLM 模型...")
        model_name = "MinerU2.5-Pro-2604-1.2B"
        model_dir = MODEL_ROOT / model_name
        if (model_dir / "model.safetensors").exists():
            log_ok("模型已存在，跳过。")
        else:
            log_info("模型不存在，尝试下载 (需要 modelscope)...")
            MODEL_ROOT.mkdir(parents=True, exist_ok=True)
            try:
                subprocess.run([python_exe, "-c", "import modelscope"], check=True, capture_output=True)
                download_script = f"""
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('OpenDataLab/{model_name}', cache_dir=r'{MODEL_ROOT}', local_dir=r'{model_dir}')
"""
                run_cmd([python_exe, "-c", download_script])
                log_ok("模型下载完成。")
            except subprocess.CalledProcessError:
                log_warn("未安装 modelscope 或下载失败，请手动下载模型。")

    # --- STEP 6: 导出产物 (ZIP 打包) ---
    log_info("导出环境与产物...")
    if config.EXPORT_DIR.exists():
        shutil.rmtree(config.EXPORT_DIR)
    config.EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 复制 requirements 文件
    if config.REQ_FILE.exists():
        shutil.copy2(config.REQ_FILE, config.EXPORT_DIR / config.REQ_FILE.name)
        log_info(f"已复制: {config.REQ_FILE.name}")
    if config.LOCK_FILE.exists():
        shutil.copy2(config.LOCK_FILE, config.EXPORT_DIR / config.LOCK_FILE.name)
        log_info(f"已复制: {config.LOCK_FILE.name}")
        
    # 打包 Wheels
    log_info("正在打包离线依赖包 (Wheels)...")
    wheel_zip_path = config.EXPORT_DIR / f"full_wheel_{config.prefix.lower()}.zip"
    zip_directory_with_progress(config.WHEEL_DIR, wheel_zip_path)
    
    # 打包 Environment
    log_info("正在打包 Conda 环境 (这可能需要几分钟)...")
    env_zip_path = config.EXPORT_DIR / f"full_env_{config.prefix.lower()}.zip"
    zip_directory_with_progress(config.ENV_DIR, env_zip_path)

    print(f"\n{Color.GREEN}====================================================={Color.END}")
    print(f"{Color.BOLD}   构建完成！全量产物已存放在：{config.EXPORT_DIR}{Color.END}")
    print(f"{Color.BOLD}   请将整个 {config.EXPORT_DIR.name} 目录拷贝至 B 机进行部署。{Color.END}")
    print(f"{Color.GREEN}====================================================={Color.END}\n")
    input("按任意键退出...")

# ─────────────────────────────────────────────────────────
# 模型下载（国内加速）
# ─────────────────────────────────────────────────────────

MODEL_NAME      = "MinerU2.5-Pro-2604-1.2B"
MS_MODEL_ID     = f"OpenDataLab/{MODEL_NAME}"         # ModelScope
HF_REPO_ID      = f"opendatalab/{MODEL_NAME}"          # HuggingFace
HF_CN_ENDPOINT  = "https://hf-mirror.com"             # 国内 HF 镜像

def download_model_only():
    """单独下载 MinerU VLM 模型到 D:\\ai_offline_pack\\models（国内加速）"""
    log_info("=" * 50)
    log_info(f"下载模型：{MODEL_NAME}")
    log_info("=" * 50)

    dest_dir = MODEL_ROOT / MODEL_NAME
    if dest_dir.exists() and any(dest_dir.iterdir()):
        log_warn(f"目标目录已存在且非空：{dest_dir}")
        overwrite = input("是否覆盖下载？(y/n) [默认 n]: ").strip().lower()
        if overwrite != "y":
            log_info("已取消。")
            input("按任意键退出...")
            return

    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{Color.CYAN}请选择下载源（国内推荐 ModelScope）{Color.END}")
    print(f"  {Color.BOLD}[1]{Color.END} ModelScope  ({MS_MODEL_ID})  ← 国内首选")
    print(f"  {Color.BOLD}[2]{Color.END} HuggingFace 镜像站 hf-mirror.com  ({HF_REPO_ID})")
    src = input("\n  请选择 (1/2) [默认 1]: ").strip() or "1"

    if src == "2":
        _download_hf_mirror(dest_dir)
    else:
        _download_modelscope(dest_dir)

def _download_modelscope(dest_dir: Path):
    """通过 ModelScope 下载"""
    log_info(f"使用 ModelScope 下载 → {dest_dir}")
    try:
        from modelscope.hub.snapshot_download import snapshot_download
        snapshot_download(
            MS_MODEL_ID,
            cache_dir=str(MODEL_ROOT),
            local_dir=str(dest_dir)
        )
        log_ok(f"下载完成：{dest_dir}")
    except ImportError:
        log_warn("未安装 modelscope，尝试 pip install modelscope 后重试。")
    except Exception as e:
        log_fatal(f"ModelScope 下载失败：{e}")
    _post_download(dest_dir)

def _download_hf_mirror(dest_dir: Path):
    """通过 hf-mirror.com 镜像下载"""
    import os
    os.environ["HF_ENDPOINT"] = HF_CN_ENDPOINT
    log_info(f"使用 HF 镜像站 {HF_CN_ENDPOINT} 下载 → {dest_dir}")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=HF_REPO_ID,
            local_dir=str(dest_dir),
            local_dir_use_symlinks=False,
            endpoint=HF_CN_ENDPOINT
        )
        log_ok(f"下载完成：{dest_dir}")
    except ImportError:
        log_fatal("未安装 huggingface_hub，请先 pip install huggingface_hub")
    except Exception as e:
        log_fatal(f"HF 镜像下载失败：{e}")
    _post_download(dest_dir)

def _post_download(dest_dir: Path):
    """下载完成后输出后续操作提示"""
    print(f"\n{Color.GREEN}====================================================={Color.END}")
    print(f"{Color.BOLD}   模型已下载至：{dest_dir}{Color.END}")
    print(f"{Color.BOLD}   B 机部署时，deploy_env.py 会自动建立 HF 缓存结构。{Color.END}")
    print(f"{Color.BOLD}   如需手动放置，目标路径为：{Color.END}")
    hf_snap = MODEL_ROOT / "hf_cache" / "hub" \
              / f"models--opendatalab--{MODEL_NAME}" / "snapshots" / "local"
    print(f"     {hf_snap}")
    print(f"{Color.GREEN}====================================================={Color.END}\n")
    input("按任意键退出...")

# ─────────────────────────────────────────────────────────
# Transformers 工具箱 (A 机)
# ─────────────────────────────────────────────────────────

def _run_transformers_validation(config: EnvConfig):
    """验证 Transformers 模型加载是否报错"""
    python_exe = config.ENV_DIR / "python.exe"
    if not python_exe.exists():
        log_fatal(f"环境不存在：{config.ENV_DIR}\n请先使用 [1] 全量构建环境。")
        return
        
    test_script = """
import sys
try:
    import transformers
    print(f"\\n当前 Transformers 版本: {transformers.__version__}")
    
    from transformers import AutoProcessor, AutoTokenizer
    from pathlib import Path
    
    # 查找模型路径 (兼容 A 机的直接下载路径和缓存路径)
    model_dir = Path(r'D:\\ai_offline_pack\\models')
    hf_path = model_dir / 'hf_cache' / 'hub' / 'models--opendatalab--MinerU2.5-Pro-2604-1.2B' / 'snapshots' / 'local'
    ms_path = model_dir / 'MinerU2.5-Pro-2604-1.2B'
    
    model_path = None
    if hf_path.exists(): model_path = hf_path
    elif ms_path.exists(): model_path = ms_path
    
    if not model_path:
        print("未找到模型文件！请先使用 [3] 下载模型。")
        sys.exit(1)
        
    print(f"正在验证模型加载 ({model_path.name})...\\n")
    
    has_error = False
    try:
        AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        print("  [OK] Tokenizer 加载成功！没有发现字典访问 Bug。")
    except Exception as e:
        print(f"  [Error] Tokenizer 报错: {e}")
        has_error = True
        
    try:
        AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
        print("  [OK] Processor 加载成功！")
    except Exception as e:
        print(f"  [Error] Processor 报错: {e}")
        has_error = True
        
    if has_error:
        print("\\n  >> 验证失败: 发现加载错误。如果你使用的是 4.57.x，请使用一键修复补丁。")
    else:
        print("\\n  >> 验证通过: 完美！")

except ImportError:
    print("未安装 transformers")
"""
    script_path = config.ENV_DIR / "temp_val.py"
    script_path.write_text(test_script, encoding="utf-8")
    run_cmd([str(python_exe), str(script_path)], check=False)
    script_path.unlink(missing_ok=True)

def _patch_transformers_bug(config: EnvConfig):
    """专门修复 transformers==4.57.2 中 dict 没有 model_type 属性的 Bug"""
    python_exe = config.ENV_DIR / "python.exe"
    if not python_exe.exists():
        log_fatal(f"环境不存在：{config.ENV_DIR}\n请先使用 [1] 全量构建环境。")
        return
        
    tokenization_utils = config.ENV_DIR / "Lib" / "site-packages" / "transformers" / "tokenization_utils_base.py"
    
    if not tokenization_utils.exists():
        log_error(f"找不到 transformers 安装目录，可能还未安装: {tokenization_utils}")
        return
        
    import shutil
    from datetime import datetime
    
    content = tokenization_utils.read_text(encoding="utf-8")
    OLD = "if _is_local and _config.model_type not in ["
    NEW = "if _is_local and _config.get(\"model_type\") not in ["
    
    if NEW in content:
        log_ok("已经打过补丁，无需重复。")
    elif OLD in content:
        bak = tokenization_utils.with_suffix(f".py.bak_{datetime.now().strftime('%H%M%S')}")
        shutil.copy2(tokenization_utils, bak)
        content = content.replace(OLD, NEW)
        tokenization_utils.write_text(content, encoding="utf-8")
        log_ok(f"已备份原文件: {bak.name}")
        log_ok("补丁已注入: 成功修复 A 机环境中 _config.model_type 字典访问问题")
        log_warn("注意：这只修复了 A 机的环境！由于 B 机是解压 wheel 安装的，B 机部署后如果仍用 4.57.2，依然会遇到这个 Bug。")
    else:
        log_warn("未找到匹配的缺陷代码。可能当前不是 transformers 4.57.2，或代码已被修复。")

def _update_transformers_version(config: EnvConfig):
    version = input("\n请输入你想更新的 transformers 版本 (例如 4.48.3, 4.56.2, 4.57.2 等): ").strip()
    if not version:
        return
        
    log_info(f"\n=" * 50)
    log_info(f"正在将 A 机环境的 transformers 更新至 {version} ...")
    log_info(f"=" * 50)
    
    python_exe = config.ENV_DIR / "python.exe"
    if not python_exe.exists():
        log_fatal("环境尚未构建，请先在主菜单使用 [1] 全量构建环境。")
        return

    # 1. 安装并更新 A 机的 env_mineru
    res = run_cmd([
        str(python_exe), "-m", "pip", "install", f"transformers=={version}",
        "--index-url", MAIN_INDEX,
        "--extra-index-url", EXTRA_INDEX_1,
        "--extra-index-url", EXTRA_INDEX_2
    ])
    
    if res.returncode != 0:
        log_error("安装失败！请检查版本号或网络连接。")
        return
        
    # 2. 更新 requirements_mineru.txt
    req_file = ROOT_DIR / f"requirements_{config.prefix.lower()}.txt"
    if req_file.exists():
        content = req_file.read_text(encoding="utf-8")
        import re
        new_content = re.sub(r'transformers==[0-9\.]+', f'transformers=={version}', content)
        if new_content == content and 'transformers' not in content:
             new_content += f"\ntransformers=={version}\n"
        req_file.write_text(new_content, encoding="utf-8")
        log_ok(f"已同步更新基础配置文件：{req_file.name}")
        
    # 3. 重新下载相关的 wheels 到 WHEEL_DIR (保证离线包完整)
    log_info("正在同步下载对应的离线 Wheel 包...")
    run_cmd([
        str(python_exe), "-m", "pip", "download", f"transformers=={version}",
        "-d", str(config.WHEEL_DIR),
        "--index-url", MAIN_INDEX,
        "--extra-index-url", EXTRA_INDEX_1,
        "--extra-index-url", EXTRA_INDEX_2
    ])
    
    # 4. 更新 A 机的 lock 文件
    import subprocess
    freeze_out = subprocess.run([str(python_exe), "-m", "pip", "freeze"], capture_output=True, text=True, check=True).stdout
    config.LOCK_FILE.write_text(freeze_out, encoding="utf-8")
    log_ok(f"已更新环境锁定文件：{config.LOCK_FILE.name}")
    
    # 5. 更新 patch_config_mineru.json，方便用户直接打补丁
    if config.PATCH_CONFIG_FILE.exists():
        try:
            import json
            patch_data = json.loads(config.PATCH_CONFIG_FILE.read_text(encoding="utf-8"))
            patch_data["upgrade"] = [f"transformers=={version}"]
            config.PATCH_CONFIG_FILE.write_text(json.dumps(patch_data, indent=4, ensure_ascii=False), encoding="utf-8")
            log_ok(f"已自动将升级目标写入 {config.PATCH_CONFIG_FILE.name}")
        except Exception as e:
            pass
    
    print(f"\n{Color.GREEN}====================================================={Color.END}")
    print(f"{Color.BOLD} A 机的 Transformers 及其依赖已成功更新至 {version}！{Color.END}")
    print(f"{Color.GREEN}====================================================={Color.END}")
    print(f"你可以立刻使用工具箱的 [1] 验证版本是否加载正常。")
    print(f"\n{Color.YELLOW}【重要】如何同步至 B 机？{Color.END}")
    print(f"如果你想在 B 机也应用这个版本更新，请执行：")
    print(f" 1. 退回最外层主菜单，选择 {Color.BOLD}[2] 增量补丁包{Color.END}")
    print(f" 2. 当询问更新包时，选择输入 {Color.BOLD}y{Color.END} (读取刚才自动修改的 patch_config)")
    print(f" 3. 生成 ZIP 后，拷给 B 机安装即可！")

def transformers_toolkit(config: EnvConfig):
    """Transformers 专用工具箱"""
    while True:
        print(f"\n{Color.CYAN}====================================================={Color.END}")
        print(f"{Color.BOLD}   Transformers 工具箱 (A 机) (仅针对 MinerU){Color.END}")
        print(f"{Color.CYAN}====================================================={Color.END}")
        print(f"  {Color.BOLD}[1]{Color.END} transformers 版本验证 (检测是否有 dict.model_type Bug)")
        print(f"  {Color.BOLD}[2]{Color.END} transformers 一键打补丁 (修复 A 机的 4.57.2 Bug)")
        print(f"  {Color.BOLD}[3]{Color.END} 更新 transformers 到指定版本 (自动化升降级)")
        print(f"  {Color.BOLD}[0]{Color.END} 返回上一级菜单")
        
        choice = input(f"\n  请输入选项 (0/1/2/3) [默认 1]: ").strip() or "1"
        
        if choice == "1":
            _run_transformers_validation(config)
        elif choice == "2":
            log_info("应用 Transformers Bug 热修复补丁...")
            _patch_transformers_bug(config)
        elif choice == "3":
            _update_transformers_version(config)
        elif choice == "0":
            break
        else:
            log_warn("无效选项")
        
        input("\n按任意键继续...")


def main():
    print(f"{Color.BOLD}========================================{Color.END}")
    print(f"{Color.BOLD}   AI 离线环境构建工具 (A 机){Color.END}")
    print(f"{Color.BOLD}========================================{Color.END}\n")
    
    print(f"{Color.CYAN}第一步：请选择要构建的工作环境{Color.END}")
    print(f"  {Color.BOLD}[1]{Color.END} 构建大模型工作环境 (env_paper)")
    print(f"  {Color.BOLD}[2]{Color.END} 构建MinerU工作环境 (env_mineru)")
    
    env_choice = input(f"\n  请输入选项 (1/2) [默认 1]: ").strip()
    
    target_config = ENV_MINERU_CONFIG if env_choice == "2" else ENV_PAPER_CONFIG
    
    print(f"\n{Color.GREEN}已选择：{target_config.name}{Color.END}")
    print(f"环境目录：{target_config.ENV_DIR}")
    print(f"Wheels 目录：{target_config.WHEEL_DIR}")
    print(f"全量产出：{target_config.EXPORT_DIR}")
    print(f"补丁产出：{target_config.PATCH_DIR}")
    print("-" * 40)
    
    print(f"{Color.CYAN}第二步：请选择构建模式{Color.END}")
    print(f"  {Color.BOLD}[1]{Color.END} 全量构建环境 - 删除旧环境重新安装，下载所有依赖并打包为 ZIP (最稳定)")
    print(f"  {Color.BOLD}[2]{Color.END} 增量补丁包   - 根据 {target_config.PATCH_CONFIG_FILE.name} 生成 ZIP 补丁包 (最快)")
    if target_config.prefix == "MinERU":
        print(f"  {Color.BOLD}[3]{Color.END} 重新下载模型 - 通过国内镜像下载 VLM 模型到 {MODEL_ROOT} (仅 MinerU)")
        print(f"  {Color.BOLD}[4]{Color.END} Transformers 工具箱 - (版本验证 / 一键打补丁 / 生成更新补丁包)")

    prompt = "(1/2/3/4)" if target_config.prefix == "MinERU" else "(1/2)"
    mode_choice = input(f"\n  请输入选项 {prompt} [默认 1]: ").strip()

    if mode_choice == "2":
        build_patch_package(target_config)
    elif mode_choice == "3" and target_config.prefix == "MinERU":
        download_model_only()
    elif mode_choice == "4" and target_config.prefix == "MinERU":
        transformers_toolkit(target_config)
    else:
        # 二次确认全量构建
        if target_config.ENV_DIR.exists():
            confirm = input(f"\n{Color.YELLOW}[警告]{Color.END} 即将删除现有环境 {target_config.ENV_DIR} 并重建。确认？(y/n): ").strip().lower()
            if confirm != 'y':
                log_info("操作已取消。")
                return
        build_full_package(target_config)

if __name__ == "__main__":
    main()
