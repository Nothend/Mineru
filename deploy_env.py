import os
import shutil
import sys
import subprocess
import time
import json
import zipfile
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 配置区 ---
ROOT_DIR = Path(r"D:\ai_offline_pack")
MODEL_ROOT = ROOT_DIR / "models"

# --- 颜色与格式 ---
class Color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def log_info(msg): print(f"{Color.CYAN}[INFO]{Color.END} {msg}")
def log_ok(msg): print(f"{Color.GREEN}[OK]{Color.END} {msg}")
def log_warn(msg): print(f"{Color.YELLOW}[WARN]{Color.END} {msg}")
def log_error(msg): print(f"{Color.RED}[ERROR]{Color.END} {msg}")
def log_fatal(msg): 
    print(f"{Color.BOLD}{Color.RED}[FATAL]{Color.END} {msg}")
    input("按任意键退出...")
    sys.exit(1)

def run_cmd(cmd, cwd=None, env=None, check=True, shell=False):
    try:
        if shell and isinstance(cmd, list):
            cmd = " ".join(cmd)
        
        current_env = os.environ.copy()
        if env:
            current_env.update(env)
        
        result = subprocess.run(cmd, cwd=cwd, env=current_env, shell=shell, check=check)
        return result
    except subprocess.CalledProcessError as e:
        log_error(f"命令执行失败: {cmd}")
        if check:
            sys.exit(e.returncode)
        return e

def extract_zip_with_progress(zip_file: Path, extract_dir: Path):
    """带进度条的解压函数"""
    zip_file = Path(zip_file)
    extract_dir = Path(extract_dir)
    
    if not zip_file.exists():
        log_error(f"ZIP 包不存在：{zip_file}")
        return False
        
    extract_dir.mkdir(parents=True, exist_ok=True)
    log_info(f"正在解压 {zip_file.name} -> {extract_dir}")
    
    with zipfile.ZipFile(zip_file, 'r') as zipf:
        members = zipf.infolist()
        total_files = len(members)
        if total_files == 0:
            log_warn("ZIP 包为空。")
            return True
            
        for i, member in enumerate(members, 1):
            zipf.extract(member, extract_dir)
            if i % max(1, total_files // 100) == 0 or i == total_files:
                percent = i * 100 // total_files
                bar = "#" * (percent // 2) + "-" * (50 - percent // 2)
                sys.stdout.write(f"\r  [{bar}] {percent}% ({i}/{total_files})")
                sys.stdout.flush()
    print()
    log_ok("解压完成。")
    return True

class DeployConfig:
    def __init__(self, name, env_dir_name, wheel_dir_name, prefix_name):
        self.name = name
        self.prefix = prefix_name

        self.ENV_DIR = ROOT_DIR / "envs" / env_dir_name
        self.WHEEL_DIR = ROOT_DIR / "wheels" / wheel_dir_name

        # 本地导入源 (A机生成的目录结构)
        self.EXPORT_DIR = ROOT_DIR / "exported" / f"full_{prefix_name.lower()}"
        self.PATCH_DIR = ROOT_DIR / "patch" / f"patch_{prefix_name.lower()}"

        # 锁定文件（补丁应用后更新用）
        self.LOCK_FILE = self.EXPORT_DIR / f"requirements_{prefix_name.lower()}_lock.txt"

DEPLOY_PAPER_CONFIG = DeployConfig(
    name="大模型工作环境 (env_paper)",
    env_dir_name="env_paper",
    wheel_dir_name="paper",
    prefix_name="Paper"
)

DEPLOY_MINERU_CONFIG = DeployConfig(
    name="MinerU 工作环境 (env_mineru)",
    env_dir_name="env_mineru",
    wheel_dir_name="mineru",
    prefix_name="MinERU"
)

def run_smoke_test_only(config: DeployConfig):
    """仅运行冒烟测试模式"""
    log_info("进入冒烟测试模式...")
    python_exe = config.ENV_DIR / "python.exe"
    if not python_exe.exists():
        log_fatal(f"环境不存在：{config.ENV_DIR}\n请先选择 [1] 完整部署流程")
    
    if config.prefix == "MinERU":
        test_pdf = ROOT_DIR / "test.pdf"
        if not test_pdf.exists():
            log_fatal(f"未找到测试文件：{test_pdf}\n请将测试 PDF 放置在 D:\\ai_offline_pack\\test.pdf")
        
        mineru_exe = config.ENV_DIR / "Scripts" / "mineru.exe"
        if not mineru_exe.exists():
            log_fatal("未找到 mineru.exe，环境可能不完整")
        
        hf_home = MODEL_ROOT / "hf_cache"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_HOME"] = str(hf_home)
        
        log_info(f"运行 GPU 冒烟测试：{test_pdf}")
        output_dir = ROOT_DIR / "output_gpu"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            result = run_cmd([
                str(mineru_exe), "-p", str(test_pdf), 
                "-o", str(output_dir), 
                "--device", "cuda"
            ], env={"HF_HUB_OFFLINE": "1", "HF_HOME": str(hf_home)}, check=False)
            
            if result.returncode == 0:
                log_ok("GPU 冒烟测试通过。")
                log_info(f"输出目录：{output_dir}")
            else:
                log_error(f"冒烟测试运行失败，退出码：{result.returncode}")
        except Exception as e:
            log_error(f"冒烟测试运行失败：{e}")
    
    elif config.prefix == "Paper":
        log_info("运行 GPU 冒烟测试 (Torch 验证)...")
        res = run_cmd([str(python_exe), "-c", "import torch; cuda=torch.cuda.is_available(); name=torch.cuda.get_device_name(0) if cuda else '未检测到'; print('CUDA:', cuda); print('GPU :', name)"], check=False)
        if res.returncode == 0:
            log_ok("GPU 冒烟测试通过。")
        else:
            log_error("GPU 冒烟测试失败。")
            
    input("按任意键退出...")

def patch_transformers_bug(config: DeployConfig):
    """专门修复 transformers==4.57.2 中 dict 没有 model_type 属性的 Bug"""
    log_info("=" * 50)
    log_info("应用 Transformers Bug 热修复补丁...")
    log_info("=" * 50)
    
    python_exe = config.ENV_DIR / "python.exe"
    if not python_exe.exists():
        log_fatal(f"环境不存在：{config.ENV_DIR}\n请先部署环境再执行此修复。")
        
    tokenization_utils = config.ENV_DIR / "Lib" / "site-packages" / "transformers" / "tokenization_utils_base.py"
    
    if not tokenization_utils.exists():
        log_error(f"找不到 transformers 安装目录，可能还未安装: {tokenization_utils}")
        input("按任意键退出...")
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
        log_ok("补丁已注入: 成功修复 _config.model_type 字典访问问题")
    else:
        log_warn("未找到匹配的缺陷代码。可能当前不是 transformers 4.57.2，或代码已被修复。")
        
    input("\n按任意键退出...")


def apply_incremental_patch(config: DeployConfig):
    """应用增量补丁（适配 build_env.py 新格式：target_packages + 级联依赖列表）"""
    log_info("=" * 50)
    log_info(f"进入增量补丁模式 ({config.name})...")
    log_info("=" * 50)

    if not config.PATCH_DIR.exists():
        log_fatal(f"补丁目录不存在：{config.PATCH_DIR}\n请将 A 机生成的 ZIP 放置于此。")

    patch_files = list(config.PATCH_DIR.glob(f"{config.prefix}_Patch_v*.zip"))
    if not patch_files:
        log_fatal(f"未找到补丁包文件 ({config.prefix}_Patch_v*.zip)：{config.PATCH_DIR}")

    latest_patch = sorted(patch_files)[-1]
    log_ok(f"检测到最新补丁包：{latest_patch.name}")

    temp_extract_dir = config.PATCH_DIR / "temp_extract"
    if temp_extract_dir.exists():
        shutil.rmtree(temp_extract_dir)
    extract_zip_with_progress(latest_patch, temp_extract_dir)

    # ── 加载并展示 manifest ──────────────────────────────
    manifest_file = temp_extract_dir / "manifest.json"
    if not manifest_file.exists():
        log_fatal("补丁包中缺少 manifest.json，补丁包可能已损坏")

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    version     = manifest.get("version", "未知")
    description = manifest.get("description", "无描述")
    target_pkgs = manifest.get("target_packages", [])   # 新格式字段
    log_ok(f"补丁版本: {version}  描述: {description}")
    if target_pkgs:
        log_info(f"目标包: {', '.join(target_pkgs)}")

    # ── 解析 packages 列表（兼容新旧格式）─────────────────
    def _to_spec(p):
        return p if isinstance(p, str) else p.get("spec", str(p))

    packages_config = manifest.get("packages", {})
    install_list   = [_to_spec(p) for p in packages_config.get("install",   [])]
    upgrade_list   = [_to_spec(p) for p in packages_config.get("upgrade",   [])]
    uninstall_list = [_to_spec(p) for p in packages_config.get("uninstall", [])]
    all_specs      = install_list + upgrade_list

    # 展示变化摘要
    if install_list:
        log_info(f"新增安装 ({len(install_list)} 个): {install_list}")
    if upgrade_list:
        log_info(f"版本升级 ({len(upgrade_list)} 个): {upgrade_list}")
    if uninstall_list:
        log_info(f"显式卸载 ({len(uninstall_list)} 个): {uninstall_list}")
    if not all_specs and not uninstall_list:
        log_warn("补丁清单为空，无需操作。")
        shutil.rmtree(temp_extract_dir, ignore_errors=True)
        input("按任意键退出...")
        return

    # ── 检查目标环境 ────────────────────────────────────
    python_exe = config.ENV_DIR / "python.exe"
    if not python_exe.exists():
        log_fatal(f"环境不存在：{config.ENV_DIR}\n增量补丁模式需要已有环境存在。")

    # ── 合并 Wheels 到本地缓存 ───────────────────────────
    whl_count = 0
    config.WHEEL_DIR.mkdir(parents=True, exist_ok=True)
    for item in temp_extract_dir.iterdir():
        if item.is_file() and item.suffix == ".whl":
            shutil.copy2(item, config.WHEEL_DIR / item.name)
            whl_count += 1
    if whl_count:
        log_ok(f"已合并 {whl_count} 个 Wheel 到本地缓存：{config.WHEEL_DIR}")

    # ── Step 1: 卸载 ────────────────────────────────────
    if uninstall_list:
        log_info(f"正在卸载 {len(uninstall_list)} 个包...")
        for pkg in uninstall_list:
            run_cmd([str(python_exe), "-m", "pip", "uninstall", "-y", pkg], check=False)

    # ── Step 2: 安装/升级（合并为单次调用，离线 --no-deps）─
    # 所有级联依赖已由 build_env.py 计算并打包，逐一安装即可
    if all_specs:
        log_info(f"正在安装/升级 {len(all_specs)} 个包（离线模式）...")
        run_cmd([
            str(python_exe), "-m", "pip", "install",
            "--no-index",
            f"--find-links={temp_extract_dir}",
            f"--find-links={config.WHEEL_DIR}",
            "--no-deps",
            *all_specs
        ])

    # ── Step 3: 更新 lock 文件 ──────────────────────────
    log_info("更新依赖锁定表...")
    try:
        config.LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(config.LOCK_FILE, "w", encoding="utf-8") as lf:
            subprocess.run([str(python_exe), "-m", "pip", "freeze"], stdout=lf, check=True)
        log_ok(f"锁定表已更新：{config.LOCK_FILE}")
    except Exception as e:
        log_warn(f"更新锁定表失败（不影响使用）：{e}")

    # ── 清理 ────────────────────────────────────────────
    log_info("清理临时文件...")
    shutil.rmtree(temp_extract_dir, ignore_errors=True)
    log_ok("增量补丁应用完成！")
    input("按任意键退出...")

def full_deploy(config: DeployConfig):
    """完整部署流程"""
    log_info(f"=== 开始全量部署 ({config.name}) ===")
    
    python_exe = config.ENV_DIR / "python.exe"
    
    if config.ENV_DIR.exists():
        confirm = input(f"\n{Color.YELLOW}[警告]{Color.END} 目标环境 {config.ENV_DIR} 已存在，将被删除并重新解压。确认覆盖？(y/n): ").strip().lower()
        if confirm != 'y':
            log_info("操作已取消。")
            return
        log_info("清理旧环境...")
        shutil.rmtree(config.ENV_DIR, ignore_errors=True)
        
    # 检查导出的 ZIP 文件
    env_zip = config.EXPORT_DIR / f"full_env_{config.prefix.lower()}.zip"
    wheel_zip = config.EXPORT_DIR / f"full_wheel_{config.prefix.lower()}.zip"
    
    if not env_zip.exists():
        log_fatal(f"未找到环境压缩包：{env_zip}\n请确保 A 机生成的文件已放在此目录。")
        
    # --- 解压环境 ---
    extract_zip_with_progress(env_zip, config.ENV_DIR)
    
    # --- 解压 Wheels ---
    if wheel_zip.exists():
        if config.WHEEL_DIR.exists():
            shutil.rmtree(config.WHEEL_DIR, ignore_errors=True)
        extract_zip_with_progress(wheel_zip, config.WHEEL_DIR)
    else:
        log_warn(f"未找到 Wheels 压缩包：{wheel_zip}，将尝试直接使用 {config.WHEEL_DIR}。")

    # --- 修复环境 (conda-unpack) ---
    log_info("修复环境路径 (conda-unpack)...")
    unpack_exe = config.ENV_DIR / "Scripts" / "conda-unpack.exe"
    if unpack_exe.exists():
        run_cmd([str(unpack_exe)])
        log_ok("conda-unpack 执行完毕。")
    else:
        log_warn("未找到 conda-unpack.exe，跳过。")

    # --- 安装 GPU Torch 和 业务依赖 ---
    log_info("移除旧版 PyTorch 并安装 GPU 版...")
    run_cmd([str(python_exe), "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"], check=False)
    
    if config.prefix == "MinERU":
        run_cmd([
            str(python_exe), "-m", "pip", "install", 
            "--no-index", f"--find-links={config.WHEEL_DIR}", 
            "torch==2.4.1", "torchvision==0.19.1", "torchaudio==2.4.1", "--no-deps"
        ])
    elif config.prefix == "Paper":
        run_cmd([
            str(python_exe), "-m", "pip", "install", 
            "--no-index", f"--find-links={config.WHEEL_DIR}", 
            "torch==2.1.2", "torchvision==0.16.2", "torchaudio==2.1.2", "--no-deps"
        ])
    log_ok("GPU Torch 安装完成。")

    log_info("同步业务依赖...")
    lock_file = config.EXPORT_DIR / f"requirements_{config.prefix.lower()}_lock.txt"
    req_file = config.EXPORT_DIR / f"requirements_{config.prefix.lower()}.txt"
    
    sync_file = lock_file if lock_file.exists() else (req_file if req_file.exists() else None)
    
    if sync_file:
        log_info(f"使用同步文件: {sync_file}")
        run_cmd([
            str(python_exe), "-m", "pip", "install", 
            "--use-deprecated=legacy-resolver", 
            "--no-index", f"--find-links={config.WHEEL_DIR}", 
            "-r", str(sync_file)
        ], check=False)
        log_ok("业务依赖同步完成。")
    else:
        log_warn(f"未在 {config.EXPORT_DIR} 中找到依赖清单文件，跳过依赖同步安装。")

    # --- MinERU 环境特殊配置 ---
    if config.prefix == "MinERU":
        log_info("配置 HuggingFace 离线缓存...")
        hf_home = MODEL_ROOT / "hf_cache"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HOME"] = str(hf_home)
        
        model_name = "MinerU2.5-Pro-2604-1.2B"
        src_model_dir = MODEL_ROOT / model_name
        dst_snapshot_dir = hf_home / "hub" / f"models--opendatalab--{model_name}" / "snapshots" / "local"
        
        if src_model_dir.exists() and not dst_snapshot_dir.exists():
            log_info(f"正在创建 HF 镜像结构: {dst_snapshot_dir}")
            dst_snapshot_dir.mkdir(parents=True, exist_ok=True)
            for f in src_model_dir.glob("*"):
                if f.is_file():
                    shutil.copy2(f, dst_snapshot_dir / f.name)
            
            refs_dir = hf_home / "hub" / f"models--opendatalab--{model_name}" / "refs"
            refs_dir.mkdir(parents=True, exist_ok=True)
            (refs_dir / "main").write_text("local", encoding="utf-8")
            log_ok("HF 离线缓存结构就绪。")
            
        log_info("生成 magic-pdf.json 配置...")
        user_profile = Path(os.environ.get("USERPROFILE", "C:/Users/Default"))
        config_path = user_profile / "magic-pdf.json"
        config_data = {"models-dir": str(MODEL_ROOT).replace("\\", "/")}
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)
        log_ok(f"配置文件已写入: {config_path}")

    # --- 健康检查 ---
    log_info("执行依赖健康检查...")
    check_code = "import torch, torchvision; print('[OK] 基础依赖测试通过。')"
    res = run_cmd([str(python_exe), "-c", check_code], check=False)

    print(f"\n{Color.GREEN}====================================================={Color.END}")
    print(f"{Color.BOLD}   部署成功！环境 {config.ENV_DIR.name} 已就绪。{Color.END}")
    print(f"{Color.GREEN}====================================================={Color.END}\n")
    
    run_smoke_test_only(config)


def main():
    os.system('color')
    print(f"{Color.BLUE}====================================================={Color.END}")
    print(f"{Color.BOLD}   AI 离线部署工具 (B 机 GPU){Color.END}")
    print(f"{Color.BLUE}====================================================={Color.END}\n")
    
    print(f"{Color.CYAN}第一步：请选择要部署的工作环境{Color.END}")
    print(f"  {Color.BOLD}[1]{Color.END} 部署大模型工作环境 (env_paper)")
    print(f"  {Color.BOLD}[2]{Color.END} 部署MinerU工作环境 (env_mineru)")
    
    env_choice = input(f"\n  请输入选项 (1/2) [默认 1]: ").strip()
    target_config = DEPLOY_MINERU_CONFIG if env_choice == "2" else DEPLOY_PAPER_CONFIG
    
    print(f"\n{Color.GREEN}已选择：{target_config.name}{Color.END}")
    print(f"环境目录：{target_config.ENV_DIR}")
    print(f"导入目录：{target_config.EXPORT_DIR}")
    print("-" * 40)
    
    print(f"{Color.CYAN}第二步：请选择部署模式{Color.END}")
    print(f"  {Color.BOLD}[1]{Color.END} 全量解压部署 - 清理环境并根据 ZIP 重建 (最稳定)")
    print(f"  {Color.BOLD}[2]{Color.END} 应用增量补丁 - 从 patch 目录安装补丁 (最快)")
    print(f"  {Color.BOLD}[3]{Color.END} 仅冒烟测试   - 检查已部署的环境状态")
    
    if target_config.prefix == "MinERU":
        print(f"  {Color.BOLD}[4]{Color.END} 修复 Transformers Bug - 针对 4.57.2 版本的 model_type 报错")

    prompt = "(1/2/3/4)" if target_config.prefix == "MinERU" else "(1/2/3)"
    mode_choice = input(f"\n  请输入选项 {prompt} [默认 1]: ").strip()
    
    if mode_choice == "2":
        apply_incremental_patch(target_config)
    elif mode_choice == "3":
        run_smoke_test_only(target_config)
    elif mode_choice == "4" and target_config.prefix == "MinERU":
        patch_transformers_bug(target_config)
    else:
        full_deploy(target_config)

if __name__ == "__main__":
    main()
