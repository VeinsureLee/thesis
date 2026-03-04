# -*- coding: utf-8 -*-
"""加载 config 目录下 YAML 配置的模块。"""

from pathlib import Path
from typing import Any, Dict, List, Union

try:
    import yaml
except ImportError:
    yaml = None


def load_yml(path: Union[str, Path]) -> Dict[str, Any]:
    """
    加载单个 YAML 文件为字典。

    Args:
        path: 文件路径，相对项目根目录或绝对路径均可。

    Returns:
        解析后的字典；若文件为空或仅含注释则返回空字典。

    Raises:
        FileNotFoundError: 文件不存在。
        ImportError: 未安装 PyYAML（pip install pyyaml）。
        yaml.YAMLError: YAML 格式错误。
    """
    if yaml is None:
        raise ImportError("加载 YAML 需要安装 PyYAML: pip install pyyaml")

    path = Path(path)
    if not path.is_absolute():
        # 相对路径：优先相对于当前工作目录，若不存在则尝试相对于项目根
        if not path.exists():
            root = _find_project_root()
            if root is not None:
                path = root / path
    if not path.exists():
        raise FileNotFoundError(f"YAML 文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def load_stock_columns(yml_path: Union[str, Path] = None) -> Dict[str, Any]:
    """
    加载股票列定义配置（如 config/stock_columns.yml）。

    Args:
        yml_path: 配置路径。默认使用项目 config 下的 stock_columns.yml。

    Returns:
        包含 raw_market_data、adjusted_price_qfq、adjusted_price_hfq、
        derived、technical_indicators、all_columns 等键的配置字典。
    """
    if yml_path is None:
        root = _find_project_root()
        if root is None:
            raise FileNotFoundError("未找到项目根目录，请显式传入 yml_path")
        yml_path = root / "config" / "stock_columns.yml"
    return load_yml(yml_path)


def get_columns_by_category(
    config: Dict[str, Any], category: str
) -> List[str]:
    """
    从 load_stock_columns() 返回的配置中取出某类列名列表。

    Args:
        config: load_stock_columns() 的返回值。
        category: 类别键，如 "raw_market_data"、"technical_indicators"。

    Returns:
        该类别下的 columns 列表；若类别或 columns 不存在则返回空列表。
    """
    block = config.get(category)
    if not isinstance(block, dict):
        return []
    columns = block.get("columns")
    if isinstance(columns, list):
        return [c for c in columns if isinstance(c, str)]
    return []


def _find_project_root() -> Path:
    """从当前文件向上查找包含 config 目录的目录作为项目根。"""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "config").is_dir():
            return parent
    return None


__all__ = [
    "load_yml",
    "load_stock_columns",
    "get_columns_by_category",
]
