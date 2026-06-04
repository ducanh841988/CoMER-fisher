"""Convert symbol relation trees to space-separated LaTeX."""

import re
from typing import Dict, Iterable, List, Optional, Set

from preprocessing.lg_srt import SRTNode, SymbolRelationTree

FRACTION_LABELS = {"-", "horizontal-line"}
SQRT_LABELS = {"\\sqrt", "sqrt"}
NUMERIC_LABEL_RE = re.compile(
    r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$"
)

DEFAULT_SYMBOL_MAP: Dict[str, str] = {
    "cos": "\\cos",
    "sin": "\\sin",
    "tan": "\\tan",
    "log": "\\log",
    "ln": "\\ln",
    "sqrt": "\\sqrt",
    "times": "\\times",
    "div": "\\div",
    "pi": "\\pi",
    "alpha": "\\alpha",
    "beta": "\\beta",
    "gamma": "\\gamma",
    "Delta": "\\Delta",
    "infty": "\\infty",
}


def _merge_tokens(tokens: Iterable[str]) -> str:
    return " ".join(token for token in tokens if token)


def _normalize_label(label: str, symbol_map: Dict[str, str]) -> str:
    cleaned = label.replace("COMMAT", ",")
    return symbol_map.get(cleaned, cleaned)


def _label_to_tokens(label: str, symbol_map: Dict[str, str]) -> List[str]:
    """Map a symbol label to one or more LaTeX tokens."""
    cleaned = label.replace("COMMAT", ",")
    if cleaned in symbol_map:
        return [symbol_map[cleaned]]
    if NUMERIC_LABEL_RE.fullmatch(cleaned):
        return list(cleaned)
    return [_normalize_label(label, symbol_map)]


def _right_child(node: SRTNode) -> Optional[str]:
    rights = node.children.get("Right", [])
    return rights[0] if rights else None


def _is_fraction(node: SRTNode) -> bool:
    return node.label in FRACTION_LABELS and "Above" in node.children and "Below" in node.children


def _is_sqrt(node: SRTNode) -> bool:
    return node.label in SQRT_LABELS and "Inside" in node.children


def _render_tokens(node_id: str, tree: SymbolRelationTree, symbol_map: Dict[str, str], visiting: Set[str]) -> List[str]:
    if node_id in visiting:
        return []

    node = tree.nodes[node_id]
    visiting = set(visiting) | {node_id}

    if _is_fraction(node):
        return [_render_fraction(node, tree, symbol_map, visiting)]
    if _is_sqrt(node):
        return [_render_sqrt(node, tree, symbol_map, visiting)]

    tokens = _label_to_tokens(node.label, symbol_map)

    for child_id in node.children.get("Sub", []):
        child_text = _render_subtree(child_id, tree, symbol_map, visiting)
        tokens.extend(["_", "{", *child_text.split(), "}"])

    for child_id in node.children.get("Sup", []):
        child_text = _render_subtree(child_id, tree, symbol_map, visiting)
        tokens.extend(["^", "{", *child_text.split(), "}"])

    return tokens


def _render_subtree(
    node_id: str,
    tree: SymbolRelationTree,
    symbol_map: Dict[str, str],
    visiting: Set[str],
) -> str:
    return _render_horizontal_chain(node_id, tree, symbol_map, visiting)


def _render_horizontal_chain(
    start_id: str,
    tree: SymbolRelationTree,
    symbol_map: Dict[str, str],
    visiting: Set[str],
) -> str:
    tokens: List[str] = []
    current: Optional[str] = start_id

    while current:
        if current in visiting:
            break
        node = tree.nodes[current]
        tokens.extend(_render_tokens(current, tree, symbol_map, visiting))
        visiting = set(visiting) | {current}
        current = _right_child(node)

    return _merge_tokens(tokens)


def _render_fraction(
    node: SRTNode,
    tree: SymbolRelationTree,
    symbol_map: Dict[str, str],
    visiting: Set[str],
) -> str:
    above_id = node.children["Above"][0]
    below_id = node.children["Below"][0]
    numerator = _render_horizontal_chain(above_id, tree, symbol_map, set(visiting))
    denominator = _render_horizontal_chain(below_id, tree, symbol_map, set(visiting))
    return _merge_tokens(
        ["\\frac", "{", *numerator.split(), "}", "{", *denominator.split(), "}"]
    )


def _render_sqrt(
    node: SRTNode,
    tree: SymbolRelationTree,
    symbol_map: Dict[str, str],
    visiting: Set[str],
) -> str:
    inside_tokens: List[str] = []
    for child_id in node.children.get("Inside", []):
        inside = _render_horizontal_chain(child_id, tree, symbol_map, set(visiting))
        inside_tokens.extend(inside.split())
    return _merge_tokens(["\\sqrt", "{", *inside_tokens, "}"])


def srt_to_latex(
    tree: SymbolRelationTree,
    symbol_map: Optional[Dict[str, str]] = None,
) -> str:
    """Render an SRT as space-separated LaTeX tokens."""
    symbol_map = {**DEFAULT_SYMBOL_MAP, **(symbol_map or {})}
    if not tree.roots:
        return ""

    parts = [
        _render_horizontal_chain(root_id, tree, symbol_map, set())
        for root_id in tree.roots
    ]
    return _merge_tokens(part for part in parts if part)
