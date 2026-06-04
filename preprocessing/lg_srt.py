"""Parse SymLG (.lg) files into symbol relation trees (SRT)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class SRTNode:
    """One symbol node with spatial relations to child symbols."""

    node_id: str
    label: str
    children: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class SymbolRelationTree:
    """Symbol-level relation tree extracted from a SymLG file."""

    header: str
    nodes: Dict[str, SRTNode]
    roots: List[str]


def _normalize_relation(name: str) -> str:
    aliases = {
        "R": "Right",
        "HOR": "Right",
        "SUP": "Sup",
        "SUB": "Sub",
        "A": "Above",
        "ABOVE": "Above",
        "B": "Below",
        "BELOW": "Below",
        "I": "Inside",
    }
    return aliases.get(name, name)


def parse_lg_text(text: str) -> SymbolRelationTree:
    """Parse SymLG text (``O`` object + ``R`` relation lines) into an SRT."""
    header = ""
    nodes: Dict[str, SRTNode] = {}
    object_order: List[str] = []
    children_of: Dict[str, List[str]] = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            if line.startswith("# IUD"):
                header = line[2:].strip()
            continue

        parts = [part.strip() for part in line.split(",")]
        if not parts:
            continue

        kind = parts[0]
        if kind == "O" and len(parts) >= 3:
            node_id, label = parts[1], parts[2]
            if node_id not in nodes:
                object_order.append(node_id)
            nodes[node_id] = SRTNode(node_id=node_id, label=label)
        elif kind == "R" and len(parts) >= 4:
            parent_id, child_id, relation = parts[1], parts[2], _normalize_relation(parts[3])
            if parent_id not in nodes:
                nodes[parent_id] = SRTNode(node_id=parent_id, label="")
            if child_id not in nodes:
                nodes[child_id] = SRTNode(node_id=child_id, label="")
            nodes[parent_id].children.setdefault(relation, []).append(child_id)
            children_of.setdefault(child_id, []).append(parent_id)

    roots = [node_id for node_id in object_order if node_id not in children_of]
    if not roots:
        roots = list(object_order)

    return SymbolRelationTree(header=header, nodes=nodes, roots=roots)


def parse_lg_file(path: Path) -> SymbolRelationTree:
    """Load and parse a ``.lg`` file into a :class:`SymbolRelationTree`."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return parse_lg_text(text)


def lg_to_srt(lg_path: Path) -> SymbolRelationTree:
    """Convert a label graph file to a symbol relation tree."""
    return parse_lg_file(lg_path)
