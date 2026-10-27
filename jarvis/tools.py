"""Registre des outils autorisés.

Un agent ne peut appeler un outil que s'il est (1) enregistré, (2) activé
dans la configuration et (3) dans la liste de l'agent. Chaque appel est
journalisé. Les outils fichiers sont enfermés dans le dossier workspace/,
et il n'y a volontairement aucun outil qui lance des commandes système.
"""

import ast
import datetime as dt
import ipaddress
import math
import operator
import re
import socket
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

MAX_OUTPUT = 6000


class ToolError(Exception):
    pass


@dataclass
class Tool:
    name: str
    description: str
    params: dict[str, str]          # nom -> description
    func: Callable[..., str]
    risk: str = "lecture"           # lecture | écriture | réseau
    required: tuple[str, ...] = field(default_factory=tuple)


class ToolRegistry:
    def __init__(self, disabled: set[str] | None = None):
        self._tools: dict[str, Tool] = {}
        self.disabled = set(disabled or ())

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def enabled(self, names) -> list[Tool]:
        return [self._tools[n] for n in names if n in self._tools and n not in self.disabled]

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def catalog(self, names) -> str:
        lines = []
        for t in self.enabled(names):
            params = ", ".join(f'"{k}": {v}' for k, v in t.params.items()) or "aucun"
            lines.append(f"- {t.name} : {t.description} Arguments : {{{params}}}")
        return "\n".join(lines) or "(aucun outil)"

    def call(self, name: str, args, allowed) -> str:
        if name not in self._tools:
            raise ToolError(f"outil inconnu : {name}")
        if name in self.disabled:
            raise ToolError(f"outil désactivé : {name}")
        if name not in allowed:
            raise ToolError(f"outil non autorisé pour cet agent : {name}")
        tool = self._tools[name]
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise ToolError("les arguments doivent être un objet JSON")
        unknown = set(args) - set(tool.params)
        if unknown:
            raise ToolError(f"arguments inconnus : {', '.join(sorted(unknown))}")
        missing = [p for p in tool.required if p not in args]
        if missing:
            raise ToolError(f"arguments manquants : {', '.join(missing)}")
        try:
            out = tool.func(**args)
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc
        out = str(out)
        if len(out) > MAX_OUTPUT:
            out = out[:MAX_OUTPUT] + f"\n…[tronqué, {len(out)} caractères au total]"
        return out


# --- outils intégrés ---------------------------------------------------

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_FUNCS = {n: getattr(math, n) for n in ("sqrt", "log", "log10", "exp", "sin", "cos", "tan", "floor", "ceil")}
_FUNCS.update(abs=abs, round=round, min=min, max=max)
_CONSTS = {"pi": math.pi, "e": math.e}


def calculate(expression: str) -> str:
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name) and node.id in _CONSTS:
            return _CONSTS[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 1000:
                raise ToolError("exposant trop grand")
            return _OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise ToolError("expression non autorisée")

    if len(expression) > 500:
        raise ToolError("expression trop longue")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        raise ToolError("expression invalide (utilise le point pour les décimales)")
    return str(ev(tree))


def now() -> str:
    return dt.datetime.now().astimezone().strftime("%A %d %B %Y, %H:%M:%S (%Z)")


class Workspace:
    """Lecture/écriture limitées à un seul dossier."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if p != self.root and self.root not in p.parents:
            raise ToolError("chemin hors du workspace")
        return p

    def read(self, path: str) -> str:
        p = self._path(path)
        if not p.is_file():
            raise ToolError(f"fichier introuvable : {path}")
        if p.stat().st_size > 500_000:
            raise ToolError("fichier trop gros (500 Ko max)")
        return p.read_text(encoding="utf-8", errors="replace")

    def write(self, path: str, content: str) -> str:
        p = self._path(path)
        if len(content) > 500_000:
            raise ToolError("contenu trop gros (500 Ko max)")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"écrit : {p.relative_to(self.root)} ({len(content)} caractères)"

    def list(self, path: str = ".") -> str:
        p = self._path(path)
        if not p.is_dir():
            raise ToolError(f"dossier introuvable : {path}")
        items = sorted(p.iterdir())[:200]
        return "\n".join(f"{'[d]' if i.is_dir() else '   '} {i.relative_to(self.root)}" for i in items) or "(vide)"


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript", "svg"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript", "svg") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip and data.strip():
            self.parts.append(data.strip())


def _public_host(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        raise ToolError(f"hôte introuvable : {host}")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if not ip.is_global:
            return False
    return True


def fetch_url(url: str) -> str:
    import httpx

    current = url
    for _ in range(4):  # suit au plus 3 redirections, en revérifiant l'hôte à chaque fois
        parsed = urlparse(current)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ToolError("seules les URL http(s) publiques sont autorisées")
        if not _public_host(parsed.hostname):
            raise ToolError("adresse privée ou locale refusée")
        with httpx.Client(timeout=15, follow_redirects=False,
                          headers={"User-Agent": "Jarvis/0.1"}) as client:
            with client.stream("GET", current) as resp:
                if resp.is_redirect:
                    current = str(resp.url.join(resp.headers.get("location", "")))
                    continue
                body = b""
                for chunk in resp.iter_bytes():
                    body += chunk
                    if len(body) > 1_000_000:
                        break
                ctype = resp.headers.get("content-type", "")
                status = resp.status_code
        text = body.decode("utf-8", errors="replace")
        if "html" in ctype:
            ex = _TextExtractor()
            ex.feed(text)
            text = "\n".join(ex.parts)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return f"[HTTP {status}] {current}\n{text}"
    raise ToolError("trop de redirections")


def build_default_registry(cfg, store=None) -> ToolRegistry:
    reg = ToolRegistry(disabled=cfg.disabled_tools)
    ws = Workspace(cfg.workspace)
    reg.register(Tool("calculatrice", "Calcule une expression mathématique.",
                      {"expression": "texte, ex. '(12*3)/4'"}, calculate, required=("expression",)))
    reg.register(Tool("heure", "Donne la date et l'heure actuelles.", {}, now))
    reg.register(Tool("lire_fichier", "Lit un fichier texte du workspace.",
                      {"path": "chemin relatif"}, ws.read, required=("path",)))
    reg.register(Tool("lister_fichiers", "Liste un dossier du workspace.",
                      {"path": "chemin relatif (défaut '.')"}, ws.list))
    reg.register(Tool("ecrire_fichier", "Écrit (ou remplace) un fichier texte dans le workspace.",
                      {"path": "chemin relatif", "content": "texte"}, ws.write,
                      risk="écriture", required=("path", "content")))
    reg.register(Tool("lire_page_web", "Télécharge une page web publique et renvoie son texte.",
                      {"url": "URL http(s)"}, fetch_url, risk="réseau", required=("url",)))
    if store is not None:
        def search_memory(query: str) -> str:
            hits = store.search(query)
            if not hits:
                return "Aucune mission passée ne correspond."
            return "\n".join(
                f"- [{h['type']} #{h['mission_id']}] " + (h.get("objective") or h.get("text", ""))
                + (f" → {h['result'][:300]}" if h.get("result") else "")
                for h in hits
            )
        reg.register(Tool("memoire_missions", "Cherche dans les missions passées et leurs leçons.",
                          {"query": "mots-clés"}, search_memory, required=("query",)))
    return reg
